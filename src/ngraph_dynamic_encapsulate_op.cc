/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include <cstdlib>
#include <mutex>
#include <utility>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include "ngraph_backend_manager.h"
#include "ngraph_builder.h"
#include "ngraph_cluster_manager.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_timer.h"
#include "ngraph_utils.h"

#include "ngraph/event_tracing.hpp"
#include "ngraph/runtime/backend.hpp"

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

// For each I/O tensor, cache TF's data ptr and nGraph's Tensor
/*using NgFunctionIOCache = std::unordered_map<
    std::shared_ptr<ngraph::runtime::Executable>,
    std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>>;*/
using NgFunctionIOCache =
    std::vector<std::pair<void*, shared_ptr<ng::runtime::Tensor>>>;

namespace ngraph_bridge {

REGISTER_OP("NGraphDynamicEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .Attr("ngraph_graph_id: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

class NGraphDynamicEncapsulateOp : public OpKernel {
 public:
  //---------------------------------------------------------------------------
  //  NGraphDynamicEncapsulateOp::ctor
  //---------------------------------------------------------------------------
  explicit NGraphDynamicEncapsulateOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        m_graph(OpRegistry::Global()),
        m_freshness_tracker(nullptr) {
    my_instance_id = s_instance_count;
    s_instance_count++;

    std::ostringstream oss;
    oss << "DynamicEncapsulate_" << my_instance_id << ": " << name();
    ngraph::Event event(oss.str(), name(), "");

    NGRAPH_VLOG(1) << "NGraphDynamicEncapsulateOp: " << my_instance_id
                   << " Name: " << name();

    GraphDef* graph_def;

    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &m_ngraph_cluster));
    graph_def = NGraphClusterManager::GetClusterGraph(m_ngraph_cluster);

    if (graph_def == nullptr) {
      string flib_key = "ngraph_cluster_" + to_string(m_ngraph_cluster);
      // Read graphdef from function library
      const FunctionLibraryDefinition flib =
          *ctx->function_library()->GetFunctionLibraryDefinition();
      const FunctionDef* fdef = flib.Find(flib_key);
      OP_REQUIRES(
          ctx, fdef != nullptr,
          errors::Internal("Did not find graphdef for encapsulate ", flib_key,
                           " in NGraphClusterManager or function library"));
      // TODO: how to convert from functiondef to graphdef. Anything easier?
      FunctionBody* fnbody;
      const auto get_func_sig = [&flib](const string& op, const OpDef** sig) {
        return flib.LookUpOpDef(op, sig);
      };
      FunctionDefToBodyHelper(*fdef, {}, &flib, get_func_sig, &fnbody);
      CopyGraph(*fnbody->graph, &m_graph);
    } else {
      GraphConstructorOptions opts;
      opts.allow_internal_ops = true;
      OP_REQUIRES_OK(ctx, ConvertGraphDefToGraph(opts, *graph_def, &m_graph));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ngraph_graph_id", &m_graph_id));

    // Set the backend type for the op
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr<string>("_ngraph_backend", &m_op_backend_name));
    NGRAPH_VLOG(4) << "NGraphDynamicEncapsulateOp::Create backend "
                   << def().name();
    BackendManager::CreateBackend(m_op_backend_name);
    event.Stop();
    ngraph::Event::write_trace(event);
  }

  //---------------------------------------------------------------------------
  //  ~NGraphDynamicEncapsulateOp()
  //---------------------------------------------------------------------------
  ~NGraphDynamicEncapsulateOp() override {
    std::ostringstream oss;
    oss << "Destroy Encapsulate_" << my_instance_id << ": " << name();
    ngraph::Event event(oss.str(), name(), "");
    NGRAPH_VLOG(2) << "~NGraphEncapsulateOp::" << name();

    // If the kernel goes away, we must de-register the function
    // from the freshness tracker.
    if (m_freshness_tracker != nullptr) {
      m_freshness_tracker->RemoveUser(m_ng_exec);

      // TODO(amprocte): We should be able to unref the tracker here, but it
      // seems to screw things up in the C++ unit tests.
      // m_freshness_tracker->Unref();
    }

    // Release the backend
    NGRAPH_VLOG(2) << "~NGraphEncapsulateOp():: ReleaseBackend";
    BackendManager::ReleaseBackend(m_op_backend_name);
    event.Stop();
    ngraph::Event::write_trace(event);
  }

  template <typename T>
  static void TensorDataToStream(std::ostream& ostream, int64 n_elements,
                                 const char* data) {
    const T* data_T = reinterpret_cast<const T*>(data);
    for (size_t i = 0; i < n_elements; i++) {
      ostream << data_T[i] << ",";
    }
  }

  //---------------------------------------------------------------------------
  //  TensorToStream
  //---------------------------------------------------------------------------
  static Status TensorToStream(std::ostream& ostream, const Tensor& tensor) {
    const char* data = tensor.tensor_data().data();
    int64 n_elements = tensor.NumElements();
    switch (tensor.dtype()) {
      case DT_HALF:
        TensorDataToStream<Eigen::half>(ostream, n_elements, data);
        break;
      case DT_FLOAT:
        TensorDataToStream<float>(ostream, n_elements, data);
        break;
      case DT_DOUBLE:
        TensorDataToStream<double>(ostream, n_elements, data);
        break;
      case DT_UINT32:
        TensorDataToStream<uint32>(ostream, n_elements, data);
        break;
      case DT_INT32:
        TensorDataToStream<int32>(ostream, n_elements, data);
        break;
      case DT_UINT8:
      case DT_QUINT8:
        TensorDataToStream<uint8>(ostream, n_elements, data);
        break;
      case DT_UINT16:
      case DT_QUINT16:
        TensorDataToStream<uint16>(ostream, n_elements, data);
        break;
      case DT_INT8:
      case DT_QINT8:
        TensorDataToStream<int8>(ostream, n_elements, data);
        break;
      case DT_INT16:
      case DT_QINT16:
        TensorDataToStream<int16>(ostream, n_elements, data);
        break;
      case DT_UINT64:
        TensorDataToStream<uint64>(ostream, n_elements, data);
        break;
      case DT_INT64:
        TensorDataToStream<int64>(ostream, n_elements, data);
        break;
      case DT_BOOL:
        TensorDataToStream<bool>(ostream, n_elements, data);
        break;
      default:
        return errors::Internal("TensorToStream got unsupported data type ",
                                DataType_Name(tensor.dtype()));
        break;
    }
    return Status::OK();
  }

  Status GetNgExec(OpKernelContext* ctx,
                   std::shared_ptr<ngraph::runtime::Executable>& ng_exec) {
    // Translate the TensorFlow graph to nGraph.
    if (m_ng_exec == nullptr) {
      std::shared_ptr<ngraph::Function> ng_function;
      NGRAPH_VLOG(1) << "Translating: " << ctx->op_kernel().name();
      // Pass empty for input shapes and static input map.
      TF_RETURN_IF_ERROR(
          Builder::TranslateGraph({}, {}, &m_graph, ng_function));
      NGRAPH_VLOG(1) << "Done translating: " << ctx->op_kernel().name();
      ng_function->set_friendly_name(name());

      // Serialize to nGraph if needed
      if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
        std::string file_name =
            "tf_function_" + ctx->op_kernel().name() + ".json";
        NgraphSerialize("tf_function_" + ctx->op_kernel().name() + ".json",
                        ng_function);
#if defined NGRAPH_DISTRIBUTED
        ngraph::Distributed dist;
        int Rank_ID;
        Rank_ID = dist.get_rank();
        NgraphSerialize("tf_function_" + ctx->op_kernel().name() + "_" +
                            to_string(Rank_ID) + ".json",
                        ng_function);
#endif
      }

      BackendManager::LockBackend(m_op_backend_name);

      ngraph::Event event_compile("Compile nGraph", name(), "");
      try {
        NGRAPH_VLOG(1) << "Compiling: " << ctx->op_kernel().name();
        ng_exec =
            BackendManager::GetBackend(m_op_backend_name)->compile(ng_function);
        NGRAPH_VLOG(1) << "Done compiling: " << ctx->op_kernel().name();
      } catch (const std::exception& exp) {
        BackendManager::UnlockBackend(m_op_backend_name);
        NgraphSerialize(
            "tf_function_error_" + ctx->op_kernel().name() + ".json",
            ng_function);
        return errors::Internal("Caught exception while compiling op_backend: ",
                                exp.what(), "\n");
      } catch (...) {
        BackendManager::UnlockBackend(m_op_backend_name);
        NgraphSerialize(
            "tf_function_error_" + ctx->op_kernel().name() + ".json",
            ng_function);
        return errors::Internal("Error in compiling op_backend\n");
      }
      BackendManager::UnlockBackend(m_op_backend_name);
      event_compile.Stop();
      ngraph::Event::write_trace(event_compile);

      m_ng_exec = ng_exec;
      // caching ng_function to serialize to ngraph if needed
      m_ng_function = ng_function;
    } else {
      ng_exec = m_ng_exec;
    }
    return Status::OK();
  }

  // NOTE: This also writes data to the input tensors, if device is not CPU.
  Status AllocateTensorInput(
      OpKernelContext* ctx,
      std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      const std::vector<TensorShape>& input_shapes,
      vector<shared_ptr<ng::runtime::Tensor>>& ng_inputs) {
    NGRAPH_VLOG(5) << "AllocateTensorInput begin";
    // Allocate tensors for input arguments.
    ngraph::Event event_alloc_input("Input: maybe create", name(), "");
    std::vector<std::unique_ptr<ngraph::Event>> input_copy_events;

    std::vector<std::pair<void*, std::shared_ptr<ng::runtime::Tensor>>>&
        input_caches = m_ng_exec_input_cache;
    input_caches.resize(input_shapes.size());

    for (int i = 0; i < input_shapes.size(); i++) {
      NGRAPH_VLOG(5) << "AllocateTensorInput begin: " << i;
      ng::Shape ng_shape(input_shapes[i].dims());
      for (int j = 0; j < input_shapes[i].dims(); ++j) {
        ng_shape[j] = input_shapes[i].dim_size(j);
      }
      NGRAPH_VLOG(5) << "Made shape: " << i;
      ng::element::Type ng_element_type;
      TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(ctx->input(i).dtype(),
                                                       &ng_element_type));
      NGRAPH_VLOG(5) << "Got ET: " << i;

      // At the first call of the ng_exec, both last_src_ptr and
      // last_ng_tensor shall point to null. Otherwise, they are retrived
      // from cache.
      void* last_src_ptr = input_caches[i].first;
      NGRAPH_VLOG(5) << "Got last src ptr: " << i << ": " << last_src_ptr;
      std::shared_ptr<ng::runtime::Tensor> last_ng_tensor =
          input_caches[i].second;
      NGRAPH_VLOG(5) << "Got last ng tensor: " << i << ": " << last_ng_tensor;

      void* current_src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      NGRAPH_VLOG(5) << "Got current src ptr: " << i << ": " << current_src_ptr;
      std::shared_ptr<ng::runtime::Tensor> current_ng_tensor =
          GetCurrentNgTensor(current_src_ptr, last_src_ptr, last_ng_tensor,
                             false, ng_exec, ng_element_type, ng_shape);
      NGRAPH_VLOG(5) << "Got current nG tensor: " << i << ": "
                     << current_ng_tensor << ", "
                     << current_ng_tensor->get_shape();

      if (current_ng_tensor->get_stale()) {
        NGRAPH_VLOG(5) << "Tensor is stale: " << i;
        try {
          size_t copy_size =
              current_ng_tensor->get_element_count() * ng_element_type.size();
          string event_name =
              "Input_" + to_string(i) + "_" + to_string(copy_size);
          std::unique_ptr<ngraph::Event> event_copy_input_next(
              new ngraph::Event(event_name, name(), ""));
          current_ng_tensor->write(
              current_src_ptr, 0,
              current_ng_tensor->get_element_count() * ng_element_type.size());
          NGRAPH_VLOG(5) << "Wrote tensor data: " << i;

          event_copy_input_next->Stop();
          input_copy_events.push_back(std::move(event_copy_input_next));

        } catch (const std::exception& exp) {
          errors::Internal(
              "Caught exception while transferring tensor data to nGraph\n");
        } catch (...) {
          errors::Internal("Error in transferring tensor data to nGraph\n");
        }
      }
      input_caches[i] = std::make_pair(current_src_ptr, current_ng_tensor);
      ng_inputs.push_back(current_ng_tensor);
    }
    // Now write the events back
    for (auto& next : input_copy_events) {
      ngraph::Event::write_trace(*next.get());
    }
    NGRAPH_VLOG(5) << "Wrote events back";

    return Status::OK();
  }

  Status AllocateNGTensorOutput(
      OpKernelContext* ctx,
      std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      std::vector<TensorShape>& input_shapes) {
    m_ng_output_tensors.resize(ng_exec->get_results().size());
    // ngraph executable returns get_results, using that to get the tensor shape
    // and element type.
    for (auto i = 0; i < ng_exec->get_results().size(); i++) {
      if (m_ng_output_tensors[i] == nullptr) {
        m_ng_output_tensors[i] =
            BackendManager::GetBackend(m_op_backend_name)
                ->create_dynamic_tensor(
                    ng_exec->get_results()[i]->get_element_type(),
                    ng_exec->get_results()[i]->get_output_partial_shape(0));
      }
    }
    return Status::OK();
  }

  Status AllocateTFTensorOutput(OpKernelContext* ctx,
                                std::vector<Tensor*>& allocated_tensors) {
    allocated_tensors.clear();
    for (auto i = 0; i < m_ng_output_tensors.size(); i++) {
      if (m_ng_output_tensors[i]->get_partial_shape().is_dynamic() ||
          m_ng_output_tensors[i]->get_element_type().is_dynamic()) {
        errors::Internal(
            "Result shape or element type of nGraph output tensor is dynamic");
      }
      auto ng_shape = m_ng_output_tensors[i]->get_shape();
      auto ng_element_type = m_ng_output_tensors[i]->get_element_type();

      // Create the TF output tensor
      vector<int64> dims;
      for (auto dim : ng_shape) {
        dims.push_back(dim);
      }
      TensorShape tf_shape(dims);
      Tensor* output_tensor = nullptr;
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, tf_shape, &output_tensor));
      allocated_tensors.push_back(output_tensor);

      // Make sure the nGraph-inferred element type agrees with what TensorFlow
      // expected.
      ng::element::Type expected_elem_type;
      TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(
          ctx->expected_output_dtype(i), &expected_elem_type));
      if (ng_element_type != expected_elem_type) {
        errors::Internal(
            "Element type inferred by nGraph does not match "
            "the element type expected by TensorFlow");
      }
    }

    return Status::OK();
  }

  //---------------------------------------------------------------------------
  // OpKernel::Compute
  //---------------------------------------------------------------------------
  void Compute(OpKernelContext* ctx) override {
    std::ostringstream oss;
    oss << "Execute: Encapsulate_" << my_instance_id << ": " << name();
    ngraph::Event event(oss.str(), name(), "");

    Timer compute_time;
    std::lock_guard<std::mutex> lock(m_compute_lock);
    NGRAPH_VLOG(4)
        << "NGraphDynamicEncapsulateOp::Compute starting for cluster "
        << m_ngraph_cluster;

    ngraph::Event event_func_maybe_create("FunctionMaybeCreate", name(), "");
    Timer function_lookup_or_create;

    std::vector<TensorShape> input_shapes;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      input_shapes.push_back(ctx->input(i).shape());
    }

    std::shared_ptr<ngraph::runtime::Executable> ng_exec;

    // Get ngraph executable and inputs information
    OP_REQUIRES_OK(ctx, GetNgExec(ctx, ng_exec));

    NGRAPH_VLOG(4) << "NGraphDynamicEncapsulateOp::Compute got ngraph "
                      "executable for cluster "
                   << m_ngraph_cluster;

    int time_func_create_or_lookup = function_lookup_or_create.ElapsedInMS();
    event_func_maybe_create.Stop();

    NGRAPH_VLOG(4)
        << "NGraphDynamicEncapsulateOp::Compute got graph for cluster "
        << m_ngraph_cluster;

    Timer create_or_lookup_tensors;

    if (m_freshness_tracker == nullptr) {
      auto creator = [](NGraphFreshnessTracker** tracker) {
        *tracker = new NGraphFreshnessTracker();
        return Status::OK();
      };
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->LookupOrCreate<NGraphFreshnessTracker>(
                   ctx->resource_manager()->default_container(),
                   "ngraph_freshness_tracker", &m_freshness_tracker, creator));
    }

    NGRAPH_VLOG(4) << "NGraphDynamicEncapsulateOp::Compute got freshness "
                      "tracker for cluster "
                   << m_ngraph_cluster;

    // Allocate tensors for input arguments.
    ngraph::Event event_alloc_input("Input: maybe create", name(), "");

    vector<shared_ptr<ng::runtime::Tensor>> ng_inputs;

    OP_REQUIRES_OK(ctx,
                   AllocateTensorInput(ctx, ng_exec, input_shapes, ng_inputs));

    NGRAPH_VLOG(4)
        << "NGraphDynamicEncapsulateOp::Compute allocated argument tensors "
           "for cluster "
        << m_ngraph_cluster;

    event_alloc_input.Stop();

    // Allocate nG tensors for the output results.
    ngraph::Event event_ng_alloc_output("Output: maybe create nG", name(), "");

    OP_REQUIRES_OK(ctx, AllocateNGTensorOutput(ctx, ng_exec, input_shapes));

    event_ng_alloc_output.Stop();

    // Execute the nGraph function.
    ngraph::Event event_execute_function("Execute nGraph", name(), "");
    Timer execute_function;
    {
      BackendManager::LockBackend(m_op_backend_name);
      NGRAPH_VLOG(4)
          << "NGraphDynamicEncapsulateOp::Compute call starting for cluster "
          << m_ngraph_cluster;
      try {
        ng_exec->call(m_ng_output_tensors, ng_inputs);
      } catch (const std::exception& exp) {
        auto ng_function = m_ng_function;
        BackendManager::UnlockBackend(m_op_backend_name);
        NgraphSerialize(
            "tf_function_error_" + ctx->op_kernel().name() + ".json",
            ng_function);
        OP_REQUIRES(ctx, false,
                    errors::Internal(
                        "Caught exception while executing nGraph computation: ",
                        exp.what(), "\n"));
      } catch (...) {
        auto ng_function = m_ng_function;
        BackendManager::UnlockBackend(m_op_backend_name);
        NgraphSerialize(
            "tf_function_error_" + ctx->op_kernel().name() + ".json",
            ng_function);
        OP_REQUIRES(
            ctx, false,
            errors::Internal("Error in executing the nGraph computation\n"));
      }
      BackendManager::UnlockBackend(m_op_backend_name);
    }
    int time_execute_function = execute_function.ElapsedInMS();
    event_execute_function.Stop();

    NGRAPH_VLOG(4)
        << "NGraphDynamicEncapsulateOp::Compute call done for cluster "
        << m_ngraph_cluster;

    // MOVED from pre-execution:
    // Allocate TF tensors for the output results.
    ngraph::Event event_tf_alloc_output("Output: maybe create TF", name(), "");
    std::vector<Tensor*> allocated_output_tensors;
    OP_REQUIRES_OK(ctx, AllocateTFTensorOutput(ctx, allocated_output_tensors));

    event_tf_alloc_output.Stop();
    NGRAPH_VLOG(4) << "NGraphDynamicEncapsulateOp::Compute allocated result "
                      "tensors for cluster "
                   << m_ngraph_cluster;

    int time_create_or_lookup_tensors = create_or_lookup_tensors.ElapsedInMS();

    // Copy value to host
    ngraph::Event event_copy_output("Output - copy back", name(), "");
    Timer copy_output_tensors_to_host;

    try {
      size_t output_tensor_count = m_ng_output_tensors.size();
      std::vector<std::unique_ptr<ngraph::Event>> output_copy_events;
      for (size_t i = 0; i < output_tensor_count; ++i) {
        NGRAPH_VLOG(5) << "Working on output tensor: " << i;
        void* dst_ptr;
        dst_ptr = (void*)DMAHelper::base(allocated_output_tensors[i]);
        NGRAPH_VLOG(5) << "Got dst ptr: " << i << ": " << dst_ptr;
        std::shared_ptr<ng::runtime::Tensor> dst_ng_tensor;
        dst_ng_tensor = m_ng_output_tensors[i];
        if (dst_ng_tensor->get_partial_shape().is_static()) {
          NGRAPH_VLOG(5) << "Got dst_ng_tensor: " << i << ": " << dst_ng_tensor
                         << ", " << dst_ng_tensor->get_shape();
        } else {
          NGRAPH_VLOG(5) << "Got dst_ng_tensor: " << i << ": " << dst_ng_tensor
                         << ", " << dst_ng_tensor->get_shape();
        }
        auto ng_element_type = dst_ng_tensor->get_element_type();
        NGRAPH_VLOG(5) << "Got ET: " << i << ": " << ng_element_type;
        std::unique_ptr<ngraph::Event> event_copy_output_next(new ngraph::Event(
            ("Output_" + std::to_string(i) + "_" +
             std::to_string(dst_ng_tensor->get_element_count() *
                            ng_element_type.size())),
            name(), ""));
        NGRAPH_VLOG(5) << "Started copy event";
        NGRAPH_VLOG(5) << "About to read: " << i << ": " << dst_ng_tensor
                       << ", " << dst_ng_tensor->get_shape();
        dst_ng_tensor->read(dst_ptr, 0, dst_ng_tensor->get_element_count() *
                                            ng_element_type.size());
        dst_ng_tensor->set_stale(true);
        NGRAPH_VLOG(5) << "Done reading: " << i << ": " << dst_ng_tensor << ", "
                       << dst_ng_tensor->get_shape();
        event_copy_output_next->Stop();
        output_copy_events.push_back(std::move(event_copy_output_next));
      }
      // Now write the events back
      for (auto& next : output_copy_events) {
        ngraph::Event::write_trace(*next.get());
      }
    } catch (const std::exception& exp) {
      OP_REQUIRES(
          ctx, false,
          errors::Internal(
              "Caught exception while transferring tensor data to host: ",
              exp.what(), "\n"));
    } catch (...) {
      OP_REQUIRES(
          ctx, false,
          errors::Internal("Error in transferring tensor data to host\n"));
    }
    event_copy_output.Stop();

    // Mark input tensors as fresh for the next time around.
    // Note: these ng_tensors are being marked fresh so that in the next
    // iteration if this encapsulate finds the tensor fresh, then it will use it
    for (int i = 0; i < input_shapes.size(); i++) {
      NGRAPH_VLOG(5) << "Marking fresh: " << i;
      void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      m_freshness_tracker->MarkFresh(src_ptr, ng_exec);
    }
    int time_copy_output_tensors_to_host =
        copy_output_tensors_to_host.ElapsedInMS();

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
        << m_ngraph_cluster;
    NGRAPH_VLOG(1) << "NGRAPH_TF_TIMING_PROFILE: OP_ID: " << my_instance_id
                   << " Step_ID: " << ctx->step_id()
                   << " Cluster: " << ctx->op_kernel().name()
                   << " Time-Compute: " << compute_time.ElapsedInMS()
                   << " Function-Create-or-Lookup: "
                   << time_func_create_or_lookup << " Create-and-copy-tensors: "
                   << time_create_or_lookup_tensors
                   << " Execute: " << time_execute_function
                   << " Copy-outputs-to-host: "
                   << time_copy_output_tensors_to_host;
    event.Stop();
    ngraph::Event::write_trace(event_func_maybe_create);
    ngraph::Event::write_trace(event_ng_alloc_output);
    ngraph::Event::write_trace(event_tf_alloc_output);
    ngraph::Event::write_trace(event_alloc_input);
    ngraph::Event::write_trace(event_execute_function);
    ngraph::Event::write_trace(event_copy_output);
    ngraph::Event::write_trace(event);
  }  // end compute

 private:
  // TF Graph for the cluster
  Graph m_graph;
  std::shared_ptr<ngraph::runtime::Executable> m_ng_exec;
  std::shared_ptr<ngraph::Function> m_ng_function;
  NgFunctionIOCache m_ng_exec_input_cache;
  std::vector<std::shared_ptr<ng::runtime::Tensor>> m_ng_output_tensors;
  // Freshness tracker maintains a set of ng::functions using a particular base
  // pointer(for Tensor)
  // A single instance of freshness_tracker is used across all
  // nGraphEncapsulateOp and nGraphVariable op
  NGraphFreshnessTracker* m_freshness_tracker;
  int m_ngraph_cluster{-1};
  int m_graph_id{-1};
  std::mutex m_compute_lock;
  string m_op_backend_name;
  std::shared_ptr<ng::runtime::Tensor> GetCurrentNgTensor(
      void* current_tf_ptr, void* last_tf_ptr,
      const std::shared_ptr<ng::runtime::Tensor>& last_ng_tensor,
      const bool& output_tensor,
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      const ng::element::Type& ng_element_type, const ng::Shape& ng_shape) {
    // NOTE: we assume that TF's pointers WILL change if it actually changes
    // values. ie, it will not reuse the same space if its rewritten it
    bool tf_tensor_has_changed = current_tf_ptr != last_tf_ptr;
    bool no_ng_tensor_found = last_ng_tensor == nullptr;

    // We need to check last_ng_tensor != nullptr, since there are cases where
    // at the first call to the ng_exec, both current_dst_ptr (when the
    // output is a 0-sized tensor) and last_dst_ptr (uninitialized at the
    // first call) are nullptr
    // A new tensor needs to be created for sure if no_ng_tensor_found
    bool need_new_tensor_creation;
    need_new_tensor_creation =
        no_ng_tensor_found || last_ng_tensor->get_shape() != ng_shape;

    // It is stale if a new tensor was created OR the tf tensor has changed OR
    // (tf tensor has not changed, but freshness tracker says its stale)
    bool is_stale;
    if (output_tensor) {
      is_stale = true;  // For output tensors, it is always set stale to true
    } else {
      is_stale = need_new_tensor_creation || tf_tensor_has_changed ||
                 (!tf_tensor_has_changed &&
                  !m_freshness_tracker->IsFresh(current_tf_ptr, ng_exec));
    }

    // create a new ng tensor or use the last one
    std::shared_ptr<ng::runtime::Tensor> current_ng_tensor;
    if (need_new_tensor_creation) {
      current_ng_tensor = BackendManager::GetBackend(m_op_backend_name)
                              ->create_tensor(ng_element_type, ng_shape);
    } else {
      current_ng_tensor = last_ng_tensor;
    }
    current_ng_tensor->set_stale(is_stale);
    return current_ng_tensor;
  }
  static int s_instance_count;
  int my_instance_id{0};
};

int NGraphDynamicEncapsulateOp::s_instance_count = 0;

}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphDynamicEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphDynamicEncapsulateOp);

}  // namespace tensorflow
