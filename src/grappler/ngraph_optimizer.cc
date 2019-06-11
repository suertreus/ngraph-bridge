/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "ngraph_optimizer.h"
#include "ngraph_cluster_manager.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"

#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"

#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"






#include <iomanip>
#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include <iostream>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

Status NgraphOptimizer::shape_hints_parser(const string& raw_shape_hint) {
  // Format of raw_shape_hint is expected to be:
  // "node1:{1,2;3,4}.node2:{1;2}"
  auto node_info = ng::split(raw_shape_hint, '.');
  string node_name;
  string hint;
  for (auto i : node_info){
    set<vector<int>> shape_hints_set;
    auto name_and_info = ng::split(i, ':');
    if (name_and_info.size() != 2){
      return  errors::Internal("Expected to be of length 2, but got length ", name_and_info.size(), " after splitting ", i);
    }
    node_name = name_and_info[0];
    hint = name_and_info[1];
    if (hint[0] != '{' && hint[hint.size()-1] != '}'){
      // TODO
      return  errors::Internal("TODO");
    }
    auto individual_hints = ng::split(hint.substr(1,hint.size()-2), ';');
    for (auto individual_hint : individual_hints) {
      vector<int> shape_hint;
      vector<string> dims = ng::split(individual_hint, ',');
      std::transform(dims.begin(), dims.end(), std::back_inserter(shape_hint),
               [](const std::string& str) { return std::stoi(str); });
      shape_hints_set.insert(shape_hint);
    }
    shape_hints[node_name] = shape_hints_set;
  }
  // TODO remove this debug print later
  // Parsing looks ok
  for (auto itr : shape_hints){
    cout << itr.first << "\n";
    for (auto itr1 : itr.second){
      for (auto itr2 : itr1){
        cout << itr2 << ",";
      }
      cout << "\n";
    }
    cout << "=====\n";
  }
  return Status::OK();
}

Status NgraphOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (config == nullptr) {
    NGRAPH_VLOG(3) << "NGTF_OPTIMIZER: config is null ";
  } else {
    NGRAPH_VLOG(3) << "NGTF_OPTIMIZER: config is not null ";
    const auto params = config->parameter_map();
    if (params.count("_shape_hints")) {
      TF_RETURN_IF_ERROR(shape_hints_parser(params.at("_shape_hints").s()));
    }
  }
  return Status::OK();
}

Status NgraphOptimizer::Optimize(tensorflow::grappler::Cluster* cluster,
                                 const tensorflow::grappler::GrapplerItem& item,
                                 GraphDef* output) {
  NGRAPH_VLOG(3) << "NGTF_OPTIMIZER: Here at NgraphOptimizer ";
  NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: grappler item id " << item.id;

  tensorflow::grappler::GrapplerItem item_copy(item);
  cout <<  SummarizeGraphDef(item_copy.graph);

  set<string> inp_placeholders;
  vector<int> locations;
  int ctr = 0;
  for (auto n : item_copy.graph.node()){
    if (n.op() == "Placeholder"){
      inp_placeholders.insert(n.name());
      locations.push_back(ctr);
    }
    ctr++;
  }

  /*
  for (auto itr : inp_placeholders) {
    NodeDef* node_def = item_copy.graph.add_node();
    Status status = NodeDefBuilder(itr + "_const", "Const")
                        .Attr("T", DT_FLOAT)
                        .Finalize(node_def);
  }*/
  for (auto x : locations){
    auto ndef = item_copy.graph.mutable_node(x);
    *(ndef->mutable_op()) = "Const";
    AttrValue av;
    TensorProto tensor_proto;
    // TODO: set appropriate dtype
    tensor_proto.set_dtype(DT_FLOAT);
    // TODO: use correct dim
    int dim = 2;
    tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
    for (int i = 0; i < dim; ++i) {
      tensor_proto.add_float_val(1);
    }
    *(av.mutable_tensor()) = tensor_proto;
    (*ndef->mutable_attr())["value"] = av;
  }
  
  for (auto n : item_copy.graph.node()){
    for (int inp_idx = 0; inp_idx < n.input_size(); inp_idx++){
      string inp_name = n.input(inp_idx);
      auto split_control = ng::split(inp_name, '^');
      auto split_out_port = ng::split(split_control[0], ':');
      string op_name = split_out_port[0];
      inp_placeholders.find(op_name);
      inp_placeholders.end();
      if (inp_placeholders.find(op_name) != inp_placeholders.end()) {
        // TODO: DO NOT MODIFY item_copy.graph while looping over it
        // This node has a placeholder as an input
        //NodeDef* node_def = item_copy.graph.add_node();
        // TODO: replace DT_FLOAT with type from placeholder
        // TODO: add device from placeholder
        // TODO check if other stuff needs to be added
        //Status status = NodeDefBuilder(n.name(), "Const")
        //                .Attr("T", DT_FLOAT)
        //                .Finalize(node_def);
        // TODO: handle Status

        //*(n.mutable_input(inp_idx)) = 
        
      }
    }
  }

  grappler::GraphProperties properties(item_copy);
  Status s = properties.InferStatically(true);
  GraphDef output_graph_def;
  cout << "HERE1\n";
  Status s1 = properties.AnnotateOutputShapes(&output_graph_def);
  if (s != Status::OK()){
    cout << "TODO: oops\n";
  }
  cout << "HERE2\n";



  // TODO: this for loop is not entered. node_size==0
  for (int ii = 0; ii < output_graph_def.node_size(); ii++){
    cout << "cvcvvcvcvvcvcvc\n";
    string ss("_output_shapes");
    auto attr_map = output_graph_def.node(ii).attr();
    auto shape_proto = attr_map[ss].shape();
    for (int jj = 0; jj < shape_proto.dim_size(); jj++){
      cout << shape_proto.dim(jj).size() << ",";
    }
    cout << "\n";
    // _output_shapes
  }

  cout <<  SummarizeGraphDef(output_graph_def);

  // TODO: Not producing right results
  for (const auto& node_name : vector<string>{"A", "B", "Add", "C", "Mul"}) {
    cout << node_name << "\n";
    auto inp_props = properties.GetInputProperties(node_name);
    cout << inp_props.size() << "\n";
    auto out_props = properties.GetOutputProperties(node_name);
    cout << out_props.size() << "\n";

    for (int ii = 0; ii < inp_props.size(); ii++){
      const OpInfo::TensorProperties& prop_1 = inp_props[ii];
      auto sh = prop_1.shape();
      for (int i = 0; i < sh.dim_size(); i++){
        cout << sh.dim(i).size() << " -- ";
      }
      cout << "\n";
    }

    for (int ii = 0; ii < out_props.size(); ii++){
      const OpInfo::TensorProperties& prop_1 = out_props[ii];
      auto sh = prop_1.shape();
      for (int i = 0; i < sh.dim_size(); i++){
        cout << sh.dim(i).size() << " -- ";
      }
      cout << "\n";
    }
  }


  // Convert the GraphDef to Graph
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = true;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, item.graph, &graph));

  // For filename generation purposes, grab a fresh index. This is just an
  // arbitrary integer to avoid filename collisions resulting from subsequent
  // runs of this pass.
  int idx = FreshIndex();

  // If requested, dump pre-capture graphs.
  if (DumpPrecaptureGraphs()) {
    DumpGraphs(graph, idx, "precapture", "Pre-Capture Graph");
  }

  // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
  // we will not do anything; all subsequent
  // passes become a no-op.
  bool ngraph_not_enabled =
      (!config::IsEnabled()) || (std::getenv("NGRAPH_TF_DISABLE") != nullptr);
  bool already_processed = IsProcessedByNgraphPass(&graph);
  if (ngraph_not_enabled || already_processed) {
    NGRAPH_VLOG(0) << "Not running through nGraph. nGraph not enabled: "
                   << ngraph_not_enabled
                   << " Already processed: " << already_processed;
    NGraphClusterManager::EvictAllClusters();
    graph.ToGraphDef(output);
    return Status::OK();
  }

  // TODO: Find out a better way to preserve feed nodes, init_ops and
  // keep_ops instead of just skipping those from clustering.
  // Get nodes to be preserved/skipped
  std::set<string> nodes_to_preserve;

  // Feed Nodes
  for (int i = 0; i < item.feed.size(); i++) {
    nodes_to_preserve.insert(item.feed[i].first);
  }

  // Keep Ops
  nodes_to_preserve.insert(item.keep_ops.begin(), item.keep_ops.end());

  // Init Ops
  nodes_to_preserve.insert(item.init_ops.begin(), item.init_ops.end());

  // Find a list of nodes that are of the types that are disabled
  std::set<string> disabled_nodes;
  std::set<string> disabled_ops_set = config::GetDisabledOps();
  for (auto itr : graph.nodes()) {
    if (disabled_ops_set.find(itr->type_string()) != disabled_ops_set.end()) {
      disabled_nodes.insert(itr->name());
    }
  }

  // Fetch Nodes
  std::set<string> fetch_nodes;
  for (const string& f : item.fetch) {
    int pos = f.find(":");
    fetch_nodes.insert(f.substr(0, pos));
  }

  // nodes_to_add_identity_to = fetch_nodes - disabled_nodes
  std::set<string> nodes_to_add_identity_to;
  std::set_difference(fetch_nodes.begin(), fetch_nodes.end(),
                      disabled_nodes.begin(), disabled_nodes.end(),
                      std::inserter(nodes_to_add_identity_to,
                                    nodes_to_add_identity_to.begin()));

  // Rewrite graph to add IdentityN node so the fetch node can be encapsulated
  // as well
  // If the fetch node in question has 0 outputs or any of the outputs
  // has ref type as a data type then don't add IdentityN node, but the fetch
  // node will be skipped from capturing and marking for clustering.
  TF_RETURN_IF_ERROR(AddIdentityN(&graph, nodes_to_add_identity_to));

  nodes_to_preserve.insert(nodes_to_add_identity_to.begin(),
                           nodes_to_add_identity_to.end());
  std::set<string>& skip_these_nodes = nodes_to_preserve;

  //
  // Variable capture: Part that replaces all instances of VariableV2 with the
  // NGraphVariable op. Making this replacement allows us to substitute in a
  // kernel that tracks the freshness of variables (invalidating freshness when
  // the reference is handed off to an "untrusted" op).
  //

  // Do variable capture then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(CaptureVariables(&graph, skip_these_nodes));
  if (DumpCapturedGraphs()) {
    DumpGraphs(graph, idx, "captured", "Graph With Variables Captured");
  }

  //
  // Encapsulation: Part that rewrites the graph for nGraph operation.
  //
  // The part has several phases, each executed in sequence:
  //
  //   1. Marking [ngraph_mark_for_clustering.cc]
  //   2. Cluster Assignment [ngraph_assign_clusters.cc]
  //   3. Cluster Deassignment [ngraph_deassign_clusters.cc]
  //   4. Cluster Encapsulation [ngraph_encapsulate_clusters.cc] - currently
  //      part of the ngraph_rewrite_pass.cc to be executed after POST_REWRITE
  //
  // Between phases, graph dumps (in both .dot and .pbtxt format) may be
  // requested by setting the following environment variables:
  //
  //   NGRAPH_TF_DUMP_UNMARKED_GRAPHS=1      dumps graphs before phase 1
  //   NGRAPH_TF_DUMP_MARKED_GRAPHS=1        dumps graphs after phase 1
  //   NGRAPH_TF_DUMP_CLUSTERED_GRAPHS=1     dumps graphs after phase 2
  //   NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS=1   dumps graphs after phase 3
  //   NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS=1  dumps graphs after phase 4
  //   NGRAPH_TF_DUMP_GRAPHS=1               all of the above
  //

  // If requested, dump unmarked graphs.
  if (DumpUnmarkedGraphs()) {
    DumpGraphs(graph, idx, "unmarked", "Unmarked Graph");
  }

  // 1. Mark for clustering then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(MarkForClustering(&graph, skip_these_nodes));
  if (DumpMarkedGraphs()) {
    DumpGraphs(graph, idx, "marked", "Graph Marked for Clustering");
  }

  // 2. Assign clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(AssignClusters(&graph));
  if (DumpClusteredGraphs()) {
    DumpGraphs(graph, idx, "clustered", "Graph with Clusters Assigned");
  }

  // 3. Deassign trivial clusters then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(DeassignClusters(&graph));
  if (DumpDeclusteredGraphs()) {
    DumpGraphs(graph, idx, "declustered",
               "Graph with Trivial Clusters De-Assigned");
  }

  // 4. Encapsulate clusters then, if requested, dump the graphs.
  FunctionDefLibrary* fdeflib_new = new FunctionDefLibrary();
  TF_RETURN_IF_ERROR(EncapsulateClusters(&graph, idx, fdeflib_new));
  if (DumpEncapsulatedGraphs()) {
    DumpGraphs(graph, idx, "encapsulated", "Graph with Clusters Encapsulated");
  }

  // Rewrite for tracking then, if requested, dump the graphs.
  TF_RETURN_IF_ERROR(RewriteForTracking(&graph, idx));
  if (DumpTrackedGraphs()) {
    DumpGraphs(graph, idx, "tracked",
               "Graph with Variables Rewritten for Tracking");
  }

  // Convert the graph back to Graphdef
  graph.ToGraphDef(output);
  // According to the doc, the message takes ownership of the allocated object
  // https://developers.google.com/protocol-buffers/docs/reference/cpp-generated#proto3_string
  // Hence no need to free fdeflib_new
  output->set_allocated_library(fdeflib_new);
  return Status::OK();
}

void NgraphOptimizer::Feedback(tensorflow::grappler::Cluster* cluster,
                               const tensorflow::grappler::GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

void NgraphOptimizer::DumpGraphs(Graph& graph, int idx,
                                 std::string filename_prefix,
                                 std::string title) {
  // If we have a "main" graph, dump that.
  auto dot_filename = DotFilename(filename_prefix, idx);
  auto pbtxt_filename = PbtxtFilename(filename_prefix, idx);
  NGRAPH_VLOG(0) << "NGTF_OPTIMIZER: Dumping main graph to " << dot_filename;
  NGRAPH_VLOG(0) << "NGTF_OPTIMIZER: Dumping main graph to " << pbtxt_filename;

  GraphToDotFile(&graph, dot_filename, title);
  GraphToPbTextFile(&graph, pbtxt_filename);
}

int NgraphOptimizer::FreshIndex() {
  mutex_lock l(s_serial_counter_mutex);
  return s_serial_counter++;
}

REGISTER_GRAPH_OPTIMIZER_AS(NgraphOptimizer, "ngraph-optimizer");

}  // end namespace ngraph_bridge

}  // end namespace tensorflow
