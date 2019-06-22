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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <thread>

#include "ngraph/event_tracing.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_timer.h"
#include "version.h"

using namespace std;
using tensorflow::Session;
using tensorflow::Status;
using tensorflow::SessionOptions;
using tensorflow::RewriterConfig;
using tensorflow::OptimizerOptions_Level_L0;

namespace tf = tensorflow;

extern tf::Status LoadGraph(const string& graph_file_name,
                            std::unique_ptr<tf::Session>* session,
                            const tf::SessionOptions& options);

extern tf::Status ReadTensorFromImageFile(const string& file_name,
                                          const int input_height,
                                          const int input_width,
                                          const float input_mean,
                                          const float input_std, bool use_NCHW,
                                          std::vector<tf::Tensor>* out_tensors);

extern tf::Status PrintTopLabels(const std::vector<tf::Tensor>& outputs,
                                 const string& labels_file_name);

// Prints the available backends
void PrintAvailableBackends() {
  // Get the list of backends
  auto supported_backends =
      tf::ngraph_bridge::BackendManager::GetSupportedBackendNames();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  cout << "Available backends: " << endl;
  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }
}

// Sets the specified backend. This backend must be set BEFORE running
// the computation
tf::Status SetNGraphBackend(const string& backend_name) {
  // Select a backend
  tf::Status status =
      tf::ngraph_bridge::BackendManager::SetBackendName(backend_name);
  return status;
}

void PrintVersion() {
  // nGraph Bridge version info
  std::cout << "Bridge version: " << tf::ngraph_bridge::ngraph_tf_version()
            << std::endl;
  std::cout << "nGraph version: " << tf::ngraph_bridge::ngraph_lib_version()
            << std::endl;
  std::cout << "CXX11_ABI Used: "
            << tf::ngraph_bridge::ngraph_tf_cxx11_abi_flag() << std::endl;
  std::cout << "Grappler Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;
  std::cout << "Variables Enabled? "
            << (tf::ngraph_bridge::ngraph_tf_are_variables_enabled()
                    ? std::string("Yes")
                    : std::string("No"))
            << std::endl;

  PrintAvailableBackends();
}

Status CreateSession(const string& graph_filename,
                     unique_ptr<Session>& session) {
  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);

  // The following is related to Grappler - which we are turning off
  // Until we get a library fully running
  if (tf::ngraph_bridge::ngraph_tf_is_grappler_enabled()) {
    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->add_custom_optimizers()
        ->set_name("ngraph-optimizer");

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_min_graph_nodes(-1);

    options.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_meta_optimizer_iterations(RewriterConfig::ONE);
  }

  // Load the network
  Status load_graph_status = LoadGraph(graph_filename, &session, options);
  return load_graph_status;
}

int main(int argc, char** argv) {
  string image = "grace_hopper.jpg";
  string graph = "inception_v3_2016_08_28_frozen.pb";
  string labels = "";
  int input_width = 299;
  int input_height = 299;
  float input_mean = 0.0;
  float input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool use_NCHW = false;

  std::vector<tf::Flag> flag_list = {
      tf::Flag("image", &image, "image to be processed"),
      tf::Flag("graph", &graph, "graph to be executed"),
      tf::Flag("labels", &labels, "name of file containing labels"),
      tf::Flag("input_width", &input_width,
               "resize image to this width in pixels"),
      tf::Flag("input_height", &input_height,
               "resize image to this height in pixels"),
      tf::Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      tf::Flag("input_std", &input_std,
               "scale pixel values to this std deviation"),
      tf::Flag("input_layer", &input_layer, "name of input layer"),
      tf::Flag("output_layer", &output_layer, "name of output layer"),
      tf::Flag("use_NCHW", &use_NCHW, "Input data in NCHW format"),
  };

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cout << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    std::cout << "Error: Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  const char* backend = "CPU";
  int num_images_for_each_thread = 10;

  if (SetNGraphBackend(backend) != tf::Status::OK()) {
    std::cout << "Error: Cannot set the backend: " << backend << std::endl;
    return -1;
  }

  std::cout << "Component versions\n";
  PrintVersion();

  std::cout << "\nCreating session\n";
  ngraph::Event session_create_event("Session Create", "", "");

  // Run the MatMul example
  unique_ptr<Session> session;
  TF_CHECK_OK(CreateSession(graph, session));
  session_create_event.Stop();
  ngraph::Event::write_trace(session_create_event);

  // Create threads and fire up the images
  const int NUM_THREADS = 1;
  std::thread threads[NUM_THREADS];

  std::cout << "Running inferences\n";

  std::vector<tf::Tensor> outputs;

  ngraph::Event read_event("Read", "Image reading", "");

  // Read image
  std::vector<tf::Tensor> resized_tensors;
  tf::Status read_tensor_status =
      ReadTensorFromImageFile(image, input_height, input_width, input_mean,
                              input_std, use_NCHW, &resized_tensors);

  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  read_event.Stop();
  ngraph::Event::write_trace(read_event);

  const tf::Tensor& input_image = resized_tensors[0];
  for (int i = 0; i < NUM_THREADS; i++) {
    threads[i] = std::thread([=, &session, &input_image, &outputs] {
      cout << "Warming up...";
      // Warm up - which also includes initial compilation time
      // that could be big for some devices such as GPU
      for (int iter_count = 0; iter_count < 4; iter_count++) {
        tf::Status run_status = session->Run({{input_layer, input_image}},
                                             {output_layer}, {}, &outputs);
        if (!run_status.ok()) {
          LOG(ERROR) << "Running model failed: " << run_status;
        }
      }
      cout << "done\n";

      tf::ngraph_bridge::Timer infer_timer;
      for (int iter_count = 0; iter_count < num_images_for_each_thread;
           iter_count++) {
        // Run inference
        ostringstream oss;
        oss.clear();
        oss.seekp(0);
        oss << "Infer(" << i << ") [" << iter_count << "]";
        ngraph::Event infer_event(oss.str(), "Inference", "");

        tf::Status run_status = session->Run({{input_layer, input_image}},
                                             {output_layer}, {}, &outputs);
        if (!run_status.ok()) {
          LOG(ERROR) << "Running model failed: " << run_status;
        }
        infer_event.Stop();

        // Write the events
        ngraph::Event::write_trace(infer_event);
      }
      infer_timer.Stop();
      auto elapsed = infer_timer.ElapsedInMS();
      std::cout << "Thread[" << i << "] Time total: "
                << (double)elapsed / num_images_for_each_thread
                << " ms per image ("
                << (double)num_images_for_each_thread / elapsed * 1000.0
                << " img/sec) \n";
    });
  }

  // Wait until everyone is done
  for (auto& next_thread : threads) {
    next_thread.join();
  }

  // Now
  std::cout << "Done\n";
  return 0;
}
