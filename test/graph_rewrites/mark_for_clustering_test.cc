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

#include "gtest/gtest.h"

#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tf_graph_writer.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"


using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

TEST(MarkForClustering, SimpleTest) {
  Graph g(OpRegistry::Global());

  Tensor t_input_0(DT_FLOAT, TensorShape{2, 3});
  Tensor t_input_1(DT_FLOAT, TensorShape{2, 3});

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_0)
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_1)
                .Finalize(&g, &node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3", "Add")
                .Input(node1, 0)
                .Input(node2, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node3));

  // Add edges from SRC to node1 and node2
  // Add edge from node3 to SINK
  // The graph is disconnected without these edges
  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(source, Graph::kControlSlot, node2, Graph::kControlSlot);
  g.AddEdge(node3, Graph::kControlSlot, sink, Graph::kControlSlot);

  ASSERT_OK(MarkForClustering(&g, {}));

  const char* ng_backend_env_value = std::getenv("NGRAPH_TF_BACKEND");
  string expected_backend{"CPU"};
  if (ng_backend_env_value != nullptr) {
    expected_backend = std::string(ng_backend_env_value);
  }

  string backend;
  for (auto node : g.op_nodes()) {
    ASSERT_OK(GetNodeBackend(node, &backend));
    ASSERT_EQ(backend, expected_backend);
  }
}


// TODO move test to different file
TEST(MarkForClustering, ShapeHints) {
  // Create Graph
  Scope root = Scope::NewRootScope();
  auto A = ops::Placeholder(root.WithOpName("A"), DataType::DT_FLOAT);
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});
  auto Add = ops::Add(root.WithOpName("Add"), A, B);
  auto C = ops::Const(root.WithOpName("C"), {3.f, 2.f});
  auto Mul = ops::Mul(root.WithOpName("Mul"), Add, C);

  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(&graph));

  // set device specification
  for (auto node : graph.op_nodes()) {
    node->set_requested_device("CPU");
  }

  // Create GraphDef and Grappler
  grappler::GrapplerItem item;
  graph.ToGraphDef(&item.graph);
  ConfigProto config_proto;
  auto shape_hints = AttrValue();
  shape_hints.set_s("node1:{1,2;3,4}.node2:{1;2}");

  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("ngraph-optimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("ngraph-optimizer");
  (*custom_config->mutable_parameter_map())["_shape_hints"] = shape_hints;

  // Run grappler
  tensorflow::grappler::MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  // const Status status = optimizer.Optimize(nullptr, item, &output);
  ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  // GraphDef to Graph
  Graph output_graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  ASSERT_OK(ConvertGraphDefToGraph(opts, output, &output_graph));

  
}
}
}
}