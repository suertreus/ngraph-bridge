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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tf_graph_writer.h"

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

//namespace grappler {

//class MarkForClustering : public ::testing::Test {};

TEST(MarkForClustering, ShapeTest) {
  //using test::function::NDef;
  //using test::function::GDef;
  MetaGraphDef meta_graph;


// TODO use graphdefbuilder instead of graph->graphdef
 Graph g(OpRegistry::Global());
 Tensor t_input_0(DT_FLOAT, TensorShape{2, 3});
  Tensor t_input_1(DT_FLOAT, TensorShape{2, 3});

  Node* node1;
  ASSERT_OK(NodeBuilder("node1", "Const")
                .Attr("dtype", DT_FLOAT)
                .Attr("value", t_input_0)
                .Finalize(&g, &node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2", "Abs")
                .Input(node1, 0)
                .Attr("T", DT_FLOAT)
                .Finalize(&g, &node2));

  // Add edges from SRC to node1 and node2
  // Add edge from node3 to SINK
  // The graph is disconnected without these edges
  Node* source = g.source_node();
  Node* sink = g.sink_node();
  g.AddEdge(source, Graph::kControlSlot, node1, Graph::kControlSlot);
  g.AddEdge(node2, Graph::kControlSlot, sink, Graph::kControlSlot);
  GraphDef gdef;
  g.ToGraphDef(&gdef);

  *meta_graph.mutable_graph_def() = gdef;
  CollectionDef collection;
  collection.mutable_node_list()->add_value("node2");
  (*meta_graph.mutable_collection_def())["train_op"] = collection;

  grappler::ItemConfig cfg;
  std::unique_ptr<grappler::GrapplerItem> item =
      GrapplerItemFromMetaGraphDef("0", meta_graph, cfg);

  //grappler::GrapplerItem item;
  grappler::GraphProperties properties(*item);
  Status s = properties.InferStatically(true);

  for (const auto& node_name : vector<string>{"node1", "node2"}) {
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
//}
}
}
}
}