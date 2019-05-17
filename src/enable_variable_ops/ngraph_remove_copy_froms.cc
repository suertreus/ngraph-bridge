/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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
#include "ngraph_remove_copy_froms.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status RemoveCopyFroms(Graph* graph, int graph_id) {
  if (graph_id != 2) {
    return Status::OK();
  }

  // Remove NGraphAssign
  // NGAssign has no other outputs
  vector<Node*> nodes_to_be_removed;
  for(auto node: graph->nodes()){
      if(node->type_string()=="NGraphAssign"){
          Node* merge_node;
          TF_RETURN_IF_ERROR(node->input_node(1, &merge_node));

          cout<<"Found merge_node of type "<<merge_node->type_string()<<endl;

          for(auto out_edge : node->out_edges()){
            // if control edges, add to merge node
            if(out_edge->IsControlEdge()){
              graph->AddEdge(merge_node, out_edge->src_output(), out_edge->dst(),
                   out_edge->dst_input());
              graph->RemoveEdge(out_edge);
            }
            else{
              // other outputs we dont handle yet
              return errors::Internal("Got NGraphAssign with an output that is not control edge");
            }
          }
          nodes_to_be_removed.push_back(node);
      }
  }


  for(auto node: nodes_to_be_removed){
      graph->RemoveNode(node);
  }

  return Status::OK();
}

}  // ngraph_bridge
}  // tensorflow
