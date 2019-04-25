import tensorflow as tf
import math, pdb
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
import ngraph_bridge

#modified from https://github.com/tensorflow/tensorflow/issues/25402

def build_graph(output_name='output'):
    with tf.device('/cpu:0'):  #TODO: this line is needed, else grappler pass does not work
        # Create graph
        x = tf.placeholder(dtype=tf.float32, shape=(10,), name='x')
        y = tf.placeholder(dtype=tf.float32, shape=(10,), name='y')
        z = x + 2*y
        return tf.identity(z, name=output_name)

def ngraphify(frozen_graph, output_nodes):
  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(frozen_graph, name="")
  grappler_meta_graph_def = tf.train.export_meta_graph(graph_def=graph.as_graph_def(add_shapes=True), graph=graph)

  _to_bytes = lambda s: s.encode("utf-8", errors="surrogateescape")
  output_collection = meta_graph_pb2.CollectionDef()
  output_list = output_collection.node_list.value
  for i in output_nodes:
    if isinstance(i, tf.Tensor):
      output_list.append(_to_bytes(i.name))
    else:
      output_list.append(_to_bytes(i))
  # TODO(laigd): use another key as the outputs are really not train_op.
  grappler_meta_graph_def.collection_def["train_op"].CopyFrom(output_collection)
  
  rewriter_config = rewriter_config_pb2.RewriterConfig(meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE, custom_optimizers=[rewriter_config_pb2.RewriterConfig.CustomGraphOptimizer(name="ngraph-optimizer")])

  session_config_with_trt = tf.ConfigProto()
  session_config_with_trt.graph_options.rewrite_options.CopyFrom(
      rewriter_config)
  frozen_graph = tf_optimizer.OptimizeGraph(session_config_with_trt, grappler_meta_graph_def, graph_id=b"tf_graph")
  return frozen_graph

def get_frozen_graph():
    with tf.Graph().as_default():
        z = build_graph()
        # Initialize
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Freeze graph
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                output_node_names=['output'])
            return frozen_graph


def create_ngraphified_pbtxt(frozen_graph):
  print('Nodes before:')
  [print(n.name, n.op) for n in frozen_graph.node]

  frozen_graph = ngraphify(frozen_graph, output_nodes=['output'])
  tf.io.write_graph(
    frozen_graph,
    'ngraphified',
    'ngraphified.pbtxt',
    as_text=True
  )

  print('----------------------------------------')
  print('Nodes after:')
  [print(n.name, n.op) for n in frozen_graph.node]


if __name__ == '__main__':
  frozen_graph = get_frozen_graph() #this is a graphdef, ready for inference (var->const)
  create_ngraphified_pbtxt(frozen_graph)

  






