import tensorflow as tf
import math, pdb
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
import ngraph_bridge
from google.protobuf import text_format
import numpy as np

#modified from https://github.com/tensorflow/tensorflow/issues/25402


def build_graph(output_name='output'):
    with tf.device(
            '/cpu:0'
    ):  #TODO: this line is needed, else grappler pass does not work
        # Create graph
        x = tf.placeholder(dtype=tf.float32, shape=(10,), name='x')
        y = tf.placeholder(dtype=tf.float32, shape=(10,), name='y')
        z = x + 2 * y
        return tf.identity(z, name=output_name)


def ngraphify(frozen_graph, output_nodes):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(frozen_graph, name="")
    grappler_meta_graph_def = tf.train.export_meta_graph(
        graph_def=graph.as_graph_def(add_shapes=True), graph=graph)

    _to_bytes = lambda s: s.encode("utf-8", errors="surrogateescape")
    output_collection = meta_graph_pb2.CollectionDef()
    output_list = output_collection.node_list.value
    for i in output_nodes:
        if isinstance(i, tf.Tensor):
            output_list.append(_to_bytes(i.name))
        else:
            output_list.append(_to_bytes(i))
    # TODO(laigd): use another key as the outputs are really not train_op.
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
        output_collection)

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE,
        custom_optimizers=[
            rewriter_config_pb2.RewriterConfig.CustomGraphOptimizer(
                name="ngraph-optimizer")
        ])

    session_config_with_trt = tf.ConfigProto()
    session_config_with_trt.graph_options.rewrite_options.CopyFrom(
        rewriter_config)
    frozen_graph = tf_optimizer.OptimizeGraph(
        session_config_with_trt, grappler_meta_graph_def, graph_id=b"tf_graph")
    return frozen_graph


def get_frozen_graph():
    with tf.Graph().as_default():
        z = build_graph()
        # Initialize
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Freeze graph
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_node_names=['output'])
            return frozen_graph


def create_ngraphified_pbtxt(frozen_graph, dumpdir, filename):
    print('Nodes before:')
    [print(n.name, n.op) for n in frozen_graph.node]

    frozen_graph = ngraphify(frozen_graph, output_nodes=['output'])
    tf.io.write_graph(frozen_graph, dumpdir, filename, as_text=True)

    print('----------------------------------------')
    print('Nodes after:')
    [print(n.name, n.op) for n in frozen_graph.node]


def import_pbtxt(pb_filename):
    graph_def = tf.GraphDef()
    with open(pb_filename, "r") as f:
        text_format.Merge(f.read(), graph_def)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def get_tensor(graph, tname):
    return graph.get_tensor_by_name("import/" + tname)


if __name__ == '__main__':
    # Phase 1: Take a graph and dump pbtxt where ngraph encapsulate has been inserted
    #this is a graphdef, ready for inference (var->const)
    frozen_graph = get_frozen_graph()
    dumpdir = 'ngraphified'
    filename = 'ngraphified.pbtxt'
    # dump ngraph enabled pbtxt by running grappler pass
    create_ngraphified_pbtxt(frozen_graph, dumpdir, filename)

    # Phase 2: Run the ngraph enabled graph
    graph = import_pbtxt(dumpdir + '/' + filename)
    with graph.as_default() as g:
        x = get_tensor(g, "x:0")
        y = get_tensor(g, "y:0")
        z = get_tensor(g, "output:0")
        sess = tf.Session()
        res = sess.run(z, feed_dict={x: np.ones([10]), y: np.ones([10])})
        assert all(res == 3.0), "Failed to match expected result"

    print('Bye')
