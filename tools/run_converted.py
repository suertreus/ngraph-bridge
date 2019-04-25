import tensorflow as tf
import ngraph_bridge
from google.protobuf import text_format
import numpy as np


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
    dumpdir = 'ngraphified'
    filename = 'ngraphified.pbtxt'

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
