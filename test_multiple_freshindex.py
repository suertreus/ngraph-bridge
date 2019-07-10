from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import ngraph_bridge


'''
def model():
    a = tf.constant(np.full((2048, 2048), 0.05, dtype=np.float32), name='alpha')
    x = tf.placeholder(tf.float32, [None, 2048], name='x')
    y = tf.placeholder(tf.float32, shape=(2048, 2048), name='y')

    c = a * x
    axpy = c + y
    return axpy, x, y


axpy, x, y = model()
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)
config_ngraph_enabled = ngraph_bridge.update_config(config)

with tf.Session(config=config_ngraph_enabled) as sess:
    for i in range(5):
        _ = sess.run(axpy, feed_dict={x: np.ones((2048, 2048)), y: np.ones((2048, 2048))})
'''


from tensorflow.python.ops.gen_math_ops import rsqrt_grad
import pdb
import os
class MyTest:
    def __init__(self):
        pass

    def run(self):
        for sh in (
        [2, 3],
        [100],
        [3, 2],
        [3, 2, 3],
        [4, 2, 1, 3],):
            self.test_rsqrtgrad(sh)

    def test_rsqrtgrad(self, shape):
        #pdb.set_trace()
        a = tf.placeholder(tf.float32, shape)
        b = tf.placeholder(tf.float32, shape)

        y = np.random.rand(*shape)
        dy = np.random.rand(*shape)

        out = rsqrt_grad(a, b)

        def run_test(sess):
            return sess.run(out, feed_dict={a: y, b: dy})

        assert np.isclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test)).all()
    
    def with_ngraph(self, l, config=tf.ConfigProto()):
        # TODO: Stop grappler on failure (Add fail_on_optimizer_errors=True)
        config = ngraph_bridge.update_config(config)

        ngraph_tf_disable_deassign_clusters = os.environ.pop(
            'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = '1'
        ngraph_bridge.enable()
        with tf.Session(config=config) as sess:
            retval = l(sess)

        os.environ.pop('NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        if ngraph_tf_disable_deassign_clusters is not None:
            os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
                ngraph_tf_disable_deassign_clusters

        return retval

    def without_ngraph(self, l, config=tf.ConfigProto()):
        ngraph_tf_disable_deassign_clusters = os.environ.pop(
            'NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS', None)

        ngraph_bridge.disable()
        with tf.Session(config=config) as sess:
            retval = l(sess)

        if ngraph_tf_disable_deassign_clusters is not None:
            os.environ['NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS'] = \
                ngraph_tf_disable_deassign_clusters

        return retval


obj = MyTest()
obj.run()