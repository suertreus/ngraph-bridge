import tensorflow as tf
import numpy as np
import ngraph_bridge


x = tf.placeholder(float, shape=(4,2))
a = tf.placeholder(tf.int32, shape=(1,1))

y = tf.math.reduce_mean(x, tf.math.reduce_sum(a))


sess = tf.Session()
res = sess.run(y, feed_dict={x:np.array([[1,3], [5,7], [9,11], [13,15]]), a:np.ones((1,1))})
print(res)
