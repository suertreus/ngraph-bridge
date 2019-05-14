# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import ctypes

import numpy as np
import tensorflow as tf
import ngraph_bridge

print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

# Setup TensorBoard
graph_location = "/tmp/" + getpass.getuser() + "/tensorboard-logs/test"
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)

# Define the data
x = tf.placeholder(tf.float32, shape=None, name='x')
y = tf.placeholder(tf.float32, shape=None, name='y')
perm = tf.placeholder(tf.int32, shape=(None), name='perm')

x_trans = tf.transpose(x+y,perm)

# Configure the session
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    inter_op_parallelism_threads=1)

# Create session and run
with tf.Session(config=config) as sess:
    for (shape,perms) in [((2, 2), [[0, 1], [1, 0]]),
                          ((2, 3), [[1, 0], [0, 1]]),
                          ((4, 4), [[0, 1], [1, 0]])]:
        print("Shape:", shape)
        print("====================")
        for p in perms:
            n_elems = np.prod(shape)
            x_data = np.linspace(start=1, stop=n_elems, num=n_elems)
            x_data.shape = shape
            y_data = np.zeros(shape)

            result_x_trans = sess.run(x_trans,
                                      feed_dict={
                                          x: x_data,
                                          y: y_data,
                                          perm: p
                                      })
            print("Permutation:", p)
            print("Result:")
            print(result_x_trans)

train_writer.add_graph(tf.get_default_graph())
tf.train.write_graph(
    tf.get_default_graph(), '.', 'transpose.pbtxt', as_text=True)
