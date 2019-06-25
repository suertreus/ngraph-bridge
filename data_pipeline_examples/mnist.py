#From here: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/tensorflow_dataset_api.py

""" TensorFlow Dataset API.

In this example, we will show how to load numpy array data into the new 
TensorFlow 'Dataset' API. The Dataset API implements an optimized data pipeline
with queues, that make data processing and training faster (especially on GPU).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function

import tensorflow as tf
#import ngraph_bridge
# Import MNIST data (Numpy format)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import time

# Parameters
learning_rate = 0.001
num_steps = 3
batch_size = 16
display_step = 1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

#sess = tf.Session(config=ngraph_bridge.update_config(tf.ConfigProto()))
sess = tf.Session()
def preprocess_fn(image, label):
    with tf.variable_scope("preprocess"):
        mean = tf.math.reduce_mean(image)
        devs_squared = tf.square(image - mean)
        var = tf.reduce_mean(devs_squared)
        final = (image-mean)/var
    return final, label

def create_dataset():
    with tf.variable_scope("datasets"):
        # Create a dataset tensor from the images and the labels
        #sarkars: only reading 1024 images to keep the pbtxts small
        dataset = tf.data.Dataset.from_tensor_slices(
            (mnist.train.images[0:64], mnist.train.labels[0:64]))
        with tf.variable_scope("repeat"):
            dataset = dataset.repeat() # Automatically refill the data queue when empty
        with tf.variable_scope("map"):
            dataset = dataset.map(preprocess_fn, num_parallel_calls=4)
        with tf.variable_scope("batch"):
            dataset = dataset.batch(batch_size) # Create batches of data
        #with tf.variable_scope("prefetch"):
        #    dataset = dataset.prefetch(batch_size) # Prefetch data for faster consumption # TODO: try 0 or -1 in prefetch
        # TODO (sarkars): tf.data.experimental.prefetch_to_device
        with tf.variable_scope("prefetch_to_device"):    
            dataset = dataset.apply(tf.data.experimental.prefetch_to_device('cpu'))

        with tf.variable_scope("iterator"):
            iterator = dataset.make_initializable_iterator() # Create an iterator over the dataset
        sess.run(iterator.initializer) # Initialize the iterator
        # TODO (sarkars): this dataset subgraph has fixed batchsize. maybe make batchsize variable by passing batchsize as placeholder
        # prefetch_to_device
        with tf.variable_scope("X_Y"):
            X, Y = iterator.get_next() # Neural Net Input (images, labels)
    return X, Y

# Create model
def conv_net(x, y, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv2 = tf.layers.conv2d(x, 4, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 100)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

        out = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        out = tf.reduce_mean(tf.cast(out, tf.float32))

    return out

X, Y = create_dataset()

accuracy = conv_net(X, Y, n_classes, dropout, reuse=False, is_training=False)

init = tf.global_variables_initializer() # Initialize the variables (i.e. assign their default value)

sess.run(init) # Run the initializer

summ_writer = tf.summary.FileWriter('summary', sess.graph)

for step in range(1, num_steps + 1):

    # Run optimization
    #sess.run(train_op)
    acc = sess.run([accuracy])

    if step % display_step == 0 or step == 1:
        print("Step " + str(step) + ", Accuracy= " + str(acc))
    #time.sleep(10)

