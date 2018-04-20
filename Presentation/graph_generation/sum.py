import numpy as np
import tensorflow as tf

# Constructing the graph

with tf.name_scope("Inputs"):
    a = tf.placeholder(dtype=tf.int32, shape=[2,2], name="A")
    b = tf.placeholder(dtype=tf.int32, shape=[2,2], name="B")
    d = tf.placeholder(dtype=tf.int32, shape=[2,2], name="C")

with tf.name_scope("Sum"):
    c = tf.add(a, b, name="Sum")

with tf.name_scope("Subtract"):
    e = tf.subtract(c, d, name="Subtract")

# Running the graph

init = tf.global_variables_initializer()

sess = tf.Session()

some_data_1 = np.ones((2,2))
some_data_2 = np.ones((2,2))

result = sess.run(c, feed_dict={a:some_data_1, b:some_data_2})

# Tensorboard

summary_writer = tf.summary.FileWriter(logdir="tensorboard", graph=sess.graph)

print(result)
