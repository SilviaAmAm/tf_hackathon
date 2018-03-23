"""
This script shows how to build a simple tensorflow graph and how to run it in a session to calculate the sum of 2 numbers.
"""

import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    a = tf.placeholder(dtype=tf.int32, shape=[1])
    b = tf.placeholder(dtype=tf.int32, shape=[1])
    sum_ab = tf.add(a, b)

with tf.Session(graph=graph) as sess:
    result = sess.run(sum_ab, feed_dict={a:[1], b:[2]})

print(result)
