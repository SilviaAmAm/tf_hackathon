"""
This script is an example of how to overfit a cubic function using a small feed forward neural network with only one hidden
layer. It uses tensorboard to visualise what is happening in the network.
"""

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# -------------- ** Generating sample data ** ------------------

x = np.linspace(-2.0, 2.0, 200)
x_col = np.reshape(x, (len(x), 1))
y_col = x_col ** 3

# Network parameters

hidden_neurons = 15
learning_rate = 0.5
iterations = 500


# -------------- ** Building the graph ** ------------------


with tf.name_scope("Data"):
    x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

with tf.name_scope("Weights"):
    weights1 = tf.Variable(tf.random_normal([hidden_neurons, 1]), name="W_in-to-hid")
    bias1 = tf.Variable(tf.zeros([hidden_neurons]), name="b_in-to-hid")
    weights2 = tf.Variable(tf.random_normal([1, hidden_neurons]), name="W_hid-to-out")
    bias2 = tf.Variable(tf.zeros([1]), name="b_hid-to-out")

    # Adding histogram summaries for the weights
    tf.summary.histogram("W_in-to-hid", weights1)
    tf.summary.histogram("W_hid-to-out", weights2)

with tf.name_scope("Model"):
    z1 = tf.add(tf.matmul(x_ph, tf.transpose(weights1)), bias1)  # output of layer1, size = n_sample x hidden_neurons
    h1 = tf.nn.sigmoid(z1)
    model = tf.add(tf.matmul(h1, tf.transpose(weights2)), bias2)  # output of last layer, size = n_samples x 1

with tf.name_scope("Cost-function"):
    cost = tf.reduce_mean(tf.nn.l2_loss(t=(model - y_ph)))

    # Adding scalar summary for the cost
    tf.summary.scalar('cost', cost)

# Optimisation operation
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# -------------- ** Running the graph ** ------------------

# Initialisation of the model
init = tf.global_variables_initializer()

# Joining all the summaries so that they don't have to be passed individually to the FileWriter object
merge = tf.summary.merge_all()

training_cost = []

# Running the graph
with tf.Session() as sess:
    sess.run(init)

    # Instantiating a SummaryWriter object
    summary_writer = tf.summary.FileWriter(logdir="tensorboard", graph=sess.graph)

    for iter in range(iterations):
        opt = sess.run(optimizer, feed_dict={x_ph: x_col, y_ph: y_col})
        # Running the merge summary operation
        merged_summaries = sess.run(merge, feed_dict={x_ph: x_col, y_ph: y_col})
        # Passing the merged summaries to the FileWriter
        summary_writer.add_summary(merged_summaries, global_step=iter)

    y_pred = sess.run(model, feed_dict={x_ph: x_col})

# -------- ** Plotting the predictions ** -------------

fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.scatter(x, y_col, label="original", marker="o")
ax2.scatter(x, y_pred, label="predictions", marker="o")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

plt.show()