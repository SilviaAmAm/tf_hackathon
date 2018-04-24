"""
This script is an example of how to overfit a cubic function using a small feed forward neural network with only one hidden
layer.
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

# Input data
x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Creating the weights
weights1 = tf.Variable(tf.random_normal([hidden_neurons, 1]), name="W_in-to-hid")
bias1 = tf.Variable(tf.zeros([hidden_neurons]), name="b_in-to-hid")
weights2 = tf.Variable(tf.random_normal([1, hidden_neurons]), name="W_hid-to-out")
bias2 = tf.Variable(tf.zeros([1]), name="b_hid-to-out")

# Model
z2 = tf.add(tf.matmul(x_ph, tf.transpose(weights1)), bias1)  # output of layer1, size = n_sample x hidden_neurons
h2 = tf.nn.sigmoid(z2)
model = tf.add(tf.matmul(h2, tf.transpose(weights2)), bias2)  # output of last layer, size = n_samples x 1

# Cost function
cost = tf.reduce_mean(tf.nn.l2_loss(t=(model - y_ph)))

# Optimisation operation
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# -------------- ** Running the graph ** ------------------

# Initialisation of the model
init = tf.global_variables_initializer()

training_cost = []

# Running the graph
with tf.Session() as sess:
    sess.run(init)

    for iter in range(iterations):
        opt, c = sess.run([optimizer, cost], feed_dict={x_ph: x_col, y_ph: y_col})
        training_cost.append(c)

    y_pred = sess.run(model, feed_dict={x_ph: x_col})

# -------- ** Plotting the results ** ------------

sns.set()
fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.scatter(range(len(training_cost)), training_cost, label="Training cost", marker="o")
ax1.set_xlabel('iterations')
ax1.set_ylabel('training cost')
ax1.legend()

plt.show()

# -------- ** Plotting the predictions ** -------------

fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.scatter(x, y_col, label="original", marker="o")
ax2.scatter(x, y_pred, label="predictions", marker="o")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

plt.show()