"""
This script is an example of how to overfit a cubic function using a small feed forward neural network with only one hidden
layer.  This script introduces how to use the tf.data.Dataset class.
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
iterations = 200
batch_size = 100
n_batches = int(len(x)/batch_size)


# -------------- ** Building the graph ** ------------------

with tf.name_scope("Data"):
    x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    dataset = tf.data.Dataset.from_tensor_slices((x_ph, y_ph))
    dataset = dataset.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    tf_x, tf_y = iterator.get_next()

with tf.name_scope("Weights"):
    weights1 = tf.Variable(tf.random_normal([hidden_neurons, 1]), name="W_in-to-hid")
    bias1 = tf.Variable(tf.zeros([hidden_neurons]), name="b_in-to-hid")
    weights2 = tf.Variable(tf.random_normal([1, hidden_neurons]), name="W_hid-to-out")
    bias2 = tf.Variable(tf.zeros([1]), name="b_hid-to-out")

    parameters = [weights1, bias1, weights2, bias2]

with tf.name_scope("Model"):
    z = tf.add(tf.matmul(tf_x, tf.transpose(weights1)), bias1)  # output of layer1, size = n_sample x hidden_neurons
    act = tf.nn.sigmoid(z)
    model = tf.add(tf.matmul(act, tf.transpose(weights2)), bias2)  # output of last layer, size = n_samples x 1

with tf.name_scope("Cost-function"):
    cost = tf.reduce_mean(tf.nn.l2_loss(t=(model - tf_y)))

# Optimisation operation
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# -------------- ** Running the graph ** ------------------

# Initialisation of the model
init = tf.global_variables_initializer()
iterator_init = iterator.make_initializer(dataset)

training_cost = []

# Running the graph
with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator_init, feed_dict={x_ph:x_col, y_ph:y_col})

    for iter in range(iterations):
        sess.run(iterator_init, feed_dict={x_ph: x_col, y_ph: y_col})
        for batch in range(n_batches):
            opt, c = sess.run([optimizer, cost])
            training_cost.append(c)

    # Obtaining predictions for each batch of data
    y_pred = []

    sess.run(iterator_init, feed_dict={x_ph: x_col, y_ph: y_col})

    for i in range(n_batches):
        y_pred.append(sess.run(model))

    y_pred = np.concatenate(y_pred, axis=0)


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