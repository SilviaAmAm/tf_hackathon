"""
This is the solution to the quadratic regression exercise.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Some sample data
train_X = np.arange(-1, 1, 0.01)
rand = np.random.uniform(-0.02, 0.02, train_X.shape)
train_Y = 1.2 * train_X**2 + 0.5*train_X + 1 + rand

num_samples = train_X.shape[0]

# Parameters
learning_rate = 0.01
iterations = 10000

### ------ ** Creating the graph ** -------

# Placeholders - where the data can come into the graph
X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])

# Creating the parameters
a_tf = tf.Variable(np.random.randn(), name='a')
b_tf = tf.Variable(np.random.randn(), name='b')
c_tf = tf.Variable(np.random.randn(), name='c')

# Creating the model: y = ax^2 + bx + c
model = tf.multiply(a_tf, tf.pow(X, 2)) + tf.multiply(b_tf, X) + c_tf

# Creating the cost function
cost_function = 0.5 * (1.0/num_samples) * tf.reduce_sum(tf.pow(model - Y, 2))

# Defining the method to do the minimisation of the cost function
optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

### -------- ** Initialising all the variables ** --------

init = tf.global_variables_initializer()

### -------- ** Starting the session ** ----------


with tf.Session() as sess:
    sess.run(init)

    for i in range(iterations):
        sess.run(optimiser, feed_dict={X: train_X, Y: train_Y})

        if (i+1)%50 == 0:
            c = sess.run(cost_function, feed_dict={X: train_X, Y: train_Y})
            print("Step:", '%04d' % (i+1), "cost=", "{:.9f}".format(c))

    a = sess.run(a_tf)
    b = sess.run(b_tf)
    c = sess.run(c_tf)


### -------- ** Plotting the results ** ------------

sns.set()
fig, ax = plt.subplots()
ax.scatter(train_X, train_Y, marker=".")
y_pred = a*(train_X**2) + b*train_X + c     # Change a, b, c to whatever you have called your parameters!
ax.plot(train_X, y_pred, color=sns.xkcd_rgb["medium green"])
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()