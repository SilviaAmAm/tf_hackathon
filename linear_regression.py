"""
This script shows how to do linear regression with tensorflow.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Generating some sample data
train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
num_samples = train_X.shape[0]

# Parameters
learning_rate = 0.01
iterations = 2000

### ------ ** Creating the graph ** -------

# Placeholders - where the data can come into the graph
X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])

# Creating the parameters for y = mx + c
c = tf.Variable(np.random.randn(), name='c')
m = tf.Variable(np.random.randn(), name='m')

# Creating the model: y = mx + c
model = tf.add(c, tf.multiply(m, X))

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
            cost = sess.run(cost_function, feed_dict={X: train_X, Y: train_Y})
            print("Step:", '%04d' % (i+1), "cost=", "{:.9f}".format(cost), "m=", sess.run(m), "c=", sess.run(c))

    slope = sess.run(m)
    intercept = sess.run(c)

### -------- ** Plotting the results ** ------------

sns.set()
fig, ax = plt.subplots()
ax.scatter(train_X, train_Y)
plot_x = [np.amin(train_X)-2, np.amax(train_X)+2]
plot_y = [slope*plot_x[0] + intercept , slope*plot_x[1] + intercept ]
ax.plot(plot_x, plot_y, color=sns.xkcd_rgb["medium green"])
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()