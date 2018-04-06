"""
In this exercise you will use TensorFlow to do a quadratic regression.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Generating ome sample data
train_X = np.arange(-1, 1, 0.01)
rand = np.random.uniform(-0.02, 0.02, train_X.shape)
train_Y = 1.2 * train_X**2 + 0.5*train_X + 1 + rand


# Parameters
learning_rate = 0.01
iterations = 2000

### ------ ** Creating the graph ** -------

# ADD CODE HERE

### -------- ** Initialising all the variables ** --------

# ADD CODE HERE

### -------- ** Starting the session ** ----------

# ADD CODE HERE


### -------- ** Plotting the results ** ------------

sns.set()
fig, ax = plt.subplots()
ax.scatter(train_X, train_Y)
y_pred = a*(train_X**2) + b*train_X + c     # Change a, b, c to whatever you have called your parameters!
ax.plot(train_X, y_pred, color=sns.xkcd_rgb["medium green"])
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()