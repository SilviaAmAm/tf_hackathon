# Introduction to TensorFlow

Hackathon 2018

---

## What is Tensorflow?

- Software library for numerical computation 

- _... but wait, why not just use Numpy?_ |

---

## Tensorflow vs Numpy

**Numpy**: expensive computations are done outside of python with very efficient code. 

_Problem_: overhead for switching in and out of python for every operation. 

**Tensorflow**: avoids this overhead by using sets of interacting operations that can be run all outside of python.

---

## Data flow graphs

Representations of the data dependencies between a number of operations.

![Example of a graph](presentation_material/example_graph1.png =200)


---

## The basics

There are two main parts to a Tensorflow program:

1. Building the graph
2. Running the graph

---

## Building the graph

* Use "placeholders" for where the data will come in
* Build the operations

```python
import tensorflow as tf

a = tf.placeholder(dtype=tf.int32, shape=[1])
b = tf.placeholder(dtype=tf.int32, shape=[1])
sum_ab = tf.add(a, b)
``` 
---






