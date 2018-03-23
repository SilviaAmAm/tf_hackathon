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

![Example of a graph](presentation_material/example_graph1.png)
 



