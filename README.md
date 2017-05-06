# Bayes networks with TensorFlow

Bayes network implementation for Tensorflow. Work in progress.

## Overarching idea

TensorFlow's main construct is the "computation graph" -- a series of mathematical operations linked by dependencies. Obviously, neural networks are the most newsworthy examples of computation graphs, and TensorFlow was built with neural network in mind. But there are other, equally natural, use-cases for TensorFlow's computation graph concept. One of these is Bayes networks.

Bayes networks comprise nodes (aka vertices) connected by directed edges, just like a neural network. Each node can either be activated or not. The probability of its activation depends on the activation of any other nodes which have edges leading to it. They are called _Bayes_ networks because Bayes Theorem is used to calculate the activations of upstream nodes, given the state of its downstream nodes.

Many efficient algorithms have been developed to perform computations on Bayes networks. Because the nodes have a limited set of dependencies, it is possible to perform many of these computations in parallel. That's where TensorFlow comes in.

The difficulty in automating those algorithms is that deriving the necessary functions requires a lot of (simple) algebraic transformations, where those transformations depend on somewhat complex conditions of the graph. For example, we might be able to simplify a probability statement if it involves nodes that are "d-separated" from each other. Most of the work of these classes involves determining which transformations to make to probability statements, doing those transformations, and generating an abstract syntax tree (AST) of the results. With the AST in-hand, we can dynamically create the necessary functions (and hence, the Tensorflow ops).

More information can be found in the docstrings throughout the `bayes_tensorflow.py` module.
