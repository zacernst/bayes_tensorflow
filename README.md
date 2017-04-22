# Bayes networks with TensorFlow

Bayes network implementation for Tensorflow. Work in progress.

## Overarching idea

TensorFlow's main construct is the "computation graph" -- a series of mathematical operations linked by dependencies. Obviously, neural networks are the most newsworthy examples of computation graphs, and TensorFlow was built with neural network in mind. But there are other, equally natural, use-cases for TensorFlow's computation graph concept. One of these is Bayes networks.

Bayes networks comprise nodes (aka vertices) connected by directed edges, just like a neural network. Each node can either be activated or not. The probability of its activation depends on the activation of any other nodes which have edges leading to it. They are called _Bayes_ networks because Bayes Theorem is used to calculate the activations of upstream nodes, given the state of its downstream nodes.

Many efficient algorithms have been developed to efficiently perform computations on Bayes networks. Because the nodes have a limited set of dependencies, it is possible to perform many of these computations in parallel. That's where TensorFlow comes in.

## Design

Each Bayes network in this module consists of two parts: A set of `BayesNode` objects connected by `BayesEdge` objects, and a so-called `FactBook` object that contains information governing the activations of the nodes.

For example, suppose you have the simplest non-trivial Bayes network, consisting of two nodes: `a` and `b`, where there is an edge connecting `a` to `b`. In order to specify `b`'s activation, we need to know the probability that `b` is activated, given that `a` is, as well as the probability that `b` is activated, given that `a` is not. In the vocabulary of this module, those data are "facts", which live in the `FactBook`. It is up to the user to put the required facts into the `FactBook`, and then one can begin to query the network.

To make this network, we would create the two nodes like so:

```
a = BayesNode(name='a')
b = BayesNode(name='b')
```

Then we'd create the `FactBook` and the relevant `Fact`:

```
fact_book = FactBook()
fact_1 = Fact(Given(b, a), .2)
fact_2 = Fact(Given(b, ~a), .5)
```

These two facts specify that the probability of `b` being activated when `a` is activated is .2, and it's .5 when `a` is not activated.

Now we put the facts into the `FactBook`:

```
fact_book += fact_1
fact_book += fact_2
```

