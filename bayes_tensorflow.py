"""
Work in progress.

Bayes network classes that will convert a network specification into
a set of Tensorflow ops.
"""

import itertools
import hashlib
import tensorflow as tf


class Statement(object):
    """
    This is a mix-in class for allowing boolean connectives to be used.
    Any connective other than conjunction and negation is immediately
    translated to one using only conjunction and negation.
    """

    def __and__(self, other):
        return Conjunction(self, other)

    def __or__(self, other):
        return ~(~self & ~other)

    def __invert__(self):
        return Negation(self)

    def __gt__(self, other):
        return ~(self & ~other)

    def __eq__(self, other):
        raise Exception('Equality tests should be defined for child classes.')

    def is_atomic(self):
        return not isinstance(self, (Negation, Conjunction,))


class FactBook(object):
    """
    Holds a list of facts.
    """

    def __init__(self):
        self.facts = []

    def __lt__(self, other):
        self.facts.append(other)

    def __contains__(self, other):
        pass

    def __iadd__(self, other):
        self.facts.append(other)
        return self

    def __repr__(self):
        return '\n'.join([str(i) for i in self.facts])


class Fact(object):
    """
    A ``Fact`` is an assertion that an event has a probability of being true.
    """

    def __init__(self, statement, probability):
        self.statement = statement
        self.probability = probability

    def __repr__(self):
        return 'P(' + str(self.statement) + ') = ' + str(self.probability)


class Given(object):
    """
    Expressions like ``P(x|y)`` are not events or states; they're used only in
    probability assignments. So they get their own class.
    """

    def __init__(self, event, given):
        self.event = event
        self.given = given

    def __repr__(self):
        return ' '.join([str(self.event), '|', str(self.given)])


def event_combinations(*events):
    """
    Returns a list of lists of statements. Each sublist is a complete description
    of the set of ``*events``..
    """

    number_of_events = len(events)
    out = []
    for boolean_combination in itertools.product(*([[True, False]] * number_of_events)):
        out.append(
            [events[i] if boolean else Negation(events[i])
             for i, boolean in enumerate(boolean_combination)])
    return out


class Negation(Statement):
    """
    A negated statement.
    """

    def __init__(self, statement):
        self.statement = statement

    def __repr__(self):
        return '~' + str(self.statement)


class Conjunction(Statement):
    """
    A list of conjuncts.
    """

    def __init__(self, *args):
        self.conjuncts = args

    def __repr__(self):
        return '(' + ' & '.join([str(conjunct) for conjunct in self.conjuncts]) + ')'


class BayesNode(Statement):
    """
    This is the main class for the module. It represents a vertex in the Bayes network.
    """

    def __init__(
        self,
        activation_probability=None,
        state=None,
        pinned=False,
            name=None):
        self.incoming_edges = []
        self.outgoing_edges = []
        self.activation_probability = activation_probability
        self.state = state
        self.pinned = pinned
        self.name = name or hashlib.md5(str(id(self))).hexdigest()
        # For now, we are not calling the parent class's ``__init__`` method.
        # super(BayesNode, self).__init__()

    def __repr__(self):
        return str(self.name)

    def __rshift__(self, other):
        """
        Create an edge from self to other.
        """

        edge = BayesEdge(self, other)
        self.outgoing_edges.append(edge)
        other.incoming_edges.append(edge)

    def fact_requirements(self):
        """
        This looks at all parents of ``self`` and returns a list of lists.
        Each sublist is a boolean combination of each of the upstream nodes.
        """

        incoming_nodes = [edge.source for edge in self.incoming_edges]
        event_tuples = event_combinations(*incoming_nodes)
        return [
            Given(self, Conjunction(*event_tuple))
            for event_tuple in event_tuples]
        

class BayesEdge(object):
    """
    An edge connecting source to target. This shouldn't be called
    directly -- ``BayesNode`` objects should be connected to each
    other, and this constructor will be called by the ``__add__``
    method in ``BayesNode``.
    """
    def __init__(self, source, target):
        self.source = source
        self.target = target


x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

with tf.Session() as sess:
    init = tf.global_variables_initializer()  # node to initialize the rest
    init.run()  # Run the initializer for all variables
    result = f.eval()

print result

a = BayesNode(name='a')
b = BayesNode(name='b')
c = BayesNode(name='c')
a >> c  # Add an edge connective ``a`` to ``b``
b >> c
fact_book = FactBook()

fact = Fact(Given(c, a & b), .5)
fact_book += fact

print c.fact_requirements()
# Next -- test whether the fact requirements are satisfied by a ``Fact``
# in the ``FactBook`` object.
