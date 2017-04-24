"""
Work in progress.

Bayes network classes that will convert a network specification into
a set of Tensorflow ops.
"""

import itertools
import hashlib
import copy
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
        if not isinstance(other, Statement):
            return False
        if isinstance(self, Negation) and not isinstance(other, Negation):
            return False
        if isinstance(self, Conjunction) and not isinstance(other, Conjunction):
            return False
        if self is other:
            return True
        return False  # This is sketchy -- there might be other cases to check
        # raise Exception('Equality tests should be defined for child classes.')

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

    def __eq__(self, other):
        if not isinstance(other, Given):
            return False
        return self.event == other.event and self.given == other.given


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

    def __eq__(self, other):
        if not isinstance(other, Conjunction):
            return False
        return self.conjuncts == other.conjuncts

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

    def connected_nodes(self):
        """
        Returns a list of all the nodes connected (directly or indirectly) to
        the node. In other words, it returns all the nodes in the graph.
        """

        node_list = []
        
        def recurse(node):
            if node in node_list:
                return
            node_list.append(node)
            for child in node.children():
                recurse(child)
            for parent in node.parents():
                recurse(parent)
        
        recurse(self)
        return node_list

    def descendants(self):
        node_list = []
        
        def recurse(node):
            if node in node_list:
                return
            node_list.append(node)
            for child in node.children():
                recurse(child)

        recurse(self)
        return node_list

    def iter_undirected_paths(self, target=None):
        """
        Returns a list of lists, which are paths connecting self to other,
        ignoring the directionality of the edges.
        """
        
        def recurse(step_list):
            current_node = step_list[-1]
            if current_node is target:
                yield step_list
            else:
                next_steps = current_node.children() + current_node.parents()
                next_steps = [i for i in next_steps if i not in step_list]
                if len(next_steps) == 0:
                    yield step_list
                for next_step in next_steps:
                    for i in recurse(copy.copy(step_list) + [next_step]):
                        yield i
        
        for path in recurse([self]):
            yield path
            
    def undirected_paths(self, target=None):
        return list(self.iter_undirected_paths(target=target))

    def is_source(self):
        """
        Tests whether there is no incoming edge.
        """

        return len(self.parents) == 0

    def annotate_path(self, *path):
        """
        Examines each pair nodes in a path and annotates them with the directionality
        of the edges in the original graph. To be used for testing d-separation.
        """
        annotated_path = []
        for index, node in enumerate(path):
            if index == len(path) - 1:
                continue
            next_node = path[index + 1]
            path_triple = (node, '->' if next_node in node.parents() else '<-', next_node,)
            annotated_path.append(path_triple)
        return annotated_path

    def annotated_paths(self, target=None):
        return [self.annotate_path(*path) for path in self.iter_undirected_paths(target=target)]

    @staticmethod
    def path_patterns(annotated_path):
        """
        The d-separation criterion requires us to check whether paths have
        arrows converging on nodes, diverging from them, or chains of arrows
        pointing in the same direction.
        """

        if len(annotated_path) < 2:
            return None
        path_pattern_list = []
        for index, first_triple in enumerate(annotated_path[:-1]):
            second_triple = annotated_path[index + 1]
            quintuple = (
                first_triple[0], first_triple[1], first_triple[2],
                second_triple[1], second_triple[2],)
            first_arrow = first_triple[1]
            second_arrow = second_triple[1]
            pattern = None
            if first_arrow == '<-' and second_arrow == '->':
                pattern = 'diverge'
            elif first_arrow == '->' and second_arrow == '<-':
                pattern = 'converge'
            elif first_arrow == second_arrow:
                pattern = 'chain'
            else:
                raise Exception('This should not happen.')
            path_pattern_list.append((pattern, quintuple,))
        return path_pattern_list

    def all_path_patterns(self, target=None):
        """
        Return all patterns, labeled with 'converge', 'diverge', etc. from ``self``
        to (optional) ``target``.
        """

        return [self.path_patterns(path) for path in self.annotated_paths(target=target)]

    def is_sink(self):
        """
        Tests whether there is no outgoing edge.
        """
        
        return len(self.children) == 0

    def parents(self):
        """
        Return the parent nodes of the current node.
        """

        return [edge.source for edge in self.incoming_edges]

    def children(self):
        """
        Return the child nodes of the current node.
        """

        return [edge.target for edge in self.outgoing_edges]

    def fact_requirements(self):
        """
        This looks at all parents of ``self`` and returns a list of lists.
        Each sublist is a boolean combination of each of the upstream nodes.
        """

        incoming_nodes = self.parents()
        if len(incoming_nodes) == 0:
            return [self, Negation(self)] 
        event_tuples = event_combinations(*incoming_nodes)
        return [
            Given(self, Conjunction(*event_tuple))
            for event_tuple in event_tuples]

    def d_separated(self, z, y):
        """
        Test whether ``z`` d-separates node ``self`` from node ``y``.
        """

        def path_d_separated(path_pattern, z):
            """
            Test whether the ``path_pattern`` is d-separated by ``z``.
            """
            pass

        path_patterns = self.all_path_patterns(target=y)
        for path_pattern in path_patterns:
            if path_pattern is None:
                return False  # Degenerate case -- no triples


        import pdb; pdb.set_trace()


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
d = BayesNode(name='d')
a >> c  # Add an edge connective ``a`` to ``b``
b >> c
d >> a
d >> b
fact_book = FactBook()

fact = Fact(Given(c, a & b), .5)
fact_book += fact

print c.fact_requirements()
print a.fact_requirements()
# Next -- test whether the fact requirements are satisfied by a ``Fact``
# in the ``FactBook`` object.
print fact_book
###

print fact_book.facts[0].statement == c.fact_requirements()[0]
for i in a.undirected_paths():
    print i
