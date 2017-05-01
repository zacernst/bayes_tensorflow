"""
Work in progress.

Bayes network classes that will convert a network specification into
a set of TensorFlow ops and a computation graph.
"""

from types import *
import itertools
import hashlib
import copy
import functools
import tensorflow as tf


def dict_to_function(arg_dict):
    """
    We need functions for Tensorflow ops, so we will use this function
    to dynamically create functions from dictionaries.
    """

    def inner_function(lookup, **inner_dict):
        return inner_dict[lookup]

    new_function = functools.partial(inner_function, **arg_dict)
    return new_function


class Arithmetic(object):

    def __add__(self, other):
        return Add(self, other)

    def __mult__(self, other):
        return Multiply(self, other)


class Inverse(Arithmetic):

    def __init__(self, expression):
        if not isinstance(expression, Arithmetic):
            raise Exception('Inverse applies only to ``Arithmetic`` objects')
        self.expression = expression


class Add(Arithmetic):

    def __init__(self, addend_1, addend_2):
        if not isinstance(addend_1, Arithmetic) or not isinstance(addend_2, Arithmetic):
            raise Exception('Add only defined for ``Arithmetic`` objects')
        self.addend_1 = addend_1
        self.addend_2 = addend_2

    def __repr__(self):

        return '({addend_1} + {addend_2})'.format(
            addend_1=str(addend_1), addend_2=str(addend_2))


class Multiply(Arithmetic):

    def __init__(self, multiplicand_1, multiplicand_2):
        if (not isinstance(multiplicand_1, Arithmetic) or
                not isinstance(multiplicand_2, Arithmetic)):
            raise Exception('Multiply only defined for ``Arithmetic`` objects')


class Probability(Arithmetic):

    def __init__(self, statement):
        if not isinstance(statement, Statement):
            raise Exception('Probability applies only to ``Statement``s')
        self.statement = statement

    def __repr__(self):
        return 'P({statement})'.format(statement=str(self.statement))


class Statement(object):
    """
    This is a mix-in class for allowing boolean connectives to be used.
    Any connective other than conjunction and negation is immediately
    translated to one using only conjunction and negation.
    """

    def __and__(self, other):
        if self is other:
            return self
        return Conjunction(self, other)

    def __or__(self, other):
        return ~(~self & ~other)

    def __invert__(self):
        return Negation(self)

    def __gt__(self, other):
        return ~(self & ~other)

    def __eq__(self, other):
        """
        Equality test.

        Not all cases are accounted for yet.
        """
        if not isinstance(other, Statement):
            return False
        if isinstance(self, Conjunction) and len(self.conjuncts) == 1:
            left = self.conjuncts[0]
        else:
            left = self
        if isinstance(other, Conjunction) and len(other.conjuncts) == 1:
            right = other.conjuncts[0]
        else:
            right = other
        if isinstance(left, Negation) != isinstance(right, Negation):
            return False
        if isinstance(left, Conjunction) and not isinstance(right, Conjunction):
            return False
        if left is right:
            return True
        if isinstance(left, Negation) and isinstance(right, Negation):
            return left.statement is right.statement
        return False  # This is sketchy -- there might be other cases to check

    def is_atomic(self):
        return not isinstance(self, (Negation, Conjunction,))


class EqualsBook(object):
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

    def __iter__(self):
        for fact in self.facts:
            yield fact


class Equals(object):
    """
    A ``Equals`` is an assertion that an event has a probability of being true.
    """

    def __init__(self, statement, probability):
        self.statement = statement
        self.probability = probability

    def __repr__(self):
        return str(self.statement) + ' = ' + str(self.probability)


class Given(Statement):
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
    for boolean_combination in itertools.product(
            *([[True, False]] * number_of_events)):
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
        return (
            '(' + ' & '.join(
                [str(conjunct) for conjunct in self.conjuncts]) + ')')

    def __eq__(self, other):
        if not isinstance(other, Conjunction):
            return False
        return self.conjuncts == other.conjuncts


class BayesNode(Statement):
    """
    This is the main class for the module.
    It represents a vertex in the Bayes network.
    """

    def __init__(
        self,
        activation_probability=None,
        state=None,
        fact_book=None,
        pinned=False,
            name=None):
        self.fact_book = fact_book
        self.incoming_edges = []
        self.outgoing_edges = []
        self.activation_probability = activation_probability
        self.state = state
        self.pinned = pinned
        self.name = name or hashlib.md5(str(id(self))).hexdigest()
        self.parent_fact_lookup = None
        # For now, we are not calling the parent class's ``__init__`` method.
        # super(BayesNode, self).__init__()

    def top_down_eval(self):
        # evaluate the value of self, given the parents only
        pass

    def parent_requirements_satisfied(self):
        # True if the ``EqualsBook`` has all necessary data on parents
        parent_requirements = self.fact_requirements()
        satisfied_requirements_tally = 0
        if self.is_source():
            return True  # Vacuously satisfied
        for fact in self.fact_book:
            statement = fact.statement
            # Only ``Given`` statements are relevant
            if not isinstance(statement, Probability):
                continue
            if not isinstance(statement.statement, Given):
                continue
            given_statement = statement.statement
            if given_statement in parent_requirements:
                satisfied_requirements_tally += 1
        return satisfied_requirements_tally == len(parent_requirements)

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
                if len(next_steps) == 0 and step_list[-1] is target or target is None:
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

        return len(self.parents()) == 0

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
            path_triple = (
                node, '->' if next_node in node.children()
                else '<-', next_node,)
            annotated_path.append(path_triple)
        return annotated_path

    def annotated_paths(self, target=None):
        return [
            self.annotate_path(*path) for path in
            self.iter_undirected_paths(target=target)]

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

        return [
            self.path_patterns(path) for path in
            self.annotated_paths(target=target)]

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
            Given(self,
                Conjunction(*event_tuple) if len(event_tuple) > 1 else
                event_tuple[0])
            for event_tuple in event_tuples]

    def relevant_parent_fact_dict(self):
        parent_requirements = self.fact_requirements()
        relevant_fact_requirements = [
            fact for fact in self.fact_book
            if fact.statement in parent_requirements]
        relevant_fact_dict = {
            fact.statement: fact.probability for fact in
            relevant_fact_requirements}
        return relevant_fact_dict

    def create_parent_fact_function(self):
        """
        Retrieves all the relevant facts from ``self.fact_book``,
        creates a dictionary for lookups, then returns a function
        that replaces dictionary lookups with function calls.
        """

        return dict_to_function(self.relevant_parent_fact_dict())

    def d_separated(self, z, y):
        """
        Test whether ``z`` d-separates node ``self`` from node ``y``.
        """

        def path_d_separated(path_pattern, z):
            """
            Test whether the ``path_pattern`` is d-separated by ``z``.
            """

            # Verify that we're handling the None case correctly
            if path_pattern is None:  # Degenerate case
                return False
            
            for category, quintuple in path_pattern:
                w = quintuple[2]
                if category == 'converge':
                    if w is z or w in z.descendants():
                        return True
                elif category == 'chain':
                    if w is z:
                        return True
                elif category == 'diverge':
                    if w is z:
                        return True
                else:
                    raise Exception('This should never happen.')
            return False  # No w satisfying d-separation was found

        path_patterns = self.all_path_patterns(target=y)
        return all(
            path_d_separated(path_pattern, z) for
            path_pattern in path_patterns)


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

def sandbox():
    """
    just for testing
    """
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
    e = BayesNode(name='e')

    a >> b
    a >> c
    b >> d
    c >> d
    d >> e

    fact_book = EqualsBook()

    fact = Equals(Probability(Given(c, a & b)), .5)
    fact_book += fact
    fact = Equals(Probability(Given(b, a)), .2)
    fact_book += fact
    fact = Equals(Probability(Given(b, ~a)), .3)
    fact_book += fact
    print c.fact_requirements()
    print a.fact_requirements()
    # Next -- test whether the fact requirements are satisfied by a ``Equals``
    # in the ``EqualsBook`` object.
    print fact_book
    ###

    print fact_book.facts[0].statement == c.fact_requirements()[0]
    for i in a.undirected_paths():
        print i

    b.fact_book = fact_book
    print b.parent_requirements_satisfied()

    if not b.parent_requirements_satisfied():
        raise Exception()

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    sandbox()
