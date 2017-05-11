"""
Work in progress.

Bayes network classes that will convert a network specification into
a set of TensorFlow ops and a computation graph.

The main work of the classes is to move from a directed graph to a
set of formulas expressing what's known about the probability distribution
of the events represented in the graph. When those formulas are derived,
they can be associated with the corresponding nodes in the graph, and used
for local message passing, Monte Carlo simulations, and translation into
Tensorflow ops to be sent to the cpu or gpu.

The flow of information through the classes is as follows: The user will
create ``BayesNodes`` and connect them into the desired graph structure.
Depending on the way in which the graph will be queried, a set of
probability statements will be required. These take the form of ``Probability``
objects, which associate a ``Statement`` (which looks like a formula from
propositional logic, where the variables correspond to graph nodes) with
a value, e.g. ``P(a & ~b) = .5``. The statements are stored in a container
called a ``FactBook`` which can be queried. At that point, we have enough
information to automatically generate an abstract syntax tree of all the
functions that are necessary for computing probabilities on the graph. The
AST can be transformed into Python functions dynamically, or into
Tensorflow ops.

Current state: We can define a graph, and we can define what's known about
direct causal influences (using the ``FactBook`` class). Basic facts about
the graph can be calculated, such as which nodes d-separate arbitrary pairs
of nodes. Message passing is underway, we can generate the expressions for
any node's ``alpha`` and ``lambda`` messages (following Pearl 1988). The
portions of the AST which have been defined can be traversed with the
usual ``__iter__`` method.

Next step is to be able to traverse the trees of all the message passing
functions and conditionally replace subexpressions based on the structure
of the tree -- for example, rewriting expressions of the form ``P(a & b | c)``
if ``c`` d-separates ``a`` from ``b`` in the directed graph.
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
    """
    This is a mix-in class for enabling arithmetic or algebraic functions over
    the objects.
    """

    def __add__(self, other):
        """
        Allows us to use familiar ``+`` to denote addition.
        """

        return Add(self, other)

    def __mul__(self, other):
        """
        Allows us to use familiar ``*`` for multiplication.
        """

        return Multiply(self, other)


class Sigma(Arithmetic):
    """
    Summation over a list of ``Arithmetic`` objects.
    """

    def __init__(self, *values):
        self.values = values

    def __repr__(self):
        return '(Sigma: ' + ', '.join([str(value) for value in self.values]) + ')'

    def __iter__(self):
        for value in self.values:
            yield value
            if hasattr(value, '__iter__'):
                for i in value:
                    yield i


class Pi(Arithmetic):
    """
    Multiplication over a list of ``Arithmetic`` objects.
    """

    def __init__(self, *values):
        self.values = values

    def __repr__(self):
        return '(Pi: ' + ', '.join([str(value) for value in self.values]) + ')'

    def __iter__(self):
        for value in self.values:
            yield value
            if hasattr(value, '__iter__'):
                for i in value:
                    yield i


def bayes(given_probability):
    """
    Takes P(a | b) and returns equivalent probability using Bayes Theorem.
    Result is alpha * P(a) * Likelihood(a).
    """

    given = given_probability.statement
    return (
        alpha(given.event, given.given) * Probability(given.event) *
        Probability(Given(given.given, given.event)))


class One(Arithmetic):
    """
    Could be handy for base case in recursive multiplications.
    """
    pass


class Number(Arithmetic):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)


class Inverse(Arithmetic):

    def __init__(self, expression):
        if not isinstance(expression, Arithmetic):
            raise Exception('Inverse applies only to ``Arithmetic`` objects')
        self.expression = expression

    def __repr__(self):
        return '1 / ' + str(self.expression)

    def __iter__(self):
        yield self.expression
        if hasattr(self.expression, '__iter__'):
            for i in self.expression:
                yield i


class Add(Arithmetic):

    def __init__(self, addend_1, addend_2):
        if not isinstance(addend_1, Arithmetic) or not isinstance(addend_2, Arithmetic):
            raise Exception('Add only defined for ``Arithmetic`` objects')
        self.addend_1 = addend_1
        self.addend_2 = addend_2

    def __repr__(self):

        return '({addend_1} + {addend_2})'.format(
            addend_1=str(self.addend_1), addend_2=str(self.addend_2))

    def __iter__(self):
        yield self.addend_1
        if hasattr(self.addend_1, '__iter__'):
            for i in self.addend_1:
                yield i
        yield self.addend_2
        if hasattr(self.addend_2, '__iter__'):
            for i in self.addend_2:
                yield i


class Multiply(Arithmetic):

    def __init__(self, multiplicand_1, multiplicand_2):
        if (not isinstance(multiplicand_1, Arithmetic) or
                not isinstance(multiplicand_2, Arithmetic)):
            raise Exception('Multiply only defined for ``Arithmetic`` objects')
        self.multiplicand_1 = multiplicand_1
        self.multiplicand_2 = multiplicand_2

    def __repr__(self):

        return '({multiplicand_1} * {multiplicand_2})'.format(
            multiplicand_1=str(self.multiplicand_1), multiplicand_2=str(self.multiplicand_2))

    def __iter__(self):
        for multiplicand in [self.multiplicand_1, self.multiplicand_2]:
            yield multiplicand
            if hasattr(multiplicand, '__iter__'):
                for i in multiplicand:
                    yield i


class Probability(Arithmetic):
    """
    ``Probability`` objects have ``Statement`` objects as attributes and are related
    to floats in the range ``[0, 1]``.
    """

    def __init__(self, statement):
        if not isinstance(statement, Statement):
            raise Exception('Probability applies only to ``Statement``s')
        self.statement = statement

    def __repr__(self):
        return 'P({statement})'.format(statement=str(self.statement))

    def __iter__(self):
        yield self.statement
        if hasattr(self.statement, '__iter__'):
            for i in self.statement:
                yield i


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

    def is_literal(self):
        """
        A ``literal`` is an atomic formula or the negation of an atomic formula.
        """

        return self.is_atomic() or (
            isinstance(self, Negation) and self.statement.is_atomic())

    def is_atomic(self):
        """
        Although you can define a new statement using connectives other than
        conjunction and negation, they are immediately transformed into
        conjunctions and negations upon instantiation. Thus, we can test
        whether a statement is atomic by checking whether it is of type
        ``Negation`` or ``Conjunction``.
        """

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

    def __iter__(self):
        for fact in self.facts:
            yield fact


class Equals(object):
    """
    An ``Equals`` is an assertion that an event has a probability of being true.
    """

    def __init__(self, statement, probability):
        self.statement = statement
        self.probability = probability

    def __repr__(self):
        return str(self.statement) + ' = ' + str(self.probability)


class Given(Statement):
    """
    Expressions like ``x|y`` are not events or states; they're used only in
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

    def __iter__(self):
        yield self.event
        if hasattr(self.event, '__iter__'):
            for i in self.event:
                yield i
        
        yield self.given
        if hasattr(self.given, '__iter__'):
            for i in self.given:
                yield i


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

    def __iter__(self):
        yield self.statement
        if hasattr(self.statement, '__iter__'):
            for i in self.statement:
                yield i


class Conjunction(Statement):
    """
    A list of conjuncts.
    """

    def __init__(self, *args):
        """
        The user will likely define conjunctions like ``a & b & c``, which
        would typically yield ``(a & b) & c``, which is correct but
        inconvenient. Better to have ``(a & b & c)`` for easier enumeration
        through the conjuncts. So the ``__init__`` function checks each
        conjunct to see if it's a conjunction, and appends those conjuncts
        to a "flattened" list.
        """

        self.conjuncts = []
        for arg in args:
            if isinstance(arg, Conjunction):
                self.conjuncts += arg.conjuncts
            else:
                self.conjuncts.append(arg)

    def __repr__(self):
        return (
            '(' + ' & '.join(
                [str(conjunct) for conjunct in self.conjuncts]) + ')')

    def __eq__(self, other):
        if not isinstance(other, Conjunction):
            return False
        return self.conjuncts == other.conjuncts

    def __iter__(self):
        for conjunct in self.conjuncts:
            yield conjunct
            if hasattr(conjunct, '__iter__'):
                for i in conjunct:
                    yield i


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


    def _alpha(self, *children):
        """
        Normalization factor for node with children.
        """
        if len(children) == 0:
            children = self.children

        general_case = (
            (Probability(self) * Pi(
                *[Probability(Given(child, self)) for child in children])) +
            (Probability(~self) * Pi(
                *[Probability(Given(child, ~self)) for child in children])))
        
        return general_case

    def _lambda(self, *children):  # I wish lambda weren't a reserved word
        """
        Likelihood of ``self``.
        """

        if len(children) == 0:
            children = self.children

        general_case = Pi(
            *[Probability(Given(child, self)) for child in children])

        return general_case

    def top_down_eval(self):
        # evaluate the value of self, given the parents only
        pass

    def parent_requirements_satisfied(self):
        """
        Return a list of all the ``pi`` requirements for
        ``self`` which are in the fact book.
        """
        
        parent_requirements = self.parent_fact_requirements()
        satisfied_requirements = []
        for fact in self.fact_book:
            statement = fact.statement
            # Only ``Given`` statements are relevant
            if not isinstance(statement, Probability):
                continue
            if not isinstance(statement.statement, Given):
                continue
            given_statement = statement.statement
            if given_statement in parent_requirements:
                satisfied_requirements.append(given_statement)
        return satisfied_requirements

    def child_requirements_satisfied(self):
        # TODO: Rewrite this function so that it returns which
        # facts, if any, are missing. Same with
        # ``parent_fact_requirements``. Also, not very DRY.
        child_requirements = self.child_fact_requirements()
        satisfied_requirements = [] 
        for fact in child_requirements:
            statement = fact.statement
            if not isinstance(statement, Probability):
                continue
            if not instance(statement.statement, Given):
                pass
            given_statement = statement.statement
            if given_statement in child_requirements:
                satisfied_requirements.append(given_statement)
        return satisfied_requirements

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
            for child in node.children:
                recurse(child)
            for parent in node.parents:
                recurse(parent)
        
        recurse(self)
        return node_list

    def associate_fact_book(self, fact_book):
        """
        When we associate a ``FactBook`` with a specific node, then we need
        to propagate it across all the nodes in the graph.
        """

        for node in self.connected_nodes():
            node.fact_book = fact_book
        self.fact_book = fact_book

    def descendants(self):
        """
        Return a list of all the descendants of the node.
        """

        node_list = []
        
        def recurse(node):
            if node in node_list:
                return
            node_list.append(node)
            for child in node.children:
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
                next_steps = current_node.children + current_node.parents
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
            path_triple = (
                node, '->' if next_node in node.children
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

    @property
    def parents(self):
        """
        Return the parent nodes of the current node.
        """

        return [edge.source for edge in self.incoming_edges]

    @property
    def children(self):
        """
        Return the child nodes of the current node.
        """

        return [edge.target for edge in self.outgoing_edges]

    def parent_fact_requirements(self):
        """
        This looks at all parents of ``self`` and returns a list of lists.
        Each sublist is a boolean combination of each of the upstream nodes.
        Each combination (e.g. ``a & b``, ``a & ~b``, ``~a & b``, ``~a & ~b``)
        has to be represented in the ``FactBook`` if we are to accurately
        determine the influence of parent nodes on their child nodes. In
        the Bayes net literature, the messages conveying this information
        from parents import to children is denoted ``pi``, whereas the
        information transmitted from children to parents is denoted ``lambda``.
        """

        incoming_nodes = self.parents
        if len(incoming_nodes) == 0:
            return [self, Negation(self)] 
        event_tuples = event_combinations(*incoming_nodes)
        return [
            Given(self,
                Conjunction(*event_tuple) if len(event_tuple) > 1 else
                event_tuple[0])
            for event_tuple in event_tuples]

    def child_fact_requirements(self):
        """
        Returns list of all facts required for lambda messages.
        """
        outgoing_nodes = self.children
        return (
            [Given(child, self) for child in outgoing_nodes] +
            [Given(child, Negation(self))
             for child in outgoing_nodes])

    def missing_parent_requirements(self):
        requirements = self.parent_fact_requirements()
        missing = [
            requirement for requirement in requirements
            if requirement not in self.parent_requirements_satisfied()]
        return missing
        
    def missing_child_requirements(self):
        requirements = self.child_fact_requirements()
        missing = [
            requirement for requirement in requirements
            if requirement not in self.child_requirements_satisfied()]
        return missing

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

        The concept of d-separation is central to Bayes networks. If ``y``
        d-separates ``x`` from ``z``, then ``x`` and ``z`` are probabilistically
        independent, given ``y``. In other parlance, it's a "screening-off"
        condition. For example, coffee drinkers get lung cancer at a higher rate
        than non-coffee drinkers. But that's because smokers are more likely
        to be coffee drinkers, and smoking causes cancer. So smoking "screens off"
        coffee from cancer. In the language of Bayes nets, smoking d-separates
        coffee and cancer. That is, if you know already whether someone is a
        smoker, then learning about their coffee consumption doesn't give you
        any information about the probability that they will get cancer.
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

    def d_separates_all(self, list_of_nodes):
        """
        Tests whether each pair of nodes in ``list_of_nodes`` is d-separated
        from each other by self. This will be used (e.g.) to determine how to
        evaluate ``Given`` statements where the ``statement`` is a conjunction.
        Specifically, if ``x`` d-separates ``y1``, ``y2``, and ``y3`` then
        ``P(y1, y2, y3 | x) == P(y1 | x) * P(y2 | x) * P(y3 | x)``.
        """
    
        return all(
            self.d_separated(list(node_pair)) for node_pair in
            itertools.combinations(list_of_nodes, 2))

    def audit(self):
        """
        Return a table of facts about the graph, which facts
        are missing, etc.
        """
        audit_dict = {}
        for node in self.connected_nodes():
            info_dict = {}
            info_dict['sink'] = node.is_sink()
            info_dict['source'] = node.is_source()
            info_dict['number_of_parents'] = len(node.parents)
            info_dict['number_of_children'] = len(node.children)
            audit_dict[node] = info_dict
        return audit_dict


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

    fact_book = FactBook()

    fact = Equals(Probability(Given(c, a & b)), .5)
    fact_book += fact
    fact = Equals(Probability(Given(b, a)), .2)
    fact_book += fact
    fact = Equals(Probability(Given(b, ~a)), .3)
    fact_book += fact
    print c.parent_fact_requirements()
    print a.parent_fact_requirements()
    # Next -- test whether the fact requirements are satisfied by a ``Equals``
    # in the ``FactBook`` object.
    print fact_book
    ###

    print fact_book.facts[0].statement == c.parent_fact_requirements()[0]
    for i in a.undirected_paths():
        print i

    b.associate_fact_book(fact_book)
    print b.parent_requirements_satisfied()

    if not b.parent_requirements_satisfied():
        raise Exception()
    
    print a.audit()

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    sandbox()
