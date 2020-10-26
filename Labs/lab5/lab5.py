# MIT 6.034 Lab 5: Bayesian Inference
# Written by 6.034 staff

from nets import *
from bayes_api import *
import numpy as np

#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    parents_visited = []
    resulting_set = set()
    direct_parents = net.get_parents(var)
    parents_list = [parent for parent in direct_parents]
    while parents_list:
        parent = parents_list.pop()
        resulting_set.add(parent)
        if parent not in parents_visited:
            parents_visited.append(parent)
            parents_list.extend(net.get_parents(parent))
    return resulting_set



def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    children_visited = []
    resulting_set = set()
    direct_children = net.get_children(var)
    children_list = [child for child in direct_children]
    while children_list:
        child = children_list.pop()
        resulting_set.add(child)
        if child not in children_visited:
            children_visited.append(child)
            children_list.extend(net.get_children(child))
    return resulting_set

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    variables = net.get_variables()
    descendants_var = get_descendants(net, var)
    for descendant in descendants_var:
        variables.remove(descendant)
    variables.remove(var)
    return set(variables)



#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    parent = net.get_parents(var)
    descendants = get_descendants(net, var)
    if parent.issubset(givens) and not descendants.intersection(givens):
        return {p: givens.get(p) for p in parent}
    return givens

    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    if givens is None:
        try:
            return net.get_probability(hypothesis, None, infer_missing=True)
        except ValueError:
            raise LookupError
    try:
        simplified_givens = simplify_givens(net, list(hypothesis)[0], givens)
        probability = net.get_probability(hypothesis, simplified_givens, infer_missing=True)
        return probability
    except ValueError:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    ordered_net = net.topological_sort()[::-1]  # children come before their parents
    p = 1
    for v in ordered_net:
        if v in hypothesis:
            val = hypothesis.pop(v)
            if hypothesis is not None:
                prob = probability_lookup(net, {v:val}, givens=hypothesis)
            else:
                prob = probability_lookup(net, {v:val}, givens=None)
            p = p*prob
    return p



    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    joints = net.combinations(net.get_variables(), hypothesis)
    joint_probabilities = [probability_joint(net, joint) for joint in joints]
    return np.sum(joint_probabilities)


def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if givens is not None:
        for k in hypothesis.keys():
            if k in givens.keys():
                if hypothesis.get(k) != givens.get(k):
                    return 0
        joint_hypothesis = dict(hypothesis, **givens)
        numerator = probability_marginal(net, joint_hypothesis)
        denominator = probability_marginal(net, givens)
        return numerator/denominator
    else:
        return probability_marginal(net, hypothesis)
    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net, hypothesis, givens)



#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    params = 0
    all_variables = net.get_variables()
    for variable in all_variables:
        domain_variable = len(net.get_domain(variable)) - 1
        parents = net.get_parents(variable)
        if len(parents) > 0:
            p = 1
            for parent in parents:
                p *= len(net.get_domain(parent))
            params += p*domain_variable
        else:
            params += domain_variable
    return params


#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    hypotheses = net.combinations([var1, var2])
    for hypothese in hypotheses:
        if givens is None:
            marginal_probability = probability(net, {var1: hypothese.get(var1)}, None)
            conditional_probability = probability(net, {var1: hypothese.get(var1)}, {var2: hypothese.get(var2)})
        else:
            marginal_probability = probability(net, {var1: hypothese.get(var1)}, givens)
            conditional_probability = probability(net, {var1: hypothese.get(var1)}, dict(givens, **{var2: hypothese.get(var2)}))
        if not (approx_equal(marginal_probability, conditional_probability)):
            return False
    return True

    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    # construct the ancestor graph
    ancestor_variables = set()
    # how to deal with givens = None ?
    if givens is not None:
        for var in [var1, var2, *givens.keys()]:
            ancestor_variables.update(get_ancestors(net, var))
        ancestor_variables.update({var1, var2, *givens.keys()})
    if givens is None:
        for var in [var1, var2]:
            ancestor_variables.update(get_ancestors(net, var))
        ancestor_variables.update({var1, var2})
    ancestor_net = net.subnet(list(ancestor_variables))
    # link parents of a same child in the ancestor graph
    variables = ancestor_net.get_variables()
    for var in variables:
        parents = ancestor_net.get_parents(var)
        for i, parent1 in enumerate(list(parents)):
            for parent2 in list(parents)[i+1:]:
                if parent1 != parent2:
                    ancestor_net.link(parent1, parent2)
    # disorient the graph
    ancestor_net = ancestor_net.make_bidirectional()
    # delete the givens and their edges
    if givens is not None:
        for given in givens:
            if given in ancestor_net.get_variables():
                ancestor_net.remove_variable(given)
    # check for connection inside the remaining graph
    return ancestor_net.find_path(var1, var2) is None






#### SURVEY ####################################################################

NAME = 'Assaraf David'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = 7
WHAT_I_FOUND_INTERESTING = 'Everything'
WHAT_I_FOUND_BORING = 'None'
SUGGESTIONS = 'None'
