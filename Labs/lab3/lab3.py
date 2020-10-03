# MIT 6.034 Lab 3: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp):
    """Returns True if the problem has one or more empty domains, otherwise False"""
    variables = csp.get_all_variables()
    for var in variables:
        if len(csp.get_domain(var)) == 0:
            return True
    return False


def check_all_constraints(csp):
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    constraints = csp.get_all_constraints()
    for c in constraints:
        var1, var2 = c.var1, c.var2
        val1, val2 = csp.get_assignment(var1), csp.get_assignment(var2)
        if val1 is not None and val2 is not None:
            if not c.check(val1, val2):
                return False
    return True


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem):
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = []
    extension_count = 0
    agenda.append(problem)
    while len(agenda) > 0:
        current_problem = agenda.pop()
        extension_count += 1
        if has_empty_domains(current_problem) or check_all_constraints(current_problem) == False:
            continue
        next_neighbor = current_problem.pop_next_unassigned_var()
        if next_neighbor is None:
            return current_problem.assignments, extension_count
        possible_future_values = current_problem.get_domain(next_neighbor)
        next_extensions = []
        for possible_future_value in possible_future_values:
            extension = current_problem.copy().set_assignment(next_neighbor, possible_future_value)
            next_extensions.append(extension)
        for next in next_extensions[::-1]:
            agenda.append(next)
    return None, extension_count


# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.
pokemon = get_pokemon_problem()
_, count_dfs = solve_constraint_dfs((pokemon))

ANSWER_1 = count_dfs


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var):
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    current_val = csp.get_domain(var)
    list_modification = []
    for neighbor in csp.get_neighbors(var):  # iterate over the neighbors of var
        values_neighbor = [value for value in csp.get_domain(neighbor)]  # domain of the neighbor
        constraints = csp.constraints_between(neighbor, var)  # get the constraints between a neighbor and var
        for neighbor_assignment in values_neighbor:  # for an assignment for neighbor
            check = False
            for assignment in current_val:  # check if there is at least one assignment for var that is compatible
                count = 1
                if len(constraints) > 1:
                    for constraint in constraints:
                        count = count * int(constraint.check(neighbor_assignment, assignment))
                        check = bool(count)
                else:
                    for constraint in constraints:
                        if constraint.check(neighbor_assignment, assignment):
                            check = True
            if not check:  # no assignment for var compatible with neighbor_assignment
                csp.eliminate(neighbor, neighbor_assignment)
                if len(csp.get_domain(neighbor)) == 0:
                    return None
                if neighbor not in list_modification:
                    list_modification.append(neighbor)
    return sorted(list_modification) if len(list_modification) > 0 else list_modification


# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors


def solve_constraint_forward_checking(problem):
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    agenda = []
    extension_count = 0
    agenda.append(problem)
    while len(agenda) > 0:
        current_problem = agenda.pop()
        extension_count += 1
        if has_empty_domains(current_problem) or check_all_constraints(current_problem) == False:
            continue
        next_neighbor = current_problem.pop_next_unassigned_var()
        if next_neighbor is None:
            return current_problem.assignments, extension_count
        possible_future_values = current_problem.get_domain(next_neighbor)
        next_extensions = []
        for possible_future_value in possible_future_values:
            extension = current_problem.copy().set_assignment(next_neighbor, possible_future_value)
            eliminate_from_neighbors(extension, next_neighbor)
            next_extensions.append(extension)
        for next in next_extensions[::-1]:
            agenda.append(next)
    return None, extension_count


# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?
pokemon = get_pokemon_problem()
_, count_fc = solve_constraint_forward_checking(pokemon)

ANSWER_2 = count_fc


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None):
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    dequeued = []
    if queue is None:
        queue = csp.get_all_variables()
    while queue:
        var = queue.pop(0)
        dequeued.append(var)
        modifications = eliminate_from_neighbors(csp, var)
        if modifications is None:
            return None
        for var_modified in modifications:
            if var_modified not in queue:
                queue.append(var_modified)
    return dequeued





# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

pokemon = get_pokemon_problem()
domain_reduction(pokemon)
_, count_dfs_domain_reduc = solve_constraint_dfs(pokemon)

ANSWER_3 = count_dfs_domain_reduc



def solve_constraint_propagate_reduced_domains(problem):
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = []
    extension_count = 0
    agenda.append(problem)
    while len(agenda) > 0:
        current_problem = agenda.pop()
        extension_count += 1
        if has_empty_domains(current_problem) or check_all_constraints(current_problem) == False:
            continue
        next_neighbor = current_problem.pop_next_unassigned_var()
        if next_neighbor is None:
            return current_problem.assignments, extension_count
        possible_future_values = current_problem.get_domain(next_neighbor)
        next_extensions = []
        for possible_future_value in possible_future_values:
            extension = current_problem.copy().set_assignment(next_neighbor, possible_future_value)
            domain_reduction(extension, [next_neighbor])
            eliminate_from_neighbors(extension, next_neighbor)
            next_extensions.append(extension)
        for next in next_extensions[::-1]:
            agenda.append(next)
    return None, extension_count


# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?
pokemon = get_pokemon_problem()
_, count_fc_rd = solve_constraint_propagate_reduced_domains(pokemon)

ANSWER_4 = count_fc_rd


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None):
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    dequeued = []
    if queue is None:
        queue = csp.get_all_variables()
    while queue:
        var = queue.pop(0)
        dequeued.append(var)
        modifications = eliminate_from_neighbors(csp, var)
        if modifications is None:
            return None
        for var_modified in modifications:
            if var_modified not in queue:
                if enqueue_condition_fn(csp, var_modified):
                    queue.append(var_modified)
    return dequeued



def condition_domain_reduction(csp, var):
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True


def condition_singleton(csp, var):
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    modifications = eliminate_from_neighbors(csp, var)
    if modifications is None:
        return False
    if len(csp.get_domain(var)) == 1:
        return True
    return False


def condition_forward_checking(csp, var):
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None):
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    agenda = []
    extension_count = 0
    agenda.append(problem)
    while len(agenda) > 0:
        current_problem = agenda.pop()
        extension_count += 1
        if has_empty_domains(current_problem) or check_all_constraints(current_problem) == False:
            continue
        next_neighbor = current_problem.pop_next_unassigned_var()
        if next_neighbor is None:
            return current_problem.assignments, extension_count
        possible_future_values = current_problem.get_domain(next_neighbor)
        next_extensions = []
        for possible_future_value in possible_future_values:
            extension = current_problem.copy().set_assignment(next_neighbor, possible_future_value)
            if enqueue_condition is not None:
                propagate(enqueue_condition, extension, [next_neighbor])
            next_extensions.append(extension)
        for next in next_extensions[::-1]:
            agenda.append(next)
    return None, extension_count


# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)
pokemon = get_pokemon_problem()
_, count_fc_prop1 = solve_constraint_generic(pokemon, condition_singleton)
ANSWER_5 = count_fc_prop1


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n):
    return abs(m-n) == 1



def constraint_not_adjacent(m, n):
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return not constraint_adjacent(m, n)


def all_different(variables):
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    constraints = []
    for i, var in enumerate(variables):
        for var_pair in variables[i+1:]:
            constraints.append(Constraint(var, var_pair, constraint_different))
    return constraints

#### SURVEY ####################################################################

NAME = 'Assaraf David'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = 'generic propagate + visualize the performances of every improvement'
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
