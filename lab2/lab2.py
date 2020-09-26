# MIT 6.034 Lab 2: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')


# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    for chains in board.get_all_chains():
        if len(chains) >= 4:
            return True
    for column in range(7):
        if not board.is_column_full(column):
            return False
    return True


def next_boards_connectfour(board):
    if is_game_over_connectfour(board):
        return []
    c = [0, 1, 2, 3, 4, 5, 6]
    return [board.add_piece(column) for column in c if not board.is_column_full(column)]


def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if is_game_over_connectfour(board):
        for chains in board.get_all_chains():
            if len(chains) >= 4:
                if is_current_player_maximizer:
                    return -1000
                else:
                    return 1000
        return 0


def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    if is_game_over_connectfour(board):
        for chains in board.get_all_chains():
            if len(chains) >= 4:
                score = 1000 + (42 - board.count_pieces()) * 300
                if is_current_player_maximizer:
                    return -score
                else:
                    return score
        return 0

import numpy as np

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    thresholds = [5, 15, -5, -15]
    heuristic_diff = {'5': [10], '15': [500], '-5': [-10], '-15': [-500]}
    chains_current_max = sum([len(chain) if len(chain) < 3 else len(chain)*3 for chain in board.get_all_chains(True)])
    chains_adversary_max = sum([len(chain) if len(chain) < 3 else len(chain)*3 for chain in board.get_all_chains(False)])
    diff_length = chains_current_max - chains_adversary_max
    if diff_length == 0:
        return 0
    diff_length_t = thresholds[np.argmin(np.abs(np.asarray(thresholds)-diff_length))]
    heuristic_score = heuristic_diff.get(str(diff_length_t))[0]
    if is_current_player_maximizer:
        return heuristic_score
    return -heuristic_score



# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot=ConnectFourBoard(),
                                               is_game_over_fn=is_game_over_connectfour,
                                               generate_next_states_fn=next_boards_connectfour,
                                               endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot=NEARLY_OVER,
                                      is_game_over_fn=is_game_over_connectfour,
                                      generate_next_states_fn=next_boards_connectfour,
                                      endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot=BOARD_UHOH,
                               is_game_over_fn=is_game_over_connectfour,
                               generate_next_states_fn=next_boards_connectfour,
                               endgame_score_fn=endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state):
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    temporary_stack = []  # will behave like a queue containint lists (=paths), with the head pointer at the end of the list
    temporary_stack.append([state])
    running_max = 0
    running_path = []
    count = 0
    while len(temporary_stack) > 0:
        state_to_be_extended = temporary_stack.pop()
        current_state = state_to_be_extended[-1]
        if current_state.is_game_over():
            endgame_score = abs(current_state.get_endgame_score())
            count = count + 1
            if endgame_score > running_max:
                running_max = endgame_score
                running_path = state_to_be_extended
        extensions_state = current_state.generate_next_states()
        if extensions_state is not []:
            for state_extended in extensions_state:
                current_path = (state_to_be_extended.copy())
                current_path.append(state_extended)
                temporary_stack.append(current_path)
    return running_path, running_max, count

# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))


counter = 0
def minimax_endgame_search(state, maximize=True):
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    count = 0
    path_list = []
    minimax_score = None
    extension_states = state.generate_next_states()
    if state.is_game_over() or extension_states is []:
        return [state], state.get_endgame_score(maximize), 1
    if maximize:
        for extension_state in extension_states:
            result = minimax_endgame_search(extension_state, False)
            count = count + result[2]
            if minimax_score == None or result[1] > minimax_score:
                path_list = [state] + result[0]
                minimax_score = result[1]
    else:
        for extension_state in extension_states:
            result = minimax_endgame_search(extension_state, True)
            count = count + result[2]
            if minimax_score == None or result[1] < minimax_score:
                path_list = [state] + result[0]
                minimax_score = result[1]

    return path_list, minimax_score, count
# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:
#counter = 0
#pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True):
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    count = 0
    path_list = []
    minimax_score = None
    extension_states = state.generate_next_states()
    if state.is_game_over() or extension_states is []:
        return [state], state.get_endgame_score(maximize), 1
    if depth_limit == 0:
        return [state], heuristic_fn(state.get_snapshot(), maximize), 1
    if maximize:
        for extension_state in extension_states:
            result = minimax_search(extension_state, heuristic_fn, depth_limit=depth_limit-1, maximize=False)
            count = count + result[2]
            if minimax_score == None or result[1] > minimax_score:
                path_list = [state] + result[0]
                minimax_score = result[1]
    else:
        for extension_state in extension_states:
            result = minimax_search(extension_state, heuristic_fn, depth_limit=depth_limit-1, maximize=True)
            count = count + result[2]
            if minimax_score == None or result[1] < minimax_score:
                path_list = [state] + result[0]
                minimax_score = result[1]

    return path_list, minimax_score, count


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True):
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    count = 0
    path_list = []
    minimax_score = None
    extension_states = state.generate_next_states()
    if state.is_game_over() or extension_states is []:
        return [state], state.get_endgame_score(maximize), 1
    if depth_limit == 0:
        return [state], heuristic_fn(state.get_snapshot(), maximize), 1
    if maximize:
        for extension_state in extension_states:
            result = minimax_search_alphabeta(extension_state, alpha=alpha, beta=beta, heuristic_fn=heuristic_fn, depth_limit=depth_limit - 1, maximize=False)
            count = count + result[2]
            if minimax_score is None or result[1] > minimax_score:
                alpha = max(alpha, result[1])
                path_list = [state] + result[0]
                minimax_score = result[1]
                if alpha >= beta:
                    return path_list, alpha, count
    else:
        for extension_state in extension_states:
            result = minimax_search_alphabeta(extension_state, alpha=alpha, beta=beta, heuristic_fn=heuristic_fn, depth_limit=depth_limit - 1, maximize=True)
            count = count + result[2]
            if minimax_score is None or result[1] < minimax_score:
                beta = min(beta, result[1])
                path_list = [state] + result[0]
                minimax_score = result[1]
                if alpha >= beta:
                    return path_list, beta, count

    return path_list, minimax_score, count


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True):
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()
    for d in range(1, depth_limit+1):
        result = minimax_search_alphabeta(state, -INF, INF, heuristic_fn, d, maximize)
        anytime_value.set_value(result)
    return anytime_value

# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError


    progressive_deepening = not_implemented

#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'

#### SURVEY ###################################################

NAME = 'David Assaraf'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = 'Everything'
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
