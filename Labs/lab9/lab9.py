# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    N = len(training_points)
    dict = {point: make_fraction(1, N) for point in training_points}
    return dict


def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    dict_error_classifier = {}
    for classifier in classifier_to_misclassified.keys():
        points_misclassified = classifier_to_misclassified.get(classifier)
        error_rate_classifier = 0
        for point in points_misclassified:
            weight_misclassification = point_to_weight.get(point)
            error_rate_classifier += weight_misclassification
        dict_error_classifier[classifier] = error_rate_classifier
    return dict_error_classifier


def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    classifier_sorted = sorted(classifier_to_error_rate)
    best_error = None
    best_c = None

    if use_smallest_error == True:
        f = lambda x: x
        g = lambda x: best_error > x
    else:
        f = lambda x: abs(make_fraction(1, 2) - x)
        g = lambda x: best_error < x

    for c in classifier_sorted:
        val = f(make_fraction(classifier_to_error_rate[c]))

        if best_error is None or g(val):
            best_error = val
            best_c = c

    if make_fraction(classifier_to_error_rate[best_c]) == make_fraction(1, 2):
        raise NoGoodClassifiersError("Best classifier gives 1/2")

    return best_c

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 1:
        return -INF
    if error_rate == 0:
        return INF
    ratio = make_fraction(1 - error_rate, error_rate)
    return make_fraction(0.5 * ln(ratio))


def classify_point(p, classifier_to_misclassified, h):
    if p in classifier_to_misclassified[h]:
        return 1
    else:
        return -1


def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    l = []  # misclassified weight 1, otherwise 0
    for p in training_points:
        classification = sum([v * classify_point(p, classifier_to_misclassified, h)
                              for (h, v) in H])
        if classification >= 0:
            l.append(p)
    return set(l)



def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    misclassified_points = len(get_overall_misclassifications(H, training_points, classifier_to_misclassified))
    return misclassified_points <= mistake_tolerance


def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    try:
        up_modif = make_fraction(0.5, 1 - error_rate)
        down_modif = make_fraction(0.5, error_rate)
    except ZeroDivisionError:
        if error_rate == 0:
            up_modif = 0.5
            down_modif = INF
        else:
            up_modif = INF
            down_modif = 0.5
    for point in point_to_weight.keys():
        if point in misclassified_points:
            point_to_weight[point] *= down_modif
        else:
            point_to_weight[point] *= up_modif
    return point_to_weight


#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    H = []
    weights = initialize_weights(training_points)  # dict
    rounds = 0
    while rounds < max_rounds:
        error_rates = calculate_error_rates(weights, classifier_to_misclassified)
        try:
            best_classifier = pick_best_classifier(error_rates, use_smallest_error)
        except NoGoodClassifiersError:
            return H
        error_rate_best_classifier = error_rates.get(best_classifier)
        voting_power_best_classifier = calculate_voting_power(error_rate_best_classifier)
        H.append((best_classifier, voting_power_best_classifier))
        weights = update_weights(weights, classifier_to_misclassified.get(best_classifier), error_rate_best_classifier)
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H
        rounds += 1
    return H

#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
