# MIT 6.034 Lab 6: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math

from api import NoGoodClassifiersError

log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    stack_descendents = [id_tree]
    while stack_descendents:
        tree = stack_descendents.pop()
        if tree.is_leaf():
            return tree.get_node_classification()
        stack_descendents.append(tree.apply_classifier(point))


#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    dict_result = {}
    for point in data:
        classification = classifier.classify(point)
        if dict_result.get(classification) is not None:
            dict_result.get(classification).append(point)
        else:
            dict_result[classification] = [point]
    return dict_result


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    dict_split = split_on_classifier(data, target_classifier)
    s = 0
    n_b = len(data)
    for category in dict_split.keys():
        n_bc = len(dict_split.get(category))
        s += -n_bc * log2(n_bc / n_b)
    return s / n_b


def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    N = len(data)
    dict_leaves = split_on_classifier(data, test_classifier)
    resulting_disorder = 0
    for branch in dict_leaves.keys():
        weight = len(dict_leaves.get(branch)) / N
        disorder_branch = branch_disorder(dict_leaves.get(branch), target_classifier)
        resulting_disorder += weight * disorder_branch
    return resulting_disorder


## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab6.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    m = INF
    try:
        for classifier in possible_classifiers:
            classifier_disorder = average_test_disorder(data, classifier, target_classifier)
            if classifier_disorder < m:
                m = classifier_disorder
                selected_classifier = classifier
        if len(split_on_classifier(data, selected_classifier)) == 1:
            raise NoGoodClassifiersError
        return selected_classifier
    except NoGoodClassifiersError:
        raise NoGoodClassifiersError


## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node is None:
        id_tree_node = IdentificationTreeNode(target_classifier)
    if branch_disorder(data, target_classifier) == 0.0:  # homogeneous data
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
    else:
        try:
            classifier_selected = find_best_classifier(data, possible_classifiers, target_classifier)
        except NoGoodClassifiersError:
            return id_tree_node
        separation = split_on_classifier(data, classifier_selected)
        id_tree_node.set_classifier_and_expand(classifier_selected, separation.keys())
        possible_classifiers.remove(classifier_selected)
        branches = id_tree_node.get_branches()
        for branch in branches.keys():
            branches[branch] = construct_greedy_id_tree(separation.get(branch), possible_classifiers, target_classifier,
                                                        branches.get(branch))
    return id_tree_node


## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = 'bark_texture'
ANSWER_2 = 'leaf_shape'
ANSWER_3 = 'orange_foliage'

ANSWER_4 = [2, 3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'

#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *

    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)

################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    return sum([u * v for (u, v) in zip(u, v)])


def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return math.sqrt(dot_product(v, v))


def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    return norm([u - v for (u, v) in zip(point1, point2)])


def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    return sum([abs(u - v) for (u, v) in zip(point1, point2)])


def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    return sum([int(u != v) for (u, v) in zip(point1, point2)])


def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    return 1 - dot_product(point1, point2) / (norm(point1) * norm(point2))


#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    sorted_coordinates = sorted(data, key=lambda d: d.coords)
    sorted_distance_coordinates = sorted(sorted_coordinates, key=lambda d: distance_metric(d, point))
    return sorted_distance_coordinates[:k]


def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    neighbors = get_k_closest_points(point, data, k, distance_metric)
    classes = [neighbor.classification for neighbor in neighbors]
    unique_classes = set(classes)
    counts = {c: classes.count(c) for c in unique_classes}
    counts_sorted = {k: v for (k, v) in sorted(counts.items(), key=lambda d: d[1])}
    return list(counts_sorted.keys())[-1]


## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    right_classifications = 0
    for point in data:
        datas = data.copy()
        datas.remove(point)
        classification = knn_classify_point(point, datas, k, distance_metric)
        right_classifications += 1*(classification == point.classification)
    return right_classifications/len(data)



def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    scoring_metrics = [euclidean_distance, hamming_distance, manhattan_distance, cosine_distance]
    ks = [1, 2, 3, 4, 5, 6]
    best_score = 0
    best_parameters = [None, None]
    for score in scoring_metrics:
        for k in ks:
            if cross_validate(data, k, score) > best_score:
                best_parameters[0] = k
                best_parameters[1] = score
    output = best_parameters[0], best_parameters[1]
    return output



## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = 'Overfitting'
kNN_ANSWER_2 = 'Underfitting'
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3

#### SURVEY ####################################################################

NAME = 'David Assaraf'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = 'Everything'
WHAT_I_FOUND_BORING = 'Nothing'
SUGGESTIONS = ''
