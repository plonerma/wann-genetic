import numpy as np
import inspect
import logging

import sklearn.metrics

from .expression import softmax

available_measures = dict()
available_prefixes = dict()


def new_measure(func, name=None):
    name = func.__name__ if name is None else name
    dependencies = inspect.signature(func).parameters.keys()
    available_measures[name] = func, set(dependencies)
    return func

def apply_measures(values, names, pending=set()):
    """Update existing dictionary of measurements.

    Calculates value for measure and all values the measures depends on.
    Edits values in place.
    """

    for measure_name in names:
        if measure_name in values:
            continue

        if measure_name not in available_measures:
            raise KeyError(f"Measure {measure_name} was not defined.")

        func, deps = available_measures[measure_name]

        # figure out which values need to be calculated first
        unresolved_deps = {d for d in deps if d not in values}

        if len(unresolved_deps) > 0:

            if not unresolved_deps.isdisjoint(pending):
                intersection = ", ".join(unresolved_deps.intersection(pending))
                raise RuntimeError("Circular dependencies in measures ({intersection} depends on and is required by measure {measure_name}).")

            # calculate unresolved values first
            values = apply_measures(values, unresolved_deps, pending=pending.union({measure_name}))

        if 'weight' in values:
            v = list()

            for i in range(len(values['weight'])):
                # grab all the values the measure depends on from values
                params = {d: values[d][i] for d in deps}

                # calculate the actual measure
                v.append(func(**params))
        else:
            # grab all the values the measure depends on from values
            params = {d: values[d] for d in deps}

            # calculate the actual measurement
            v = func(**params)

        values[measure_name] = v

    return values # not really necessary since dict is edited in place

@new_measure
def y_pred(y_raw):
    return np.argmax(y_raw, axis=-1)

@new_measure
def y_prob(y_raw):
    return softmax(y_raw, axis=-1)

@new_measure
def mean_squared_error(y_true, y_raw):
    return np.mean((y_true - y_raw) ** 2)

@new_measure
def cm(y_true, y_pred):
    try:
        return sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    except Exception as e:
        logging.warning(y_true)
        logging.warning(y_pred)
        raise e

@new_measure
def log_loss(y_true, y_prob):
    # nan is same as maximally falsely predicted
    y_prob[np.isnan(y_prob)] = 0
    return sklearn.metrics.log_loss(y_true, y_prob)

@new_measure
def true_positives(cm):
    # axis 0 : true class axis
    # axis 1 : predicted class axis
    cm = cm / np.sum(cm)
    return np.diagonal(cm, axis1=0, axis2=1)

@new_measure
def accuracy(true_positives):
    return np.sum(true_positives, axis=0)

@new_measure
def true_class_sum(cm):
    return np.sum(cm, axis=1)

@new_measure
def pred_class_sum(cm):
    return np.sum(cm, axis=0)

@new_measure
def false_positives(true_positives, pred_class_sum):
    return pred_class_sum - true_positives

@new_measure
def false_negatives(true_positives, true_class_sum):
    return true_class_sum - true_positives

@new_measure
def accuracy_per_class(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

@new_measure
def recall_per_class(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)

@new_measure
def f1_per_class(precision_per_class, recall_per_class):
    return (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)

@new_measure
def kappa(accuracy, true_class_sum, pred_class_sum):
    """Cohens Kappa"""
    expected_acc = np.sum(true_class_sum * pred_class_sum, axis=0)
    return (accuracy - expected_acc) / (1 - expected_acc)

def num_used_activation_functions(nodes, available_funcs):
    prefix = 'n_nodes_with_act_func_'

    values = dict()
    for func in available_funcs:
        name = prefix + func[0]
        values[name] = 0

    unique, counts = np.unique(nodes['func'], return_counts=True)
    for func_id, num in zip(unique, counts):
        name = prefix + available_funcs[func_id][0]
        values[name] = num

    return values
