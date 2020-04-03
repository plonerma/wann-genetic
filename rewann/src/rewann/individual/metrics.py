import numpy as np
import inspect

available_metrics = dict()
available_prefixes = dict()


def prediction_metric(func, name=None):
    name = func.__name__ if name is None else name
    dependencies = inspect.signature(func).parameters.keys()
    available_metrics[name] = func, set(dependencies)
    return func

def prefix_metric(func, prefix=None):
    prefix = func.__name__ if prefix is None else prefix
    available_prefixes[prefix] = func
    return func

def apply_metrics(values, names, pending=set()):
    """ Updates existing dictionary of metric values by calculating metrics.

    Calculates value for metric and all values the metrics depends on.
    Edits values in place.
    """

    for metric_name in names:
        if metric_name in values:
            continue

        *prefixes, name = metric_name.split(':')

        for p in prefixes:
            if p not in available_prefixes:
                raise RuntimeError(f"Metric prefix {p} was not defined.")

        if name not in available_metrics:
            raise RuntimeError(f"Metric {name} was not defined.")

        func, deps = available_metrics[name]

        # figure out which values need to be calculated first
        unresolved_deps = {d for d in deps if d not in values}

        if len(unresolved_deps) > 0:

            if not unresolved_deps.isdisjoint(pending):
                intersection = ", ".join(unresolved_deps.intersection(pending))
                raise RuntimeError("Circular dependencies in metrics ({intersection} depends on and is required by metric {name}).")

            # calculate unresolved values first
            values = apply_metrics(values, unresolved_deps, pending=pending.union({name}))

        # grab all the values the metric depends on from values
        params = {d: values[d] for d in deps}

        # calculate the actual metric
        value = func(**params)

        # calculate prefix funcs
        for p in reversed(prefixes):
            value = available_prefixes[p](value)
        values[metric_name] = value

    return values # not really necessary since dict is edited in place


@prediction_metric
def true_positives(cm_stack):
    # axis 0 = stack axis
    # axis 1 = true class axis
    # axis 2 = predicted class axis
    return np.diagonal(cm_stack, axis1=1, axis2=2)

@prediction_metric
def accuracy(true_positives):
    return np.sum(true_positives, axis=1)

@prediction_metric
def true_class_sum(cm_stack):
    return np.sum(cm_stack, axis=2)

@prediction_metric
def pred_class_sum(cm_stack):
    return np.sum(cm_stack, axis=1)

@prediction_metric
def false_positives(true_positives, pred_class_sum):
    return pred_class_sum - true_positives

@prediction_metric
def false_negatives(true_positives, true_class_sum):
    return true_class_sum - true_positives

@prediction_metric
def precision_per_class(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

@prediction_metric
def recall_per_class(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)

@prediction_metric
def f1_per_class(precision_per_class, recall_per_class):
    return (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)

@prediction_metric
def kappa(accuracy, true_class_sum, pred_class_sum):
    """Cohens Kappa"""
    expected_acc = np.sum(true_class_sum * pred_class_sum, axis=1)
    return (accuracy - expected_acc) / (1 - expected_acc)

@prefix_metric
def median(x):
    return np.median(x)

@prefix_metric
def mean(x):
    return np.mean(x)

@prefix_metric
def min(x):
    return np.min(x)

@prefix_metric
def max(x):
    return np.max(x)
