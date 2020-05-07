import numpy as np
import inspect
import warnings

import sklearn.metrics

available_metrics = dict()
available_prefixes = dict()


def prediction_metric(func, name=None):
    name = func.__name__ if name is None else name
    dependencies = inspect.signature(func).parameters.keys()
    available_metrics[name] = func, set(dependencies)
    return func

def apply_metrics(values, names, pending=set()):
    """ Updates existing dictionary of metric values by calculating metrics.

    Calculates value for metric and all values the metrics depends on.
    Edits values in place.
    """

    for metric_name in names:
        if metric_name in values:
            continue

        if metric_name not in available_metrics:
            warnings.warn(f"Metric {metric_name} was not defined.")
            continue

        func, deps = available_metrics[metric_name]

        # figure out which values need to be calculated first
        unresolved_deps = {d for d in deps if d not in values}

        if len(unresolved_deps) > 0:

            if not unresolved_deps.isdisjoint(pending):
                intersection = ", ".join(unresolved_deps.intersection(pending))
                raise RuntimeError("Circular dependencies in metrics ({intersection} depends on and is required by metric {metric_name}).")

            # calculate unresolved values first
            values = apply_metrics(values, unresolved_deps, pending=pending.union({metric_name}))

        # grab all the values the metric depends on from values
        params = {d: values[d] for d in deps}

        # calculate the actual metric
        value = func(**params)

        values[metric_name] = value

    return values # not really necessary since dict is edited in place


@prediction_metric
def cm(y_true, y_pred, labels):
    valid = y_true >= 0  # only use predictions, where labels are set
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    return sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize='all')

@prediction_metric
def log_loss(y_true, y_prob, labels):
    return sklearn.metrics.log_loss(y_true, y_prob, labels=labels)

@prediction_metric
def true_positives(cm):
    # axis 0 : true class axis
    # axis 1 : predicted class axis
    return np.diagonal(cm, axis1=0, axis2=1)

@prediction_metric
def accuracy(true_positives):
    return np.sum(true_positives, axis=0)

@prediction_metric
def true_class_sum(cm):
    return np.sum(cm, axis=1)

@prediction_metric
def pred_class_sum(cm):
    return np.sum(cm, axis=0)

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
    expected_acc = np.sum(true_class_sum * pred_class_sum, axis=0)
    return (accuracy - expected_acc) / (1 - expected_acc)
