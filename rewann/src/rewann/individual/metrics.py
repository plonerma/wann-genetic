from sklearn.metrics import confusion_matrix
import numpy as np
import inspect

import logging

from ..util import serialize_array, deserialize_array

# TODO: Maybe built a singleton class out of this?

available_metrics = dict()

def new_metric(func, name=None):
    name = func.__name__ if name is None else name
    dependencies = inspect.signature(func).parameters.keys()
    available_metrics[name] = func, set(dependencies)
    return name

def apply_metrics(values, names, pending=set()):
    """ Updates existing dictionary of metric values by calculating metrics.

    Calculates value for metric and all values the metrics depends on.
    Edits values in place.
    """

    for metric_name in names:
        if metric_name in values:
            continue

        if metric_name not in available_metrics:
            raise RuntimeError(f"Metric {metric_name} was not defined.")

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
        values[metric_name] = func(**params)

    return values # not really necessary since dict is edited in place


@new_metric
def total(cm_stack):
    return np.ones(cm_stack.shape[0])  # cms are normalized

@new_metric
def true_positives(cm_stack):
    # axis 0 = stack axis
    # axis 1 = true class axis
    # axis 2 = predicted class axis
    return np.diagonal(cm_stack, axis1=1, axis2=2)

@new_metric
def accuracy(true_positives):
    return np.sum(true_positives, axis=1)

@new_metric
def true_class_sum(cm_stack):
    return np.sum(cm_stack, axis=2)

@new_metric
def pred_class_sum(cm_stack):
    return np.sum(cm_stack, axis=1)

@new_metric
def false_positives(true_positives, pred_class_sum):
    return pred_class_sum - true_positives

@new_metric
def false_negatives(true_positives, true_class_sum):
    return true_class_sum - true_positives

@new_metric
def precision_per_class(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

@new_metric
def recall_per_class(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)

@new_metric
def f1_per_class(precision_per_class, recall_per_class):
    return (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)

@new_metric
def cohen_kappa(accuracy, true_class_sum, pred_class_sum, total):
    logging.debug(f"True class sum:\n{true_class_sum}")
    logging.debug(f"Predicted class sum:\n{pred_class_sum}")
    logging.debug(f"Total class sum:\n{total}")
    expected_acc = true_class_sum * pred_class_sum / total**2
    logging.debug(f"Expected accuracy:\n{expected_acc}")
    logging.debug(f"Accuracy:\n{accuracy}")
    return (accuracy - expected_acc) / (1 - expected_acc)

@new_metric
def avg_cohen_kappa(cohen_kappa):
    return np.average(cohen_kappa)

@new_metric
def avg_accuracy(accuracy):
    return np.average(accuracy)


class ClassificationRecord:
    """ Track performances in classification tasks. """

    def __init__(self, n_classes : int, env):
        self.env = env
        self.n_classes = n_classes

        # stack for storing confusion matrices
        self.cm_stack = np.empty((0, n_classes, n_classes))

        # weights used during evaluations
        self.used_weights = np.array([])

        self.metric_values = dict(cm_stack=self.cm_stack)

        #self.y_preds = np.array([[]])

    def stack_predictions(self, y_true, y_pred, w):
        cm = confusion_matrix(y_true, y_pred, normalize='all')
        self.cm_stack = np.vstack([self.cm_stack, [cm]])
        self.used_weights = np.hstack([self.used_weights, w])

        # reset all metric values
        self.metric_values = dict(cm_stack=self.cm_stack)

    def get_metrics(self, *metrics, as_dict=False):
        """ Metrics provided as strings are calculated in order. """

        values = apply_metrics(self.metric_values, metrics)

        if as_dict:
            return {
                m: values[m] for m in metrics
            }
        elif len(metrics) == 1:
            return values[metrics[0]]
        else:
            return tuple([values[m] for m in metrics])

    def serialize(self):
        return {
            'size': self.cm_stack.shape[:2],
            'stack': serialize_array(self.cm_stack)
        }

    @classmethod
    def deserialize(cls, d):
        stack_size, n_classes = d['size']
        cm = deserialize_array(d['stack'], dtype=int)
        cm.reshape((stack_size, n_classes, n_classes))
        return cls(n_classes=n_classes, confusion_matrix=cm)
