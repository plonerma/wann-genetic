from sklearn.metrics import confusion_matrix
import numpy as np

from ..util import serialize_array, deserialize_array


class ClassificationPerformance:
    """Track performances in classification tasks."""

    def __init__(self, arg):
        if not isinstance(arg, np.ndarray):
            # arg is the number of classes
            assert isinstance(arg, int)
            self.confusion_matrix = np.zeros((arg, arg), dtype=int)
        else:
            # arg is the confusion matrix
            # assert that confusion matrix is quadratic
            assert len(arg.shape) == 2
            assert min(arg.shape) == max(arg.shape)

            self.confusion_matrix = arg

        self.n_classes = self.confusion_matrix.shape[0]

    def __iadd__(self, other):
        if isinstance(other, Performance):
            ocm = other.confusion_matrix
            assert self.confusion_matrix.shape == ocm.shape
            self.confusion_matrix += ocm
        else:
            assert isinstance(other, tuple)
            self.add(*other)
        return self

    def add(self, y_true, y_pred):
        """Add new results to the performance record."""
        self.confusion_matrix += confusion_matrix(y_true, y_pred)

    @property
    def total(self):
        return np.sum(self.confusion_matrix)

    @property
    def true_positives(self):
        return np.diag(self.confusion_matrix)

    @property
    def correct(self):
        return np.sum(self.true_positives)

    @property
    def accuracy(self):
        return self.correct / self.total

    @property
    def normal(self):
        return self.confusion_matrix / self.total

    def normalize(self):
        self.confusion_matrix = self.normal

    def normalized(self):
        return Performance(confusion_matrix=self.normal())

    def get_measures(self):
        cm = self.confusion_matrix

        tp = self.true_positives
        total = self.total
        correct = self.correct
        accuracy = self.accuracy

        true_class_sum = np.sum(cm, axis=0)
        pred_class_sum = np.sum(cm, axis=1)

        fp = pred_class_sum - tp # false positives
        fn = true_class_sum - tp # false negatives

        p_estimate = np.sum(true_class_sum * true_class_sum)

        cohen_kappa = accuracy / (1 - p_estimate)

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'precision_per_class': tp / ( tp + fp ),
            'recall_per_class': tp / ( tp + fn ),
            'f1_per_class': (precision*recall) / (precision + recall),
            'cohen_kappa': cohen_kappa,
        }

    def serialize(self):
        return {
            'n_classes': self.n_classes,
            'confusion_matrix': serialize_array(self.confusion_matrix)
        }

    @classmethod
    def deserialize(cls, d):
        n = d['n_classes']
        cm = deserialize_array(d['confusion_matrix'], dtype=int)
        cm.reshape((n, n))
        return cls(n_classes=n, confusion_matrix=cm)


Performance = ClassificationPerformance
