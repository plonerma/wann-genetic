import logging
import numpy as np


class Task:
    is_recurrent = False

    def load_training(self):
        self.load()

    def load_test(self):
        self.load(test=True)

    def load(self, test=False):
        pass

    def get_data(self):
        raise NotImplementedError()


class RecurrentTask(Task):
    is_recurrent = True


class ClassificationTask(Task):
    x, y, test_x, test_y, _y_labels = None, None, None, None, None

    def __init__(self, n_in, n_out, load_func):
        self.n_in = n_in
        self.n_out = n_out

        self.load_func = load_func

    @property
    def y_labels(self):
        if self._y_labels is None:
            return list(range(self.n_out))
        else:
            return self._y_labels

    def load(self, test=False):
        if (not test and self.x is None) or (test and self.test_x is None):
            d = self.load_func(test=test)

            if len(d) == 2:
                (x, y), y_labels = d, None
            else:
                x, y, y_labels = d

            logging.debug(f"Loaded {len(y)} samples.")

            if test:
                self.test_x, self.test_y = x, y
                self._y_lables = y_labels
            else:
                self.x, self.y = x, y
                self._y_lables = y_labels

    def get_data(self, samples, test=False):
        if test:
            if self.test_x is None:
                logging.warning("Evaluation on test data requires loading test data.")
            x = self.test_x
            y = self.test_y

        else:
            if self.x is None:
                logging.warning("Evaluation on training data requires loading training data.")

            x = self.x
            y = self.y

        if samples > 0 and samples < len(x):
            chosen = np.random.choice(len(x), size=samples, replace=False)
            x = x[chosen]
            y = y[chosen]

        return x, y
