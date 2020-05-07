import numpy as np

from .base import Task

class EchoTask(Task):
    def __init__(self, length, delay=1, dim=2):
        assert delay >= 0 # can't go back in time
        assert dim > 1 # (case 1 is trivial)
        self.sample_length = length
        self.n_in = self.n_out = dim
        self.delay = delay

    def get_data(self, samples, test=False):
        assert samples > 0

        delay = self.delay

        # NOTE: This implementation assumes that all sequences have the same
        # length.
        # If non-constant sequence should be required, consider filling the
        # last inputs with buffer values
        # This allows matrix computation even on sequence data
        size = (samples, self.sample_length)

        x_class = np.random.randint(self.n_in, size=size)

        # one hot encoding
        # source: https://stackoverflow.com/a/36960495
        x = (np.arange(2) == x_class[...,None]).astype(int)

        y = np.empty(size, dtype=int)

        y[:, :delay] = -1  # there is nothing to echo
        y[:, delay:] = x_class[:, :-delay]  # output is input shifted by delay

        return x, y
