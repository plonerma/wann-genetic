import numpy as np

from .base import RecurrentTask

class EchoTask(RecurrentTask):
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

        y = np.empty(size, dtype=float)

        y[:, :delay] = np.nan  # there is nothing to echo
        y[:, delay:] = x_class[:, :-delay]  # output is input shifted by delay

        return x, y


class AddingTask(RecurrentTask):
    """ As described in Unitary Evolution Recurrent Neural Networks """

    n_in = 2
    n_out = 1

    def __init__(self, length):
        self.sample_length = length

    def get_data(self, samples, test=False):
        T = self.sample_length
        x = np.empty((samples, T, 2), dtype=float)
        x[:, :, 1] = 0 # marker
        x[:, :, 0] = np.random.uniform(0, 1, size=(samples, T)) # value

        half = T//2
        a = np.random.randint(half, size=samples)
        b = half + np.random.randint(T - half, size=samples)

        sample = np.arange(samples)

        x[sample, a, 1] = 1 # set marker for a
        x[sample, b, 1] = 1 # set marker for b

        values_a = x[sample, a, 0]
        values_b = x[sample, b, 0]

        y = np.full((samples, T), np.nan, dtype=float)
        y[:, -1] = values_a + values_b

        return x, y
