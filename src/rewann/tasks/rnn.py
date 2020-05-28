"""Module containig basic recurrent toy tasks."""

import numpy as np

from .base import RecurrentTask


class EchoTask(RecurrentTask):
    """(see :ref:`echo_task`)"""

    def __init__(self, length, delay=1, dim=2):
        assert delay >= 0  # can't go back in time
        assert dim > 1  # (case 1 is trivial)
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
        x = (np.arange(2) == x_class[..., None]).astype(int)

        y = np.empty(size, dtype=float)

        y[:, :delay] = 0  # there is nothing to echo
        y[:, delay:] = x_class[:, :-delay]  # output is input shifted by delay

        return x, y


class AddingTask(RecurrentTask):
    """As described in "Unitary Evolution Recurrent Neural Networks"

    (see :ref:`adding_task`)
    """

    n_in = 2
    n_out = 1

    def __init__(self, length):
        self.sample_length = length

    def get_data(self, samples, test=False):
        T = self.sample_length
        x = np.empty((samples, T, 2), dtype=float)
        x[:, :, 1] = 0  # marker
        x[:, :, 0] = np.random.uniform(0, 1, size=(samples, T))  # value

        half = T//2
        a = np.random.randint(half, size=samples)
        b = half + np.random.randint(T - half, size=samples)

        sample = np.arange(samples)

        x[sample, a, 1] = 1  # set marker for a
        x[sample, b, 1] = 1  # set marker for b

        values_a = x[sample, a, 0]
        values_b = x[sample, b, 0]

        y = np.full((samples, T), np.nan, dtype=float)
        y[:, -1] = values_a + values_b

        return x, y


class CopyTask(RecurrentTask):
    """As described in "Unitary Evolution Recurrent Neural Networks"

    (see :ref:`copy_task`)
    """
    T = 10  # lenght of the memorization phase
    rep_seq_len = 10  # length of the sequence to be reproduced
    num_categories = 8

    def __init__(self, T=10):
        self.T = T
        self.n_in = self.num_categories + 2
        self.n_out = self.num_categories + 2

    def get_data(self, samples=100, test=False):
        T = self.T
        L = self.rep_seq_len
        K = self.num_categories

        # L-long sequences of uniformly sampled categories
        cat_in = np.random.randint(K, size=(samples, L))

        # same sequences one-hot enccoded
        onehot_sequence = np.zeros((samples, L, K + 2))
        s, i = np.meshgrid(np.arange(samples), np.arange(L))
        onehot_sequence[s.T, i.T, cat_in] = 1

        e = K  # empty symbol
        d = e + 1  # delimiter symbol

        # input data
        x = np.zeros((samples, T+2*L, K + 2))

        # input starts with sequence
        x[:,   :L,      :] = onehot_sequence
        x[:, L :-(L+1), e] = 1  # T-1 empty fields
        x[:, -(L+1),    d] = 1  # delimiter
        x[:, -L:,       e] = 1  # L empty fields

        # expected output data
        y = np.zeros((samples, T+2*L))

        y[:,   :-L] = e  # empty except for last ten elements
        y[:, -L:  ] = cat_in

        return x, y
