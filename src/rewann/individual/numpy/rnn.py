import numpy as np

import logging

from rewann.individual.network_base import BaseRNN

from .ffnn import Network as FFNN


class Network(FFNN, BaseRNN):
    """Numpy implementation of Recurrrent Neural Network"""

    def get_measurements(self, weights, x, y_true=None, measures=['predictions']):
        assert len(x.shape) == 3
        num_samples, sample_length, dim = x.shape

        assert dim == self.n_in

        num_weights, = weights.shape  # weight array should be one-dimensional

        # outputs in each sequence step is stored
        y_raw = np.empty((num_weights, num_samples, sample_length, self.n_out), dtype=float)

        # activation is only stored for current iteration
        act_vec = np.empty((num_weights, num_samples, self.n_nodes), dtype=float)

        for i in range(sample_length):
            # set input nodes
            # input activation for each weight is the same (due to broadcasting)
            act_vec[..., :self.n_in] = x[:, i, :]
            act_vec[..., self.n_in] = 1  # bias is one


            if i > 0: # not the first iteration
                # propagate signal through time

                M = self.recurrent_weight_matrix

                # multiply weight matrix with base weights
                M = M[None, :, :] * weights[:, None, None]

                recurrent_sum =  np.matmul(act_vec, M)

            else:
                recurrent_sum = None


            # propagate signal through all layers
            for active_nodes in self.layers():
                if recurrent_sum is None:
                    add_to_sum = 0
                else:
                    add_to_sum = recurrent_sum[..., active_nodes - self.offset]


                act_vec[..., active_nodes] = self.calc_act(
                    act_vec, active_nodes, weights,
                    add_to_sum=add_to_sum)

            y_raw[:, :, i, :] = act_vec[..., -self.n_out:]

        # if any node is nan, we cant rely on the result
        valid = np.all(~np.isnan(act_vec), axis=-1)
        act_vec[~valid, :] = np.nan
        
        logging.debug(y_raw)

        return self.measurements_from_output(y_raw, y_true, measures)
