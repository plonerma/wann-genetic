import numpy as np
import sklearn

import logging

from rewann.individual.network_base import BaseFFNN


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x.

    Returns:
      softmax - softmax normalized in dim axis
    """
    e_x = np.exp(x - np.expand_dims(np.max(x,axis=axis), axis=axis))
    s = (e_x / np.expand_dims(e_x.sum(axis=-1), axis=axis))
    return s


def apply_act_function(available_funcs, selected_funcs, x=None):
    """Apply the activation function of the selected nodes to their sums.

    This fullfils the same function as the
    :class:`rewann.individual.torch.ffn.MultiActivationModule`.
    """
    if x is not None:
        result = np.empty(x.shape)
        for i, func in enumerate(selected_funcs):
            assert func < len(available_funcs)
            result[..., i] = available_funcs[func][1](x[..., i])
        return result
    else:
        return np.array([  # return function names
            available_funcs[func][0] for func in selected_funcs
        ])


class Network(BaseFFNN):
    """Numpy implmentation of a Feed Forward Neural Network

    For an explanation of how propagation works, see :doc:`numpy_network`.
    """

    ### Definition of the activations functions
    available_act_functions = [
        ('relu', lambda x: np.maximum(0, x)),
        ('sigmoid', lambda x: (np.tanh(x/2.0) + 1.0)/2.0),
        ('tanh', lambda x: np.tanh(x)),
        ('gaussian (standard)', lambda x: np.exp(-np.multiply(x, x) / 2.0)),
        ('step', lambda x: 1.0*(x>0.0)),
        ('identity', lambda x: x),
        ('inverse', lambda x: -x),
        ('squared', lambda x: x**2), #  unstable if applied multiple times
        ('abs', lambda x: np.abs(x)),
        ('cos', lambda x: np.cos(np.pi*x)),
        ('sin ', lambda x: np.sin(np.pi*x)),

    ]

    def get_measurements(self, weights, x, y_true=None, measures=['predictions']):
        assert len(x.shape) == 2 # multiple one dimensional input arrays
        assert isinstance(weights, np.ndarray)

        # initial activations
        act_vec = np.empty((weights.shape[0], x.shape[0], self.n_nodes), dtype=float)
        act_vec[..., :self.n_in] = x[...]
        act_vec[..., self.n_in] = 1 # bias

        # propagate signal through all layers
        for active_nodes in self.layers():
            act_vec[..., active_nodes] = self.calc_act(act_vec, active_nodes, weights)

        # if any node is nan, we cant rely on the result
        valid = np.all(~np.isnan(act_vec), axis=-1)
        act_vec[~valid, :] = np.nan

        y_raw = act_vec[..., -self.n_out:]

        return self.measurements_from_output(y_raw, y_true, measures)

    def measurements_from_output(self, y_raw, y_true, measures):
        return_values = dict()

        if 'y_raw' in measures:
            return_values['raw'] = y_raw

        y_pred = np.argmax(y_raw, axis=-1)
        y_prob = softmax(y_raw, axis=-1)

        if 'probabilities' in measures:
            return_values['probabilities'] = y_prob

        if 'predictions' in measures:
            return_values['predictions'] = y_pred


        y_raw  = y_raw.reshape(y_raw.shape[0], -1, self.n_out)
        y_prob  = y_prob.reshape(y_raw.shape[0], -1, self.n_out)
        y_pred  = y_pred.reshape(y_raw.shape[0], -1)

        if y_true is not None:
            y_true = y_true.reshape(-1)

        if 'log_loss' in measures:
            # nan is same as maximally falsely predicted
            y_prob[~np.isfinite(y_prob)] = 0

            return_values['log_loss'] = np.array([
                sklearn.metrics.log_loss(y_true, prob)
                for prob in y_prob
            ])

        if 'mse_loss' in measures:
            return_values['mse_loss'] = np.array([
                sklearn.metrics.mean_squared_error(y_true, raw)
                for raw in y_raw
            ])

        if 'accuracy' in measures:
            return_values['accuracy'] = np.array([
                sklearn.metrics.accuracy_score(y_true, pred)
                for pred in y_pred
            ])

        if 'kappa' in measures:
            return_values['kappa'] = np.array([
                sklearn.metrics.cohen_kappa_score(y_true, pred)
                for pred in y_pred
            ])

        return return_values

    def activation_functions(self, nodes, x=None):
        funcs = self.nodes['func'][nodes - self.offset]
        return apply_act_function(self.available_act_functions, funcs, x)

    def calc_act(self, x, active_nodes, base_weights, add_to_sum=0):
        """Apply updates for active nodes (active nodes can't share edges).
        """

        addend_nodes = active_nodes[0]
        M = self.weight_matrix[:addend_nodes, active_nodes - self.offset]

        # x3d: weights, samples, source nodes
        # M3d: weights, source, target

        # multiply relevant weight matrix with base weights
        M3d = M[None, :, :] * base_weights[:, None, None]

        x3d = x[..., :addend_nodes]

        act_sums = np.matmul(x3d, M3d) + add_to_sum

        # apply activation function for active nodes
        return self.activation_functions(active_nodes, act_sums)
