import torch

import numpy as np
import sklearn

import logging

from functools import reduce

from rewann.individual.network_base import BaseFFNN


class MultiActivationModule(torch.nn.Module):
    """Applies multiple elementwise activation functions to a tensor."""
    def __init__(self, node_act_funcs, all_act_funcs):
        super().__init__()
        num_nodes = len(node_act_funcs)

        mask = torch.zeros((num_nodes, len(all_act_funcs)))

        for node, func in enumerate(node_act_funcs):
            mask[node, func] = 1

        self.mask = mask
        self.all_act_funcs = all_act_funcs

    def forward(self, x):
        return reduce(
            lambda first, act: (
                torch.add(
                    first,
                    torch.mul(
                        act[1](x),  # apply activation func
                        self.mask[..., act[0]]))  # mask output
            ),
            enumerate(self.all_act_funcs),  # index, func
            torch.zeros_like(x)  # start value
        )


class ConcatLayer(torch.nn.Module):
    """Contatenates output of the active nodes and prior nodes."""
    def __init__(self, shared_weight, connections, node_act_funcs, all_act_funcs):
        super().__init__()

        self.weight = connections
        self.activation = MultiActivationModule(node_act_funcs, all_act_funcs)
        self.shared_weight = shared_weight

    def forward(self, x):
        linear = torch.nn.functional.linear(x, self.weight)
        linear = linear * self.shared_weight[:, None, None]
        inner_out = self.activation(linear)
        return torch.cat([x, inner_out], dim=-1)


class Network(BaseFFNN):
    """Torch implmentation of a Feed Forward Neural Network

    .. seealso::

        :doc:`torch_network`.
    """

    available_act_functions = [
        ('relu', torch.relu),
        ('sigmoid', torch.sigmoid),
        ('tanh', torch.tanh),
        ('gaussian (standard)', lambda x: torch.exp(-torch.square(x) / 2.0)),
        ('step', lambda t: (t > 0.0) * 1.0),
        ('identity', lambda x: x),
        ('inverse', torch.neg),
        ('squared', torch.square),
        ('abs', torch.abs),
        ('cos', torch.cos),
        ('sin', torch.sin),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shared_weight = torch.nn.Parameter(torch.Tensor([1]))

        all_act_funcs = [f for _, f in self.available_act_functions]

        # prepare ConcatLayers
        layers = list()

        for indices in self.layers(include_output=True):
            # connections from prior nodes to nodes in layer
            conns = torch.Tensor(self.weight_matrix[:np.min(indices), indices - self.offset].T)

            # activation funcs of nodes in layer
            funcs = self.nodes['func'][indices - self.offset]

            layers.append(ConcatLayer(
                self.shared_weight,
                conns, funcs, all_act_funcs))

        # set up the network
        self.model = torch.nn.Sequential(*layers)

        # share memory with all workers
        self.model.share_memory()

    def get_measurements(self, weights, x, y_true=None, measures=['predictions']):
        assert len(x.shape) == 2  # multiple one dimensional input arrays
        assert isinstance(weights, np.ndarray)

        x = torch.Tensor(x)

        # calculate model output
        with torch.no_grad():
            self.shared_weight.data = torch.Tensor(weights)

            # add bias to x
            bias = torch.ones(x.size()[:-1] + (1,))
            x = torch.cat([x, bias], dim=-1)

            # expand x for the weights
            x = x.expand(len(weights), -1, -1)

            y = self.model(x)

        y_raw = y[..., -self.n_out:]

        return self.measurements_from_output(y_raw, y_true, measures)

    def measurements_from_output(self, y_raw, y_true, measures):
        return_values = dict()

        if 'raw' in measures:
            return_values['raw'] = y_raw

        with torch.no_grad():
            y_pred = torch.argmax(y_raw, dim=-1).numpy()

            if 'probabilities' in measures:
                return_values['probabilities'] = torch.softmax(y_raw, dim=-1).numpy()

            if 'predictions' in measures:
                return_values['predictions'] = y_pred

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

            if y_true is not None:
                y_true = torch.LongTensor(y_true.astype(int))

                if 'log_loss' in measures:
                    return_values['log_loss'] = np.array([
                        torch.nn.functional.cross_entropy(w_y_raw, y_true)
                        for w_y_raw in y_raw
                    ])

                if 'mse_loss' in measures:
                    return_values['mse_loss'] = np.array([
                        torch.nn.functional.mse_loss(w_y_raw, y_true)
                        for w_y_raw in y_raw
                    ])

        return return_values
