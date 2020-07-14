import torch
import numpy as np
import logging

from .ffnn import MultiActivationModule, Network as TorchFFNN

from wann_genetic.individual.network_base import BaseRNN


class ReConcatLayer(torch.nn.Module):
    """Contatenates output of the active nodes and prior nodes (recurrent)."""

    def __init__(self, shared_weight, ff_weight, re_weight, node_act_funcs, all_act_funcs):
        super().__init__()

        self.ff_weight = ff_weight
        self.re_weight = re_weight
        self.shared_weight = shared_weight

        self.activation = MultiActivationModule(node_act_funcs, all_act_funcs)

    def forward(self, input):

        x_partial, last_x = input

        linear = torch.add(
            torch.nn.functional.linear(x_partial, self.ff_weight),
            torch.nn.functional.linear(last_x, self.re_weight)
        )

        linear = linear * self.shared_weight[:, None, None]

        inner_out = self.activation(linear)

        x_partial = torch.cat([x_partial, inner_out], dim=-1)

        return x_partial, last_x


class Network(BaseRNN, TorchFFNN):
    """Torch implmentation of a Recurrent Neural Network

    .. seealso::

        :doc:`torch_network`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shared_weight = torch.nn.Parameter(torch.Tensor([1]))

        all_act_funcs = [f for _, f in self.available_act_functions]

        # prepare ConcatLayers
        layers = list()

        for indices in self.layers(include_output=True):
            # connections from prior nodes to nodes in layer
            ff_weight = torch.Tensor(self.weight_matrix[:np.min(indices), indices - self.offset].T)
            re_weight = torch.Tensor(self.recurrent_weight_matrix[:, indices - self.offset].T)

            # activation funcs of nodes in layer
            funcs = self.nodes['func'][indices - self.offset]

            layers.append(ReConcatLayer(
                self.shared_weight,
                ff_weight=ff_weight,
                re_weight=re_weight,
                node_act_funcs=funcs,
                all_act_funcs=all_act_funcs))

        # set up the network
        self.model = torch.nn.Sequential(*layers)

        # share memory with all workers
        self.model.share_memory()

    def get_measurements(self, weights, x, y_true=None, measures=['predictions']):
        assert len(x.shape) == 3
        assert isinstance(weights, np.ndarray)
        assert len(weights.shape) == 1

        num_samples, sample_length, dim = x.shape
        num_weights, = weights.shape

        # outputs in each sequence step is stored
        y_raw = torch.empty((num_weights, num_samples, sample_length, self.n_out), dtype=float)

        # calculate model output
        with torch.no_grad():
            self.shared_weight.data = torch.Tensor(weights)

            state = torch.zeros(num_weights, num_samples, self.n_nodes)

            for i in range(sample_length):
                x_partial = torch.Tensor(x[:, i, :])
                # add bias to x
                bias = torch.ones(x_partial.size()[:-1] + (1,))
                x_partial = torch.cat([x_partial, bias], dim=-1)

                # expand x for the weights
                x_partial = x_partial.expand(len(weights), -1, -1)

                state, _ = self.model((x_partial, state))

            y_raw[:, :, i, :] = state[..., -self.n_out:]

        if y_true is not None:
            valid = ~np.isnan(y_true)
            y_true = y_true[valid]
            y_raw = y_raw[:, valid, :]

        return self.measurements_from_output(y_raw, y_true, measures)
