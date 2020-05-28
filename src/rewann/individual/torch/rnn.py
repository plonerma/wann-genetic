import torch

from .ffnn import MultiActivationModule

from rewann.individual.network_base import BaseRNN


class ReConcatLayer(torch.nn.Module):
    """Contatenates output of the active nodes and prior nodes (recurrent)."""

    def __init__(self, shared_weight, ff_weight, re_weight, node_act_funcs, all_act_funcs):
        super().__init__()

        self.ff_weight = ff_weight
        self.re_weight = re_weight
        self.shared_weight = shared_weight

        self.activation = MultiActivationModule(node_act_funcs, all_act_funcs)

    def forward(self, input):
        x, last_X = input

        linear = np.add(
            torch.nn.functional.linear(x, self.ff_weight),
            torch.nn.functional.linear(last_X, self.re_weight)
        )

        linear = linear * self.shared_weight[:, None, None]
        inner_out = self.activation(linear)
        return torch.cat([x, inner_out], dim=-1), last_X


class ReWannModule(torch.nn.Module):
    def __init__(self, offset, prop_steps, ff_weight_mat, re_weight_mat, node_act_funcs, all_act_funcs):
        super().__init__()

        self.shared_weight = torch.nn.Parameter(torch.Tensor([1]))

        layers = list()

        shift = 0  # start of layer
        for step in prop_steps:
            # indices of nodes in layer
            indices = np.arange(step) + shift

            # connections from prior nodes to nodes in layer
            ff_weight = torch.Tensor(ff_weight_mat[:offset + shift, indices].T)
            re_weight = torch.Tensor(re_weight_mat[:, indices].T)

            # activation funcs of nodes in layer
            funcs = node_act_funcs[indices]

            layers.append(ReConcatLayer(
                self.shared_weight,
                ff_weight=ff_weight, re_weight=re_weight,
                node_act_funcs=funcs, all_act_funcs=all_act_funcs))

            shift += step

        # set up the network
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Network(BaseRNN):
    """Torch implmentation of a Recurrent Neural Network

    .. seealso::

        :doc:`torch_network`.
    """

    available_act_functions = [
        ('relu', torch.relu),
        ('sigmoid', torch.sigmoid),
        ('tanh', torch.tanh)
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = WannModule(
            offset=self.n_in + 1, prop_steps=np.hstack([self.propagation_steps, [self.n_out]]),
            weight_mat=self.weight_matrix,
            node_act_funcs = self.nodes['func'],
            all_act_funcs=[f for _, f in self.available_act_functions])

        # share memory with all workers
        self.model.share_memory()

    def apply(self, x, weights, func='softmax'):
        assert len(x.shape) == 2 # multiple one dimensional input arrays
        assert isinstance(weights, np.ndarray)

        x = torch.Tensor(x)

        with torch.no_grad():
            self.model.shared_weight.data = torch.Tensor(weights)

            # add bias to x
            bias = torch.ones(x.size()[:-1] + (1,))
            x = torch.cat([x, bias], dim=-1)

            # expand x for the weights
            x = x.expand(len(weights), -1, -1)

            y = self.model(x).numpy()

        # if any node is nan, we cant rely on the result
        valid = np.all(~np.isnan(y), axis=-1)
        y[~valid, :] = np.nan

        y = y[..., -self.n_out:]

        if func == 'argmax':
            return np.argmax(y, axis=-1)
        elif func == 'softmax':
            return softmax(y, axis=-1)
        else:
            return y
