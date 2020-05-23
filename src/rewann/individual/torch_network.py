import torch

from rewann.individual.network import NetworkBase

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
                        act[1](x), # apply activation func
                        self.mask[..., act[0]])) # mask output
            ),
            enumerate(self.all_act_funcs), # index, func
            torch.zeros_like(x) # start value
        )

class ConcatLayer(torch.nn.Module):
    """Contatenates output of the active nodes and prior nodes."""
    def __init__(self, shared_weight, connections, node_act_funcs, all_act_funcs):
        super().__init__()

        size_in, size_out = connections.size()
        self.linear = torch.nn.Linear(size_in, len(node_act_funcs), bias=False)
        self.weight = connections
        self.activation = MultiActivationModule(node_act_funcs, all_act_funcs)
        self.shared_weight = shared_weight

    def forward(self, x):
        linear = torch.nn.functional.linear(x, self.weight)
        linear = linear * self.shared_weight[:, None, None]
        inner_out = self.activation(linear)
        return torch.cat([x, inner_out], dim=-1)

class WannModule(torch.nn.Module):
    def __init__(self, offset, prop_steps, weight_mat, node_act_funcs, all_act_funcs):
        super().__init__()

        self.shared_weight = torch.nn.Parameter(torch.Tensor([1]))

        layers = list()

        shift = 0  # start of layer
        for step in prop_steps:
            # indices of nodes in layer
            indices = np.arange(step) + shift

            # connections from prior nodes to nodes in layer
            conns = torch.Tensor(weight_mat[:offset + shift, indices].T)

            # activation funcs of nodes in layer
            funcs = node_act_funcs[indices]

            layers.append(ConcatLayer(
                self.shared_weight,
                conns, funcs, all_act_funcs))

            shift += step

        # set up the network
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TorchNetwork(Network):
    available_act_functions = [torch.relu, torch.sigmoid, torch.tanh]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = WannModule(
            offset=self.n_in + 1, prop_steps=np.hstack([self.propagation_steps, [self.n_out]]),
            weight_mat=self.weight_matrix,
            node_act_funcs = self.nodes['func'],
            all_act_funcs=self.available_act_functions)

    def apply(self, x, weights, func='softmax'):
        assert len(x.shape) == 2 # multiple one dimensional input arrays
        assert isinstance(weights, np.ndarray)

        self.model.shared_weight.data = torch.Tensor(weights)
        x = x.expand(len(weights), -1, -1)
        y = self.model(x)

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
