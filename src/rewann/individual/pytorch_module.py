import torch

class MultiActivationModule(torch.nn.Module):
    """Module to apply multiple elementwise activation functions to a tensor."""
    def __init__(self, node_act_funcs, all_act_funcs):

        num_nodes = len(node_act_funcs)

        mask = torch.zeros((len(all_act_funcs), num_nodes))

        for node, func in enumerate(node_act_funcs):
            mask[node, func] = 1

        self.mask = mask
        self.all_act_funcs = all_act_funcs

    def forward(self, x):
        return reduce(
            lambda first, act: (
                torch.add(
                    first,
                    act[1](
                        torch.mul(x, self.mask[act[0], ...]))
                    )
            ),
            enumerate(self.all_act_funcs), torch.zeros_like(x)
        )


class PytorchModule(nn.Module):
    def __init__(self):
        super().__init__()
        # set up the network
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()
