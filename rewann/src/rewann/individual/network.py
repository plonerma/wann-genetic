import numpy as np

from .expression import (
    apply_act_function, remap_node_ids, get_array_field, sort_hidden_nodes,
    weight_matrix_arrangement, softmax)

class Network:
    """Representation for all kinds of NNs (wann, ffnn, rnn, rewann)."""

    ### Definition of the activations functions
    available_act_functions = [
#        ('linear', lambda x: x),
        ('step (unsigned)', lambda x: 1.0*(x>0.0)),
#        ('sin ', lambda x: np.sin(np.pi*x)),
        ('gaussian with mean 0 and sigma 1', lambda x: np.exp(-np.multiply(x, x) / 2.0)),
        ('tanh (signed)', lambda x: np.tanh(x)),
        ('sigmoid (unsigned)', lambda x: (np.tanh(x/2.0) + 1.0)/2.0),
        ('inverse linear', lambda x: -x),
#        ('abs', lambda x: np.abs(x)),
        ('relu', lambda x: np.maximum(0, x)),
#        ('cos', lambda x: np.cos(np.pi*x)),
#        ('squared', lambda x: x**2),
    ]

    def __init__(self, n_in, n_out, nodes, weight_matrix,
                 propagation_steps=None,
                 **params):
        # Relevant for computation
        self.n_in = n_in # without bias
        self.n_out = n_out
        self.nodes = nodes
        self.weight_matrix = weight_matrix
        self.propagation_steps = propagation_steps

        # For inspection
        self.params = params

    @property
    def offset(self):
        """Offset for nodes that won't be updated (inputs & bias)."""
        return self.n_in + 1

    @property
    def n_hidden(self):
        return len(self.nodes) - self.n_out

    @property
    def n_nodes(self):
        return self.offset + len(self.nodes)

    @property
    def n_act_funcs(self):
        return len(self.available_act_functions)

    def index_to_gene_id(self, i):
        if i < self.offset:
            return i
        else:
            return self.nodes[i - self.offset]['id']

    @classmethod
    def from_genes(cls, genes, **kwargs):
        """Convert genes to weight matrix and activation vector."""
        edges = remap_node_ids(genes)

        # actual number of nodes that will be present in the network
        n_nodes = len(genes.nodes) + genes.n_in + 1

        w_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        conn_mat = np.zeros((n_nodes, n_nodes), dtype=int)

        # if there is a disabled connection between two nodes, there is a
        # directed path between the two anyways
        conn_mat[edges['src'], edges['dest']] = 1

        # reorder hidden nodes
        hidden_node_order, prop_steps = sort_hidden_nodes(conn_mat[genes.n_static:, genes.n_static:])

        # output nodes appear first in genes and last in network nodes
        nodes = np.empty(genes.nodes.shape, dtype=genes.nodes.dtype)
        nodes[: -genes.n_out] = genes.nodes[hidden_node_order + genes.n_out]
        nodes[-genes.n_out: ] = genes.nodes[:genes.n_out]

        # if a field does not exist, use 1 as default
        w_matrix[edges['src'], edges['dest']] = get_array_field(edges, 'enabled', 1) * get_array_field(edges, 'weight', 1)

        # rearrange weight matrix
        i_rows, i_cols = weight_matrix_arrangement(genes.n_in, genes.n_out, hidden_node_order)
        w_matrix = w_matrix[i_rows, :]
        w_matrix = w_matrix[:, i_cols]

        return cls(
            n_in=genes.n_in, n_out=genes.n_out,
            nodes=nodes,
            weight_matrix=w_matrix,
            propagation_steps=prop_steps,
        )

    def layers(self, including_input=False):
        if self.propagation_steps is None:
            return np.arange(self.n_hidden + self.n_out)
        i = self.offset
        if including_input:
            yield np.arange(0, i)
        for n in self.propagation_steps:
            yield np.arange(i, i+n)
            i = i + n
        yield np.arange(i, i+self.n_out)

    def node_layers(self):
        """Return layer index for each node"""
        layers = np.full(self.n_nodes, np.nan)
        for l, i in enumerate(self.layers(include_input=True)):
            layers[l] = i

    def apply(self, x, func='softmax', return_activation=False, weights=1):
        # one dimensional for single input, two dimensional for multiple inputs
        assert len(x.shape) <= 2

        multiple_inputs = (len(x.shape) == 2)

        if isinstance(weights, list):
            weights = np.array(weights)

        multiple_weights = isinstance(weights, np.ndarray)

        if not multiple_inputs: x = np.expand_dims(x, axis=0)

        if not multiple_weights: weights = np.array([weights])

        # initial activations
        x_full = np.empty((weights.shape[0], x.shape[0], self.n_nodes))
        x_full[...] = np.nan
        x_full[..., :self.n_in] = x[:, :self.n_in]
        x_full[..., self.n_in] = 1 # bias

        y_full = self.fully_propagate(x_full, w=weights)

        # only look at output activations
        y = y_full[..., -self.n_out:]

        if func == 'argmax':
            y = np.argmax(y, axis=-1)
        elif func == 'softmax':
            y = softmax(y, axis=-1)


        return_slice = (
            np.s_[:] if multiple_weights else np.s_[0],
            np.s_[:] if multiple_inputs else np.s_[0],
        )

        y = y[return_slice]

        if return_activation:
            y_full = y_full[return_slice]
            return y, y_full
        else:
            return y

    def fully_propagate(self, act_vec, w):
        """Iterate through all nodes that can be updated."""
        for active_nodes in self.layers():
            act_vec = self.propagate(act_vec, active_nodes, w=w)
        return act_vec

    def activation_functions(self, nodes, x=None):
        return apply_act_function(self.available_act_functions,
                                  self.nodes['func'][nodes], x)

    def propagate(self, x_full, active_nodes, w):
        """Apply updates for active nodes (active nodes can't share edges).

        Args:
            act_vec: current activation values of each node
            active_nodes: nodes to propagate in this step
            slice_in : slice of act_vec to use (less null-calculations)
        """
        # at most use all input, bias, and hidden nodes

        # calculate sum of all incoming edges for each active node
        # use all nodes before first active node as input

        M = self.weight_matrix[:active_nodes[0], active_nodes - self.offset] # Only return sums for active nodes

        # multiply relevant weight matrix with base weights w
        M3d = M[None, :, :] * w[:, None, None]

        # use same inputs for all weights and use all samples,
        # use all nodes lower then the first active node as inputs
        x3d = x_full[..., :active_nodes[0]]

        act_sums = np.matmul(x3d, M3d)

        # apply activation function for active nodes
        y = self.activation_functions(active_nodes - self.offset, act_sums)

        #st.write(
        #    "active_nodes", active_nodes,
        #    "x", x_full,
        #    "relevant part of weight matrix", M,
        #    "sum", act_sum,
        #    "y", y,
        #)

        x_full[:, :, active_nodes] = y
        return x_full
