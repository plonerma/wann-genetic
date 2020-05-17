import numpy as np
import logging

from .expression import (
    apply_act_function, remap_node_ids, sort_hidden_nodes,
    softmax, get_array_field, build_weight_matrix, rearrange_matrix)




class Network:
    """Representation for all kinds of NNs (wann, ffnn, rnn, rewann)."""

    ### Definition of the activations functions
    available_act_functions = [
        ('relu', lambda x: np.maximum(0, x)),
        ('sigmoid', lambda x: (np.tanh(x/2.0) + 1.0)/2.0),
        ('tanh', lambda x: np.tanh(x)),
        ('gaussian (standard)', lambda x: np.exp(-np.multiply(x, x) / 2.0)),
        ('step', lambda x: 1.0*(x>0.0)),
        ('identity', lambda x: x),
        ('inverse', lambda x: -x),
        #('squared', lambda x: x**2), #  unstable if applied multiple times
        ('abs', lambda x: np.abs(x)),
        ('cos', lambda x: np.cos(np.pi*x)),
        ('sin ', lambda x: np.sin(np.pi*x)),

    ]

    def __init__(self, n_in, n_out, nodes, weight_matrix, conn_mat,
                 propagation_steps=None,
                 **params):
        # Relevant for computation
        self.n_in = n_in # without bias
        self.n_out = n_out
        self.nodes = nodes
        self.weight_matrix = weight_matrix
        self.conn_matrix = conn_mat
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
        return i if (i < self.offset) else self.nodes[i - self.offset]['id']

    @classmethod
    def from_genes(cls, genes, **kwargs):
        """Convert genes to weight matrix and activation vector."""
        edges = remap_node_ids(genes)

        # actual number of nodes that will be present in the network
        n_in = genes.n_in
        n_out = genes.n_out
        n_nodes = len(genes.nodes) + genes.n_in + 1

        conn_mat = np.zeros((n_nodes, n_nodes), dtype=bool)   # connectivity

        conn_mat[edges['src'], edges['dest']] = True

        # reorder hidden nodes
        hidden_node_order, prop_steps = sort_hidden_nodes(conn_mat[genes.n_static:, genes.n_static:])

        indices_in  = np.r_[: n_in + 1] # includes bias
        indices_out = np.r_[  n_in + 1 : n_in + 1 + n_out]
        indices_hid = n_in + 1 + n_out + hidden_node_order

        w_matrix = build_weight_matrix(n_nodes, edges)

        indices = np.append(indices_in, indices_hid), np.append(indices_hid, indices_out)

        conn_mat = rearrange_matrix(conn_mat, indices)
        w_matrix = rearrange_matrix(w_matrix, indices)

        # output nodes appear first in genes and last in network nodes
        nodes = np.empty(genes.nodes.shape, dtype=genes.nodes.dtype)
        nodes[: -genes.n_out] = genes.nodes[hidden_node_order + genes.n_out]
        nodes[-genes.n_out: ] = genes.nodes[:genes.n_out]

        return cls(
            n_in=genes.n_in, n_out=genes.n_out,
            nodes=nodes,
            weight_matrix=w_matrix,
            conn_mat=conn_mat!=0,
            propagation_steps=prop_steps,
        )

    def layers(self, including_input=False):
        o = self.offset
        if including_input:
            yield np.arange(0, o)

        if self.propagation_steps is None:
            yield np.arange(self.n_hidden) + o
            o += self.n_hidden
        else:
            for n in self.propagation_steps:
                yield np.arange(n) + o
                o += n
        yield np.arange(self.n_out) + o

    @property
    def n_layers(self):
        return len(self.propagation_steps)

    def node_layers(self):
        """Return layer index for each node"""
        layers = np.full(self.n_nodes, np.nan)
        for i, l in enumerate(self.layers(including_input=True)):
            layers[l] = i

        return layers

    def apply(self, x, func='softmax', weights=1):
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

        y = act_vec[..., -self.n_out:]

        if func == 'argmax':
            return np.argmax(y, axis=-1)
        elif func == 'softmax':
            return softmax(y, axis=-1)
        else:
            return y

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


class RecurrentNetwork(Network):
    def __init__(self, *args, recurrent_weight_matrix=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.recurrent_weight_matrix = recurrent_weight_matrix


    def apply(self, x, weights, func='softmax'):
        assert len(x.shape) == 3
        num_samples, sample_length, dim = x.shape

        assert dim == self.n_in

        num_weights, = weights.shape  # weight array should be one-dimensional

        # outputs in each sequence step is stored
        outputs = np.empty((num_weights, num_samples, sample_length, self.n_out), dtype=float)

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

            outputs[:, :, i, :] = act_vec[..., -self.n_out:]

        if func == 'argmax':
            return np.argmax(outputs, axis=-1)
        elif func == 'softmax':
            return softmax(outputs, axis=-1)
        else:
            return outputs

    @classmethod
    def from_genes(cls, genes, **kwargs):
        """Convert genes to weight matrix and activation vector."""
        edges = remap_node_ids(genes)

        n_in = genes.n_in
        n_out = genes.n_out
        n_nodes = len(genes.nodes) + n_in + 1

        conn_mat = np.zeros((n_nodes, n_nodes), dtype=bool)   # connectivity

        recurrent = get_array_field(edges, 'recurrent', False)

        # only use feed forward edges for topological sorting
        conn_mat[edges[~recurrent]['src'], edges[~recurrent]['dest']] = True

        # reorder hidden nodes
        hidden_node_order, prop_steps = sort_hidden_nodes(conn_mat[genes.n_static:, genes.n_static:])

        indices_in  = np.r_[: n_in + 1] # includes bias
        indices_out = np.r_[  n_in + 1 : n_in + 1 + n_out]
        indices_hid = n_in + 1 + n_out + hidden_node_order

        ff_w_mat = build_weight_matrix(n_nodes, edges[~recurrent])
        re_w_mat = build_weight_matrix(n_nodes, edges[ recurrent])

        src_ff = np.hstack([indices_in, indices_hid])
        src_re = np.hstack([indices_in, indices_hid, indices_out])
        dst    = np.hstack([indices_hid, indices_out])

        conn_mat = rearrange_matrix(conn_mat, (src_ff, dst))
        ff_w_mat = rearrange_matrix(ff_w_mat, (src_ff, dst))
        re_w_mat = rearrange_matrix(re_w_mat, (src_re, dst))

        # output nodes appear first in genes and last in network nodes
        nodes = np.empty(genes.nodes.shape, dtype=genes.nodes.dtype)
        nodes[: -genes.n_out] = genes.nodes[hidden_node_order + genes.n_out]
        nodes[-genes.n_out: ] = genes.nodes[:genes.n_out]

        return cls(
            n_in=genes.n_in, n_out=genes.n_out,
            nodes=nodes,
            weight_matrix=ff_w_mat,
            conn_mat=conn_mat!=0,
            propagation_steps=prop_steps,
            recurrent_weight_matrix=re_w_mat,
        )
