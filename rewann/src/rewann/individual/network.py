import numpy as np

from .genes import Genotype
from ..util import serialize_array, deserialize_array
from .ann import (
    apply_act_function, remap_node_ids, get_array_field, sort_hidden_nodes,
    weight_matrix_arrangement, softmax)

import streamlit as st

class Network:
    """Representation for all kinds of NNs (wann, ffnn, rnn, rewann)."""

    ### Definition of the activations functions
    available_act_functions = [
        ('linear', lambda x: x),
        ('step (unsigned)', lambda x: 1.0*(x>0.0)),
        ('sin ', lambda x: np.sin(np.pi*x)),
        ('gaussian with mean 0 and sigma 1', lambda x: np.exp(-np.multiply(x, x) / 2.0)),
        ('tanh (signed)', lambda x: np.tanh(x)),
        ('sigmoid (unsigned)', lambda x: (np.tanh(x/2.0) + 1.0)/2.0),
        ('inverse linear', lambda x: -x),
        ('abs', lambda x: np.abs(x)),
        ('relu', lambda x: np.maximum(0, x)),
        ('cos', lambda x: np.cos(np.pi*x)),
        ('squared', lambda x: x**2),
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
    def from_genes(cls, genes : Genotype, **kwargs):
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

    def initial_act_vec(self, x):
        x_full = np.empty((x.shape[0], self.n_nodes))
        x_full[:, :] = np.nan
        x_full[:, :self.n_in] = x[:, :self.n_in]
        x_full[:, self.n_in] = 1 # bias
        return x_full

    def apply(self, x, func='softmax', return_activation=False):
        changed_shape = False
        if len(x.shape) == 1:
            x = np.array([x])
            changed_shape = True


        y_full = self.fully_propagate(self.initial_act_vec(x))
        y = y_full[:, -self.n_out:]

        if func == 'argmax':
            y = np.argmax(y, axis=1)
        elif func == 'softmax':
            y = softmax(y, axis=1)

        if changed_shape:
            y = y[0]
            y_full = y_full[0]

        if return_activation:
            return y, y_full
        else:
            return y

    def fully_propagate(self, act_vec): # activation vector
        """Iterate through all nodes that can be updated."""
        for active_nodes in self.layers():
            act_vec = self.propagate(act_vec, active_nodes)
        return act_vec

    def activation_functions(self, nodes, x=None):
        return apply_act_function(self.available_act_functions,
                                  self.nodes['func'][nodes], x)

    def propagate(self, x_full, active_nodes):
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

        act_sum = np.dot(x_full[:, :active_nodes[0]], M)

        # apply activation function for active nodes
        y = self.activation_functions(active_nodes - self.offset, act_sum)

        #st.write(
        #    "active_nodes", active_nodes,
        #    "x", x_full,
        #    "relevant part of weight matrix", M,
        #    "sum", act_sum,
        #    "y", y,
        #)

        x_full[:, active_nodes] = y
        return x_full

    def to_pytorch(self):
        raise NotImplemented

    def serialize(self):
        raise NotImplemented

    def deserialize(self):
        raise NotImplemented

    @property
    def node_names(self):
        return (
            [f"$x_{i}$" for i in range(self.n_in)]          # inputs
            + ["$b$"]                               # bias
            + [f"$h_{i}$" for i in range(self.n_hidden)]    # hidden
            + [f"$y_{i}$" for i in range(self.n_out)]       # outputs
        )


    def draw_graph(self, ax=None, activation=None, pos_iterations=None):
        import networkx as nx
        g = nx.DiGraph()

        # Add nodes
        nodes = self.node_names
        g.add_nodes_from(nodes)

        # Colors
        color=(
            ['#fdb462'] * self.n_in         # inputs
            + ['#ffed6f']                   # bias
            + ['#80b1d3'] * self.n_hidden   # hidden
            + ['#b3de69'] * self.n_out      # outputs
        )

        # Positions
        N = len(nodes)

        layers = np.array([len(s) for s in self.layers(including_input=True)])

        # pos x will be determined by layer, pos y will be iterated on
        # input, bias and output nodes will have static position, hidden node
        # positions will be determine iteratively
        pos = np.array([
            ((li), 0)
            for li, ln in enumerate(layers)
            for ni in range(ln)
        ], dtype=[('x', float), ('y', float)])

        spread = lambda n: ((n-1) / 2 - np.arange(n)) / (n + 1)

        pos['y'][0: self.offset] = spread(self.offset)
        pos['y'][-self.n_out:] = spread(self.n_out)

        a = self.weight_matrix[:self.offset, :self.n_hidden]
        b = self.weight_matrix[self.offset:, :self.n_hidden]
        c = self.weight_matrix[self.offset:, self.n_hidden:]

        M = np.vstack([
            a, b+b.T, c.T
        ])

        M_sum = np.sum(M,axis=0)
        M_sum[np.where(M_sum == 0)] = 1

        M = M / M_sum

        if pos_iterations is None:
            pos_iterations = len(layers)

        for i in range(pos_iterations):
            update = np.dot(pos['y'], M)
            pos['y'][self.offset:-self.n_out] = update

        pos = dict(zip(nodes, np.array([pos['x'], pos['y']]).T))

        if activation is not None:
            node_size = 600 * activation / np.max(activation)
        else:
            node_size = 300

        # Edges
        for row, col in zip(*np.where(self.weight_matrix != 0)):
            ns = nodes[row], nodes[col + self.offset]
            w = self.weight_matrix[row, col]
            a = 1 if activation is None else activation[row]
            g.add_edge(*ns, weight=w, activation=a)

        edge_weights = np.array([g[u][v]['weight'] for u,v in g.edges()])
        edge_activ = np.array([g[u][v]['activation'] for u,v in g.edges()])

        edge_color = -edge_weights*edge_activ
        edge_widths = edge_weights

        nx.draw(
            g, ax=ax, pos=pos, node_color=color,
            with_labels=True, node_size=node_size, font_color="k", font_size=9,
            arrows=False, linewidths=0,
            edge_color=edge_color, vmax=0, vmin=np.min(edge_color), cmap='magma',
            width=edge_widths,
            label="CAPTION")

    def draw_weight_matrix(self, ax=None):
        x_ticks = list(range(self.n_hidden + self.n_out))
        x_ticklabels = self.node_names[self.offset:]
        y_ticks = list(range(self.n_nodes - self.n_out))
        y_ticklabels = self.node_names[:self.n_out]

        if ax is None:
            import matplotlib.pyplot as plt
            plt.xticks(x_ticks, x_ticklabels)
            plt.yticks(y_ticks, y_ticklabels)
            imshow = plt.imshow
        else:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels)
            imshow = ax.imshow

        imshow(np.max(self.weight_matrix) - self.weight_matrix, cmap='magma')
