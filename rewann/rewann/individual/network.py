import numpy as np

from .genes import Genotype
from .ann import apply_act_function, remap_node_ids, get_array_field, sort_hidden_nodes, weight_matrix_arrangement
from ..util import serialize_array, deserialize_array

import streamlit as st

class Network:
    """Representation for all kinds of NNs (wann, ffnn, rnn, rewann)."""

    def __init__(self, n_in, n_out, activation_funcs, weight_matrix,
                 **params):
        # Relevant for computation
        self.n_in = n_in # without bias
        self.n_out = n_out
        self.n_hidden = weight_matrix.shape[1] - n_out
        self.act_funcs = activation_funcs
        self.weight_matrix = weight_matrix

        # Speeds up computation
        self.propagation_steps = params.get('propagation_steps', None)

        self.n_nodes = n_in + 1 + weight_matrix.shape[1]

        # For inspection
        self.params = params

    @property
    def offset(self):
        """Offset for nodes that won't be updated (inputs & bias)."""
        return self.n_in + 1

    @classmethod
    def from_genes(cls, genes : Genotype):
        """Convert genes to weight matrix and activation vector."""
        hidden_nodes, edges = remap_node_ids(genes)

        # actual number of nodes that will be present in the network
        n_nodes = len(hidden_nodes) + genes.n_static

        w_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        conn_mat = np.zeros((n_nodes, n_nodes), dtype=int)

        # if there is no `enabled` field in the edge encoding enable by default
        enabled = get_array_field(edges, 'enabled', 1)
        conn_mat[edges['src'], edges['dest']] = enabled

        # if there is no `weight` field in the edge encoding use default weight 1
        w_matrix[edges['src'], edges['dest']] = enabled * get_array_field(edges, 'weight', 1)

        # reorder hidden nodes
        hidden_node_order, prop_steps = sort_hidden_nodes(conn_mat[genes.n_static:, genes.n_static:])
        hidden_nodes = hidden_nodes[hidden_node_order]

        # rearrange weight matrix
        i_rows, i_cols = weight_matrix_arrangement(genes.n_in, genes.n_out, hidden_node_order)
        w_matrix = w_matrix[i_rows, :]
        w_matrix = w_matrix[:, i_cols]

        return cls(
            n_in=genes.n_in, n_out=genes.n_out,
            # output nodes appear first in node genes and last in act-func vector
            activation_funcs=np.append(hidden_nodes['func'], genes.nodes['func'][:genes.n_out]),
            weight_matrix=w_matrix,
            propagation_steps=prop_steps,
            num_enabled_connections=np.sum(enabled),
            hidden_nodes=hidden_nodes,
            n_nodes=n_nodes
        )

    def layers(self):
        if self.propagation_steps is None:
            return np.arange(self.n_hidden + self.n_out)
        i = self.offset
        for n in self.propagation_steps:
            yield np.arange(i, i+n)
            i = i + n
        yield np.arange(i, i+self.n_out)

    def apply(self, x, return_complete=False):
        x_full = np.empty((self.n_nodes))
        x_full[:] = np.nan
        x_full[:self.n_in] = x[:self.n_in]
        x_full[self.n_in] = 1 # bias
        y = self.fully_propagate(x_full)
        if not return_complete:
            y = y[-self.n_out:]
        return y

    def fully_propagate(self, act_vec): # activation vector
        """Iterate through all nodes that can be updated."""
        for active_nodes in self.layers():
            act_vec = self.propagate(act_vec, active_nodes)
        return act_vec

    def propagate(self, act_vec, active_nodes):
        """Apply updates for active nodes (active nodes can't share edges).

        Args:
            act_vec: current activation values of each node
            active_nodes: nodes to propagate in this step
            slice_in : slice of act_vec to use (less null-calculations)
        """
        # at most use all input, bias, and hidden nodes

        # calculate sum of all incoming edges for each active node
        # use all nodes before first active node as input
        s_in = np.s_[:active_nodes[0]]


        x = act_vec[s_in]
        M = self.weight_matrix[s_in, active_nodes - self.offset] # Only return sums for active nodes

        act_sum = np.dot(x, M)

        # apply activation function for active nodes
        act = apply_act_function(self.act_funcs[active_nodes - self.offset], act_sum)

        #st.write(
        #    "active_nodes", active_nodes,
        #    "x", x,
        #    "relevant part of weight matrix", M,
        #    "sum", act_sum,
        #    "activation", act
        #)

        # offset output indeces by number of nodes that won't be updated
        act_vec[active_nodes] = act
        return act_vec

    def to_pytorch(self):
        pass

    def serialize(self):
        pass

    def deserialize(self):
        pass

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

        layers = [len(s) for s in self.layers()]
        layers = np.array([(self.offset), *layers])

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

        M = M / np.sum(M,axis=0)

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
