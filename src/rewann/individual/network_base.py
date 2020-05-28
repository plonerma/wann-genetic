import numpy as np
import logging

from itertools import dropwhile

from .util import rearrange_matrix, get_array_field


class NetworkCyclicException(Exception):
    """Raised when encountering a gene which would lead to a cyclic network."""


def build_weight_matrix(n_nodes, edges):
    """Build weight matrix for provided edges.

    Parameters
    ----------
    n_nodes : int
        The number of nodes the network consists of (including inputs, bias and
        output).
    edges : np.ndarray
        A structured array representing the edge genes of the network.

    Returns
    -------
    np.array
        :math:``
    """
    m = np.zeros((n_nodes, n_nodes), dtype=float)

    # get values from edges
    m[edges['src'], edges['dest']] = (
        get_array_field(edges, 'sign', 1)
        * get_array_field(edges, 'enabled', True)
    )
    return m


def remap_node_ids(genes):
    """Map node ids to continous node indices.

    Returns
    -------
    np.array
        Copy of gene edges with src and dest replaced by indeces.
    """

    # Assumption: There are never hidden nodes in the gene that do not have a
    # enabled edge (edges are only disabled if they are replaced by a node and
    # two connections).

    ids_in_edges = np.hstack([
        # make sure ids of static nodes stay the same even if they have no edge
        np.arange(genes.n_static),
        # translate ids in src and dest field
        genes.edges['src'], genes.edges['dest']
    ])

    # gene_node_ids[index] = node id in genes
    gene_node_ids, new_indices = np.unique(ids_in_edges, return_inverse=True)

    # copy edge genes
    new_edges = np.copy(genes.edges)

    # replace src and dest field with new indices
    new_edges['src']  = new_indices[genes.n_static:-len(genes.edges)]
    new_edges['dest'] = new_indices[genes.n_static + len(genes.edges):]

    return new_edges


class BaseFFNN:
    """Base class for Feed Forward Neural Networks"""

    is_recurrent = False

    def __init__(self, n_in, n_out, nodes, weight_matrix, conn_mat,
                 hidden_layers,
                 **params):
        self.n_in = n_in # without bias
        self.n_out = n_out
        self.nodes = nodes
        self.weight_matrix = weight_matrix
        self.conn_matrix = conn_mat
        self.hidden_layers = hidden_layers

        # For inspection
        self.params = params

    @property
    def offset(self):
        """Offset for nodes that won't be updated (inputs & bias)."""
        return self.n_in + 1

    @property
    def n_hidden(self):
        """Number of hidden nodes the network contains."""
        return len(self.nodes) - self.n_out

    @property
    def n_nodes(self):
        """Total number of nodes the network contains."""
        return self.offset + len(self.nodes)

    @property
    def n_act_funcs(self):
        """Number of enabled activations functions."""
        return len(self.available_act_functions)

    def index_to_gene_id(self, i):
        """Return gene node id from network node index."""
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
        hidden_node_order, hidden_layers = cls.sort_hidden_nodes(conn_mat[genes.n_static:, genes.n_static:])

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
            hidden_layers=hidden_layers,
        )

    def layers(self, include_input=False, include_output=True):
        """Get layers of nodes (a hierachical sorting).

        Nodes in layer :math:`i` will not have any incoming edges from nodes in
        layers :math:`\ge i`.

        Parameters
        ----------
        include_input : bool, default=False
            Yield input nodes as first layer.
        include_output : bool, default=True
            Yield output nodes as last layer.
        """
        offset = self.offset

        if include_input:
            yield np.arange(0, offset)

        for size in self.hidden_layers:
            yield np.arange(size) + offset
            offset += size

        if include_output:
            yield np.arange(self.n_out) + offset

    @property
    def n_layers(self):
        """Number of hidden layers in the network."""
        return len(self.hidden_layers)

    def node_layers(self):
        """Return layer index of each node"""
        layers = np.full(self.n_nodes, np.nan)
        for i, l in enumerate(self.layers(include_input=True)):
            layers[l] = i

        return layers

    @classmethod
    def sort_hidden_nodes(cls, conn_mat, latest_possible=True):
        """Topologically sort hidden nodes given a connectivity matrix.

        This is essentially a scheduling problem. A node can only be be
        computed once the source nodes of incoming edges have been computed.
        Additionally, it is preferable to have a minimum number of layers.
        Many schedules can potentially fullfil these requirements. Two notable
        once are `earliest possible scheduling` and `latest possible
        scheduling`. For the numpy implementation the sorting makes no
        difference (as long the number of layers does not change). For the
        torch implementation it is preferable to compute nodes at the latest
        possible moment to reduce the required memory in earlier layers. Hence,
        this is what we will use as a default.

        Parameters
        ----------
        conn_mat : np.ndarray
            Connection matrix of the hidden nodes.
        latest_possible : bool, optional
            If true (default), `latest possible scheduling` is used,
            else `earliest possible scheduling`.
        """

        # next nodes will be selected by number of blocking edges (0 -> no
        # dependencies left)
        # axis 0 -> use incoming edges
        # axis 1 -> use outgoing edges
        blocking_edges = np.sum(conn_mat, axis=(1 if latest_possible else 0))

        # for layered computation of nodes
        layers = list()
        num_sorted_nodes = 0

        for _ in range(len(conn_mat)): # maximum number of steps

            # look at all nodes with no blocking edges
            next_nodes, = np.where(blocking_edges==0)

            # If there are no new nodes to look at,
            if len(next_nodes) == 0:

                # we have already looked at all nodes and are done
                if num_sorted_nodes == conn_mat.shape[0]:
                    break

                # or there is a cylcle in the topology defined by the genes
                else:
                    raise NetworkCyclicException()

            if latest_possible:
                unblocked_edges = np.sum(conn_mat[:, next_nodes], axis=1)
            else:
                unblocked_edges =  np.sum(conn_mat[next_nodes, :], axis=0)

            # unblock edges
            blocking_edges = blocking_edges - unblocked_edges

            # don't look at those again
            blocking_edges[next_nodes] = -1

            layers.append(next_nodes)
            num_sorted_nodes += len(next_nodes)

        if latest_possible:
            layers.reverse()

        if len(layers) > 0:
            # build new order from layers
            node_order = np.concatenate(layers, axis=0)

            # only return layers sizes : the node ids will change anyways
            layers = [len(l) for l in layers]

        else:
            node_order = np.array([], dtype=int)

        return node_order, layers


class BaseRNN(BaseFFNN):
    is_recurrent = True

    def __init__(self, *args, recurrent_weight_matrix=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.recurrent_weight_matrix = recurrent_weight_matrix

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
        hidden_conn_mat = conn_mat[genes.n_static:, genes.n_static:]
        hidden_node_order, hidden_layers = cls.sort_hidden_nodes(hidden_conn_mat)

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
            hidden_layers=hidden_layers,
            recurrent_weight_matrix=re_w_mat,
        )
