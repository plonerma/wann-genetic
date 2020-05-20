"""Low-level functions for translating genes to anns."""


import numpy as np

from itertools import dropwhile

## Translating genes to network

class NetworkCyclicException(Exception):
    """Raised when encountering a gene which would lead to a cyclic network."""


def build_weight_matrix(n_nodes, edges):
    """Build weight matrix for provided edges."""
    m = np.zeros((n_nodes, n_nodes), dtype=float)

    # get values from edges
    m[edges['src'], edges['dest']] = (
        get_array_field(edges, 'sign', 1)
        * get_array_field(edges, 'enabled', True)
    )
    return m

def rearrange_matrix(m, indices):
    """Rearrange matrix `m` according to provided indices."""
    # rearrange
    i_rows, i_cols = indices
    m = m[i_rows, :]
    m = m[:, i_cols]
    return m

def get_array_field(array : np.ndarray, key : str, default=None):
    """Return field if it exists else return default value."""
    return array[key] if key in array.dtype.names else default


def remap_node_ids(genes):
    """Map node ids to continous node indices.

    Returns
    -------
        np.array - copy of gene edges with src and dest replaced by indeces
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


def sort_hidden_nodes(conn_mat):
    """Topologically sorting hidden nodes given a connectivity matrix."""

    # next nodes will be selected by number of incoming edges (0 -> no
    # dependencies left)
    edges_in = np.sum(conn_mat,axis=0)

    # stored for computing nodes faster (number of nodes that can be computed
    # during a propagation step)
    propagation_steps = np.array([], dtype=int)

    hidden_node_order = np.array([], dtype=int)

    for _ in range(len(conn_mat)): # maximum number of steps

        # Look at all nodes with no incoming connections
        next_nodes, = np.where(edges_in==0)

        # If there are no new nodes to look at,
        if len(next_nodes) == 0:
            # we have already looked at all nodes and are done
            if len(hidden_node_order) == conn_mat.shape[0]:
                break
            # or there is a cylcle in the topology defined by the genes
            else:
                raise NetworkCyclicException()

        edges_out = np.sum(conn_mat[next_nodes, :], axis=0)
        edges_in = edges_in - edges_out
        # don't look at those again
        edges_in[next_nodes] = -1

        hidden_node_order = np.append(hidden_node_order, next_nodes)
        propagation_steps = np.append(propagation_steps, [len(next_nodes)])

    return hidden_node_order, propagation_steps


## Applying activation functions to an array of nodes

def apply_act_function(available_funcs, selected_funcs, x=None):
    if x is not None:
        result = np.empty(x.shape)
        for i, func in enumerate(selected_funcs):
            assert func < len(available_funcs)
            result[..., i] = available_funcs[func][1](x[..., i])
        return result
    else:
        return np.array([  # return function names
            available_funcs[func][0] for func in selected_funcs
        ])



def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x.

    Returns:
      softmax - softmax normalized in dim axis
    """
    e_x = np.exp(x - np.expand_dims(np.max(x,axis=axis), axis=axis))
    s = (e_x / np.expand_dims(e_x.sum(axis=-1), axis=axis))
    return s
