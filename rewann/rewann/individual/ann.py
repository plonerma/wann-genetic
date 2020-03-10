"""Low-level functions for working with unstructured neural networks."""


import numpy as np

from itertools import dropwhile

## Translating genes to network

class NetworkCyclicException(Exception):
    """Raised when encountering a which would lead to a cyclic network."""

def weight_matrix_arrangement(n_in, n_out, hidden_node_order):
    indices_in  = np.r_[: n_in + 1] # includes bias
    indices_out = np.r_[  n_in + 1 : n_in + 1 + n_out]
    indices_hid = n_in + 1 + n_out + hidden_node_order

    # indices to use for rows, columns
    return np.append(indices_in, indices_hid), np.append(indices_hid, indices_out)

def get_array_field(array, key, default=None):
    return array[key] if key in array.dtype.names else default

def remap_node_ids(genes):
    """Remove unconnected nodes and map node ids to continous node indices.

    Returns:
        edges: np.array - copy of gene edges with src and dest replaced by indeces
        hidden_nodes: np.array - representation of the hidden nodes
    """

    ids_in_edges = np.hstack([
        np.arange(genes.n_static),  # make sure ids for static nodes stay the same
        genes.edges['src'], genes.edges['dest']  # translate ids in src and dest field
    ])

    # gene_node_ids[index] = node id in genes
    gene_node_ids, new_indices = np.unique(ids_in_edges, return_inverse=True)

    # copy edge genes
    new_edges = np.array(genes.edges, dtype=genes.edges.dtype)

    # replace src and dest field with new indices
    new_edges['src']  = new_indices[genes.n_static:-len(genes.edges)]
    new_edges['dest'] = new_indices[-len(genes.edges):]

    # Remove unconnected hidden nodes
    remaining_hidden_nodes = iter(genes.nodes[genes.n_out:])
    new_hidden_nodes = np.array([
        # both arrays (genes.nodes and gene_node_ids) are ordered
        # drop nodes until right id is found
        next(dropwhile(lambda node: node['id'] != hidden_id, remaining_hidden_nodes))
        for hidden_id in gene_node_ids[genes.n_static:]
    ],  dtype=genes.nodes.dtype)

    return new_hidden_nodes, new_edges



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

### Definition of the activations functions
activation_functions = {
    0: lambda x: x,                                # Linear
    1: lambda x: 1.0*(x>0.0),                      # Unsigned Step Function
    2: lambda x: np.sin(np.pi*x),                  # Sin
    3: lambda x: np.exp(-np.multiply(x, x) / 2.0), # Gaussian with mean 0 and sigma 1
    4: lambda x: np.tanh(x) ,                      # Hyperbolic Tangent (signed)
    5: lambda x: (np.tanh(x/2.0) + 1.0)/2.0,       # Sigmoid (unsigned)
    6: lambda x: -x,                               # Inverse
    7: lambda x: np.abs(x),                        # Absolute Value
    8: lambda x: np.maximum(0, x),                 # Relu
    9: lambda x: np.cos(np.pi*x),                 # Cosine
    10: lambda x: x**2,                            # Squared

    #### Unused functions:
    # (np.tanh(50*x/2.0) + 1.0)/2.0
}

@np.vectorize  # vectorize function so it can be applied to an array
def apply_act_function(func, x):
    return activation_functions[func](x)
