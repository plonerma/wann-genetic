import numpy as np
import logging

def add_node(ind, env, innov):
    # Choose an enabled edge
    options, = np.where(ind.genes.edges['enabled'])
    if len(options) == 0: return None

    edge_to_disable = np.random.choice(options)

    # new node with random act func
    new_node = np.zeros(1, dtype=ind.genes.nodes.dtype)
    new_node['id'] = innov.next_node_id()
    new_node['func'] = np.random.randint(ind.network.n_act_funcs)

    new_edges = np.zeros(2, dtype=ind.genes.edges.dtype)
    new_edges['enabled'] = True

    # edge from src to new node
    new_edges[0]['src'] = ind.genes.edges[edge_to_disable]['src']
    new_edges[0]['dest'] = new_node['id']

    # edge from new node to old dest
    new_edges[1]['src'] = new_node['id']
    new_edges[1]['dest'] = ind.genes.edges[edge_to_disable]['dest']

    new_edges['id'] = innov.next_edge_id(), innov.next_edge_id()

    edges = np.append(ind.genes.edges, new_edges)  # creates copy of old edges as well

    # disable edge
    edges[edge_to_disable]['enabled'] = False

    return ind.genes.__class__(
        edges=edges, nodes=np.append(ind.genes.nodes, new_node),
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def add_edge_layer_based(ind, env, innov):
    """Layer-based edge introduction strategy.

    The implementation of this method is equivalent to strategy used in the
    original implementation.
    """

def add_edge_layer_agnostic(ind, env, innov):
    """Introduce edges regardless of node layers.

    See discussion for details: https://github.com/google/brain-tokyo-workshop/issues/18
    """

    network = ind.network
    hidden = np.s_[network.offset:, :-network.n_out]

    reachability_matrix = np.copy(network.conn_matrix[hidden])

    np.fill_diagonal(reachability_matrix, True)


    n_paths = 0
    while n_paths < np.sum(reachability_matrix):
        n_paths = np.sum(reachability_matrix)
        reachability_matrix = np.dot(reachability_matrix, reachability_matrix)


    # edges are possible where src can not be reached from dest and there is no
    # direct connection from src to dest
    possible_edges = (network.conn_matrix == False)
    possible_edges[hidden] = np.logical_and(possible_edges[hidden], (reachability_matrix == False).T)

    # Disallow edges from output and to inputs
    #possible_edges[network.offset:, :] = False
    #possible_edges[:, :-network.n_out] = False

    # only the edges that are possible are relevant
    src_options, dest_options = np.where(possible_edges)

    if len(src_options) == 0: return None  # We can't introduce another edge

    # select one of the edes (with equal probabilities)
    i = np.random.randint(len(src_options))
    src, dest = src_options[i],  dest_options[i]

    if src >= network.offset:
        src = network.nodes['id'][src - network.offset]

    if dest >= network.n_hidden:
        dest = network.nodes['id'][dest]
    else:
        dest = network.nodes['id'][dest]

    new_edge = np.zeros(1, dtype=ind.genes.edges.dtype)
    new_edge['src'] = src
    new_edge['dest'] = dest
    new_edge['enabled'] = True
    new_edge['id'] = innov.next_edge_id()

    return ind.genes.__class__(
        edges=np.append(ind.genes.edges, new_edge), nodes=np.copy(ind.genes.nodes),
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def add_edge(ind, env, innov):
    return {
        'layer_based': add_edge_layer_based,
        'layer_agnostic': add_edge_layer_agnostic
    }[env['mutation', 'new_edge', 'strategy']](ind, env, innov)

def reenable_edge(ind, env, innov):
    edges = np.copy(ind.genes.edges)
    nodes = np.copy(ind.genes.nodes)

    # Choose an disabled edge
    options, = np.where(edges['enabled'] == False)
    if len(options) == 0: return None # Make sure one exists

    edge_to_enable = np.random.choice(options)

    edges[edge_to_enable]['enabled'] = True

    return ind.genes.__class__(
        edges=edges, nodes=nodes,
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def change_activation(ind, env, innov):
    nodes = np.copy(ind.genes.nodes)
    edges = np.copy(ind.genes.edges)

    selected_node = np.random.randint(len(nodes))

    # choose one of all but the current act funcs
    new_act = np.random.randint(ind.network.n_act_funcs - 1)
    if new_act >= nodes[selected_node]['func']: new_act += 1

    nodes[selected_node]['func'] = new_act

    return ind.genes.__class__(
        edges=edges, nodes=nodes,
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

# Genetic Operations
def mutation(ind, env, innov):
    supported_mutation_types = np.array([
        ('new_edge', add_edge),
        ('new_node', add_node),
        ('reenable_edge', reenable_edge),
        ('change_activation', change_activation),
    ], dtype=[('name', 'U32'), ('func', object)])

    # get probabilities from params
    probabilities = np.array([
        env['mutation', name, 'propability']
        for name in supported_mutation_types['name']
    ])

    # normalize
    probabilities = probabilities / np.sum(probabilities)

    # Generate permutation of functions (every function occurs only once)

    permutated_mutation_functions = np.random.choice(
        supported_mutation_types, len(supported_mutation_types),
        p=probabilities, replace=False)

    # apply functions until one actually generates a mutation
    for name, func in permutated_mutation_functions:
        new_genes = func(ind, env, innov)
        if new_genes is not None: # mutation was successful
            #logging.debug(f"Mutation via {name}.")
            child = ind.__class__(genes=new_genes, id=innov.next_ind_id(), birth=innov.generation)
            child.parent = ind.id
            #if name == 'new_edge':
            #    logging.debug(f'new edge in {child.birth} {child.id}')
            return child

    logging.warning("No mutation possible.")
    raise RuntimeError("No mutation possible.")

def path_exists(network, src, dest):
    # if a disabled connections exists, there is also another path
    # src, dest are indices for nodes in the network

    if src > dest: # no path possible (otherwise ordering would have to be different)
        return False

    visited = np.zeros(network.n_nodes, dtype=int)
    visited[src] = True
    conn_mat = network.weight_matrix != 0

    n_visited = 0
    while n_visited < sum(visited):
        n_visited = sum(visited)

        visited = np.any(conn_mat[visited], axis=1)
        if visited[dest]:
            return True
    return False
