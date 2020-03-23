import numpy as np

def add_node(ind, pop, params):
    edges = ind.genes.edges
    nodes = ind.genes.nodes

    # Choose an enabled edge
    options, = np.where(edges['enabled'])
    if len(options) == 0:
        return None

    edge_to_disable = np.random.choice(options)

    # new node with random act func
    new_node = np.zeros(1, dtype=nodes.dtype)
    new_node['id'] = pop.next_node_id()
    new_node['func'] = np.random.randint(ind.network.n_act_funcs)

    new_edges = np.zeros(2, dtype=ind.genes.edges.dtype)
    new_edges['enabled'] = True

    # edge from src to new node
    new_edges[0]['src'] = edges[edge_to_disable]['src']
    new_edges[0]['dest'] = new_node['id']

    # edge from new node to old dest
    new_edges[1]['src'] = new_node['id']
    new_edges[1]['dest'] = edges[edge_to_disable]['dest']

    new_edges['id'] = pop.next_edge_id(), pop.next_edge_id()

    edges = np.append(edges, new_edges)  # creates copy of old edges as well

    # disable edge
    edges[edge_to_disable]['enabled'] = False

    return ind.genes.__class__(
        edges=edges, nodes=np.append(ind.genes.nodes, new_node),
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def add_edge_layer_based(ind, pop, params):
    """Layer-based edge introduction strategy.

    The implementation of this method is equivalent to strategy used in the
    original implementation.
    """

def add_edge_layer_agnostic(ind, pop, params):
    """Introduce edges regardless of node layers.

    See discussion for details: https://github.com/google/brain-tokyo-workshop/issues/18
    """

    genes = ind.genes
    network = ind.network

    connectivity_matrix = np.zeros((network.n_nodes, network.n_nodes), dtype=bool)
    connectivity_matrix[:-network.n_out, network.offset:] = network.weight_matrix != 0

    np.fill_diagonal(connectivity_matrix, True)
    reachability_matrix = np.copy(connectivity_matrix)

    n_paths = 0
    while n_paths < np.sum(reachability_matrix):
        n_paths = np.sum(reachability_matrix)
        reachability_matrix = np.dot(connectivity_matrix, reachability_matrix)


    # edges are possible where src can not be reached from dest and there is no
    # direct connection from src to dest (* can be used as logical and)
    possible_edges = (reachability_matrix == False).T * (connectivity_matrix == False)

    # Disallow edges from output and to inputs
    possible_edges[network.offset:, :] = False
    possible_edges[:, :-network.n_out] = False

    # only the edges that are possible are relevant
    src_options, dest_options = np.where(possible_edges)

    if len(src_options) == 0:  # We can't introduce another edge
        return None

    # select one of the edes (with equal probabilities)
    i = np.random.randint(len(src_options))
    src, dest = src_options[i],  dest_options[i]

    new_edge = np.zeros(1, dtype=genes.edges.dtype)
    new_edge['src'] = src if src < network.offset else network.nodes['id'][src - network.offset]
    new_edge['dest'] = network.nodes['id'][dest - network.offset]
    new_edge['enabled'] = True
    new_edge['id'] = pop.next_edge_id()

    return ind.genes.__class__(
        edges=np.append(genes.edges, new_edge), nodes=np.copy(genes.nodes),
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def add_edge(ind, pop, params):
    return {
        'layer_based': add_edge_layer_based,
        'layer_agnostic': add_edge_layer_agnostic
    }[params['mutation', 'new_edge', 'strategy']](ind, pop, params)

def reenable_edge(ind, pop, params):
    edges = np.copy(ind.genes.edges)

    # Choose an disabled edge
    options, = np.where(edges['enabled'] == False)
    if len(options) == 0:
        return None

    edge_to_enable = np.random.choice(options)

    edges[edge_to_enable]['enabled'] = True

    return ind.genes.__class__(
        edges=edges, nodes=np.copy(ind.genes.nodes),
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def change_activation(ind, pop, params):
    nodes = np.copy(ind.genes.nodes)

    selected_node = np.random.randint(len(nodes))

    new_act = np.random.randint(ind.network.n_act_funcs - 1)
    if new_act >= nodes[selected_node]['func']:
        new_act += 1

    nodes[selected_node]['func'] = new_act

    return ind.genes.__class__(
        edges=np.copy(ind.genes.edges), nodes=nodes,
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

# Genetic Operations
def mutation(ind, pop, params):
    supported_mutation_types = np.array([
        ('new_edge', add_edge),
        ('new_node', add_node),
        ('reenable_edge', reenable_edge),
        ('change_activation', change_activation),
    ], dtype=[('name', 'U32'), ('func', object)])

    # get probabilities from params
    probabilities = np.array([
        params['mutation', name, 'propability']
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
        new_genes = func(ind, pop, params)
        if new_genes is not None: # mutation was successful
            params.log.debug(f"Mutation via {name}.")
            return ind.__class__(genes=new_genes)

    params.log.warning("No mutation possible.")
    raise RuntimeError("No mutation possible.")

def path_exists(network, src, dest):
    # TODO: test this!!!
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


def crossover(ind_a, ind_b, params):
    # just do mutation for now
    params.log.warning("Cross-over is not supported yet.")
    return ind_a.mutation(params)
