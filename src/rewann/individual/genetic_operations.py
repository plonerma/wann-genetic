import numpy as np
import logging

from .expression import get_array_field

def add_node(ind, env, innov):
    # Choose an enabled edge

    enabled = get_array_field(ind.genes.edges, 'enabled', True)
    recurrent = get_array_field(ind.genes.edges, 'recurrent', False)


    options, = np.where(enabled & ~recurrent)
    if len(options) == 0: return None

    edge_to_disable = np.random.choice(options)

    # new node with random act func
    new_node = np.zeros(1, dtype=ind.genes.nodes.dtype)
    new_node['id'] = innov.next_node_id()
    new_node['func'] = np.random.randint(ind.network.n_act_funcs)

    new_edges = np.zeros(2, dtype=ind.genes.edges.dtype)

    new_edges['enabled'] = True

    # first edge will have same sign as original edge, the second one will be 1
    # (so the total factor will stay the same if act func is linear)
    new_edges[0]['sign'] = ind.genes.edges[edge_to_disable]['sign']
    new_edges[1]['sign'] = 1

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

def new_edge(env, innov, ind, src, dest, recurrent=False):
    e = np.zeros(1, dtype=ind.genes.edges.dtype)
    e['src'] = src
    e['dest'] = dest
    e['enabled'] = True
    if env['population']['enable_edge_signs']:
        e['sign'] = np.random.choice([-1,1])
    else:
        e['sign'] = 1

    if env.task.is_recurrent:
        e['recurrent'] = recurrent

    e['id'] = innov.next_edge_id()
    return e

def add_edge_layer_based(ind, env, innov):
    """Layer-based edge introduction strategy.

    The implementation of this method is equivalent to strategy used in the
    original implementation.
    """

    # Source: https://github.com/google/brain-tokyo-workshop/blob/73eb4531746825203a3c591896a79ac563d393e7/WANNRelease/prettyNeatWann/neat_src/ind.py#L293

    # To avoid recurrent connections nodes are sorted into layers, and
    # connections are only allowed from lower to higher layers

    layers = ind.network.node_layers()

    src_options = np.arange(ind.network.offset + ind.network.n_hidden)
    np.random.shuffle(src_options)

    for src in src_options:
        src_layer = layers[src]

        dest_options = NotImplementedError()

        # source: https://stackoverflow.com/a/16244044
        first_possible_dest = np.argmax(layers > src_layer)

        dest_options = np.arange(first_possible_dest, ind.network.n_nodes)
        np.random.shuffle(dest_options)

        for dest in dest_options:
            if ind.network.conn_matrix[src, dest - ind.network.offset]:
                continue # connection already exists

            if src >= ind.network.offset:
                src = ind.network.nodes['id'][src - ind.network.offset]

            dest = ind.network.nodes['id'][dest - ind.network.offset]


            return ind.genes.__class__(
                edges=np.append(ind.genes.edges, new_edge(env, innov, ind, src, dest)),
                nodes=ind.genes.nodes,
                n_in=ind.genes.n_in, n_out=ind.genes.n_out
            )


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

    dest = network.nodes['id'][dest]

    return ind.genes.__class__(
        edges=np.append(ind.genes.edges, new_edge(env, innov, ind, src, dest)),
        nodes=ind.genes.nodes,
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def add_edge(ind, env, innov):
    return {
        'layer_based': add_edge_layer_based,
        'layer_agnostic': add_edge_layer_agnostic
    }[env['mutation', 'new_edge', 'strategy']](ind, env, innov)

def reenable_edge(ind, env, innov):
    edges = np.copy(ind.genes.edges)

    # Choose an disabled edge
    options, = np.where(edges['enabled'] == False)
    if len(options) == 0: return None # Make sure one exists

    edge_to_enable = np.random.choice(options)

    edges[edge_to_enable]['enabled'] = True

    return ind.genes.__class__(
        edges=edges, nodes=ind.genes.nodes,
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def change_activation(ind, env, innov):
    nodes = np.copy(ind.genes.nodes)

    selected_node = np.random.randint(len(nodes))

    # choose one of all but the current act funcs
    new_act = np.random.randint(ind.network.n_act_funcs - 1)
    if new_act >= nodes[selected_node]['func']: new_act += 1

    nodes[selected_node]['func'] = new_act

    return ind.genes.__class__(
        edges=ind.genes.edges, nodes=nodes,
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def change_edge_sign(ind, env, innov):
    edges = np.copy(ind.genes.edges)
    # Choose an enabled edge
    options, = np.where(ind.genes.edges['sign'].astype(bool))
    if len(options) == 0: return None

    edge_to_change = np.random.choice(options)

    edges['sign'][edge_to_change] = -1

    return ind.genes.__class__(
        edges=edges, nodes=ind.genes.nodes,
        n_in=ind.genes.n_in, n_out=ind.genes.n_out
    )

def add_recurrent_edge(ind, env, innov):
    network = ind.network
    src_options, dest_options = np.where(network.recurrent_weight_matrix == 0)

    if len(src_options) == 0: return None  # We can't introduce another edge

    i_options = np.arange(len(src_options))
    np.random.shuffle(i_options)

    for i in i_options:
        src = src_options[i]
        dest = dest_options[i]

        if src >= network.offset:
            src = network.nodes['id'][src - network.offset]
        dest = network.nodes['id'][dest]

        e = ind.genes.edges
        matches = e['recurrent'] & (e['src'] == src) & (e['dest'] == dest)
        if np.any(matches): continue

        return ind.genes.__class__(
            edges=np.append(ind.genes.edges, new_edge(env, innov, ind, src, dest, recurrent=True)),
            nodes=ind.genes.nodes,
            n_in=ind.genes.n_in, n_out=ind.genes.n_out
        )


# Genetic Operations
def mutation(ind, env, innov):
    supported_mutation_types = np.array([
        ('new_edge', add_edge),
        ('new_node', add_node),
        ('reenable_edge', reenable_edge),
        ('change_activation', change_activation),
        ('change_edge_sign', change_edge_sign), # might be disabled
        ('add_recurrent_edge', add_recurrent_edge), # might be disabled
    ], dtype=[('name', 'U32'), ('func', object)])

    # get probabilities from params
    probabilities = np.array([
        env['mutation', name, 'probability']
        for name in supported_mutation_types['name']
    ])

    num_choices = len(supported_mutation_types)

    if not env['population']['enable_edge_signs']:
        probabilities[-2] = 0
        num_choices -= 1

    if not env.task.is_recurrent:
        probabilities[-1] = 0
        num_choices -= 1

    # normalize
    probabilities = probabilities / np.sum(probabilities)

    # Generate permutation of functions (every function occurs only once)

    permutated_mutation_functions = np.random.choice(
        supported_mutation_types, num_choices,
        p=probabilities, replace=False)

    # apply functions until one actually generates a mutation
    for name, func in permutated_mutation_functions:
        new_genes = func(ind, env, innov)
        if new_genes is not None: # mutation was successful

            child = ind.__class__(
                genes=new_genes, id=innov.next_ind_id(),
                birth=innov.generation,
                parent=ind.id, mutations=ind.mutations+1)

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
