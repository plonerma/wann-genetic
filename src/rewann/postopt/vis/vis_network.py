import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def node_names(net):
    return (
        [f"$x_{{{i}}}$" for i in range(net.n_in)]          # inputs
        + ["$b$"]                               # bias
        + [f"$h_{{{i}}}$" for i in range(net.n_hidden)]    # hidden
        + [f"$y_{{{i}}}$" for i in range(net.n_out)]       # outputs
    )


def draw_graph(net, ax=None, pos_iterations=None, layer_h=17, labels=None):

    if ax is None:
        ax = plt.gca()

    ax.set_axis_off()

    g = nx.DiGraph()

    # Add nodes
    nodes = node_names(net)

    g.add_nodes_from(nodes)

    # Colors
    color = (
        ['#fdb462'] * net.n_in         # inputs
        + ['#ffed6f']                   # bias
        + ['#80b1d3'] * net.n_hidden   # hidden
        + ['#b3de69'] * net.n_out      # outputs
    )

    layers = list([len(s) for s in net.layers(include_input=True)])

    # pos x will be determined by layer, pos y will be iterated on
    # input, bias and output nodes will have static position, hidden node
    # positions will be determine iteratively
    pos = np.zeros(net.n_nodes, dtype=[('x', float), ('y', float)])

    pos['y'][0: net.offset] = np.arange(net.offset)
    pos['y'][-net.n_out:] = np.arange(net.n_out)

    a = net.weight_matrix[:net.offset, :net.n_hidden]
    b = net.weight_matrix[net.offset:, :net.n_hidden]
    c = net.weight_matrix[net.offset:, net.n_hidden:]

    M = np.vstack([
        a, b+b.T, c.T
    ])

    M_sum = np.sum(M, axis=0)
    M_sum[np.where(M_sum == 0)] = 1

    M = M / M_sum

    if pos_iterations is None:
        pos_iterations = len(layers)

    for i in range(pos_iterations):
        update = np.dot(pos['y'], M)
        pos['y'][net.offset:-net.n_out] = update

    i_0 = 0
    x_0 = 0
    for layer_size in layers:
        bound = np.log(layer_size)
        ns = i_0 + np.arange(layer_size)
        order = np.argsort(pos['y'][ns]) + i_0

        x_n = layer_size // layer_h + 1

        xi = np.mod(np.arange(layer_size), x_n)
        yi = np.arange(layer_size) // x_n

        y_n = min(layer_h, layer_size)

        y_rel = np.linspace(bound, -bound, y_n)
        x_rel = np.linspace(-np.log(x_n), np.log(x_n), x_n)

        x_0 += np.log(x_n) + 1

        pos['y'][order] = y_rel[yi] - x_rel[xi]*0.1
        pos['x'][order] = x_rel[xi] + x_0

        x_0 += np.log(x_n) + 1
        i_0 += layer_size

    if labels == 'func_names':
        node_labels = [''] * (net.n_in + 1) + [
            net.available_act_functions[func][0][:5] for func in net.nodes['func']
        ]
        pos = dict(zip(nodes, np.array([pos['x'], pos['y']]).T))
        nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_color=color, node_size=150)
        nx.draw_networkx_labels(
            g, ax=ax, pos=pos, labels=dict(zip(nodes, node_labels)),
            font_size=8,
        )

    elif labels == 'names':
        pos = dict(zip(nodes, np.array([pos['x'], pos['y']]).T))
        nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_color=color, node_size=150)
        nx.draw_networkx_labels(
            g, ax=ax, pos=pos, labels=dict(zip(nodes, nodes)),
            font_size=8,
        )

    elif labels == 'func_plots':
        # draw circles
        ax.scatter(pos['x'], pos['y'], 150, color, marker='o', zorder=10)

        pos = dict(zip(nodes, np.array([pos['x'], pos['y']]).T))

        for func, n in zip(net.nodes['func'], nodes[net.offset:]):
            func = net.available_act_functions[func][1]

            x = np.linspace(0, 2, 10)
            x = np.hstack([x, 1-x, -x, x-1])
            y = func(x)
            y = y - np.min(y)
            y = y - np.max(y)/2

            verts = np.column_stack([x, y])

            ax.scatter(*pos[n], 50, 'k', marker=verts, zorder=200)
    else:
        pos = dict(zip(nodes, np.array([pos['x'], pos['y']]).T))
        nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_color=color, node_size=150)

    edge_params = dict(
        edge_cmap=plt.get_cmap('tab10'),
        alpha=.6, ax=ax, pos=pos,
        edge_vmin=0, edge_vmax=9,
        arrows=True,
    )

    # draw feed forward edges

    edge_col = list()
    edgelist = list()

    for row, col in zip(*np.where(net.weight_matrix != 0)):
        edgelist.append((nodes[row], nodes[col + net.offset]))
        edge_col.append(
            2 if net.weight_matrix[row][col] > 0
            else 3
        )

    nx.draw_networkx_edges(g, edgelist=edgelist, edge_color=edge_col,
                           width=1, arrowstyle='-',
                           min_source_margin=10, min_target_margin=5,
                           **edge_params)

    if net.is_recurrent:
        # draw recurrent edges

        edge_col = list()
        edgelist = list()

        for row, col in zip(*np.where(net.recurrent_weight_matrix != 0)):
            edgelist.append((nodes[row], nodes[col + net.offset]))
            edge_col.append(
                0 if net.recurrent_weight_matrix[row][col] > 0
                else 1
            )

        nx.draw_networkx_edges(g, edgelist=edgelist, edge_color=edge_col,
                               min_source_margin=30, min_target_margin=20,
                               width=2, **edge_params)


def draw_weight_matrix(net, ax=None):
    x_ticks = list(range(net.n_hidden + net.n_out))
    x_ticklabels = node_names(net)[net.offset:]
    y_ticks = list(range(net.n_nodes - net.n_out))
    y_ticklabels = node_names(net)[:net.n_out]

    if ax is None:
        plt.xticks(x_ticks, x_ticklabels)
        plt.yticks(y_ticks, y_ticklabels)
        imshow = plt.imshow
    else:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)
        imshow = ax.imshow

    imshow(np.max(net.weight_matrix) - net.weight_matrix, cmap='magma')
