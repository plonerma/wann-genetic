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


def draw_graph(net, ax=None, pos_iterations=None, layer_h=17):

    if ax is None:
        ax = plt.gca()


    ax.set_axis_off()

    g = nx.DiGraph()

    # Add nodes
    nodes = node_names(net)
    g.add_nodes_from(nodes)

    # Colors
    color=(
        ['#fdb462'] * net.n_in         # inputs
        + ['#ffed6f']                   # bias
        + ['#80b1d3'] * net.n_hidden   # hidden
        + ['#b3de69'] * net.n_out      # outputs
    )

    # Positions
    N = len(nodes)

    layers = list([len(s) for s in net.layers(including_input=True)])

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

    M_sum = np.sum(M,axis=0)
    M_sum[np.where(M_sum == 0)] = 1

    M = M / M_sum

    if pos_iterations is None:
        pos_iterations = len(layers)

    for i in range(pos_iterations):
        update = np.dot(pos['y'], M)
        pos['y'][net.offset:-net.n_out] = update

    i_0 = 0
    x_0 = 0
    for l in layers:
        bound = np.log(l)
        ns = i_0 + np.arange(l)
        order = np.argsort(pos['y'][ns]) + i_0


        x_n = l // layer_h + 1

        xi = np.mod(np.arange(l), x_n)
        yi = np.arange(l) // x_n

        y_n = min(layer_h, l)

        y_rel = np.linspace(bound, -bound, y_n)
        x_rel = np.linspace(-np.log(x_n), np.log(x_n), x_n)

        x_0 += np.log(x_n) + 1

        pos['y'][order] = y_rel[yi] - x_rel[xi]*0.1
        pos['x'][order] = x_rel[xi] + x_0

        x_0 += np.log(x_n) + 1
        i_0 += l

    pos = dict(zip(nodes, np.array([pos['x'], pos['y']]).T))

    edge_params=dict(
        edge_cmap = plt.get_cmap('tab10'),
        alpha=.6,
        ax=ax, pos=pos,
        edge_vmin=0,
        edge_vmax=9
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
        width=1, arrows=False, **edge_params)

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
                           width=2,  arrows=True, **edge_params)

    nx.draw_networkx_nodes(
        g, ax=ax, pos=pos, node_color=color,
        with_labels=False, node_size=50, linewidths=0)

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
