import streamlit as st
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


def draw_graph(net, ax=None, activation=None, pos_iterations=None):
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

    layers = list(enumerate([len(s) for s in net.layers(including_input=True)]))

    # pos x will be determined by layer, pos y will be iterated on
    # input, bias and output nodes will have static position, hidden node
    # positions will be determine iteratively
    pos = np.array([
        ((li), 0)
        for li, ln in layers
        for ni in range(ln)
    ], dtype=[('x', float), ('y', float)])

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
    for j, l in layers:
        bound = np.log(l)
        ns = i_0 + np.arange(l)
        pos['y'][np.argsort(pos['y'][ns]) + i_0] = np.linspace(-bound, bound, l)
        i_0 += l

    pos = dict(zip(nodes, np.array([pos['x'], pos['y']]).T))

    if activation is not None:
        node_size = 600 * activation / np.max(activation)
    else:
        node_size = 300

    # Edges
    for row, col in zip(*np.where(net.weight_matrix != 0)):
        ns = nodes[row], nodes[col + net.offset]
        w = net.weight_matrix[row, col]
        a = 1 if activation is None else activation[row]
        g.add_edge(*ns, weight=w, activation=a)

    edge_weights = np.array([g[u][v]['weight'] for u,v in g.edges()])
    edge_activ = np.array([g[u][v]['activation'] for u,v in g.edges()])

    edge_color = -edge_weights*edge_activ
    edge_widths = edge_weights

    if len(edge_color) == 0:
        edge_color = [0]

    nx.draw(
        g, ax=ax, pos=pos, node_color=color,
        with_labels=True, node_size=node_size, font_color="k", font_size=9,
        arrows=True, linewidths=0, alpha=.9, arrowstyle='-|>', style='dotted',
        edge_color=edge_color, vmax=0, vmin=np.min(edge_color), cmap='magma',
        width=edge_widths,
        label="CAPTION")

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
