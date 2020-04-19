import numpy as np

import streamlit as st
import matplotlib.pyplot as plt

import networkx as nx

import pandas as pd

from rewann.individual import Individual
from rewann.vis import draw_graph, draw_weight_matrix, node_names


class NetworkCyclicException(Exception):
    """Raised when encountering a which would lead to a cyclic network."""

sample = Individual(genes=Individual.Genotype(
    n_in=3,
    n_out=2,
    nodes=[
        # id, type, activation function
        (4, True, 0),
        (5, True, 1),
        (7, False, 2),
        (10, False, 4),
        (12, False, 5),
    ],
    edges=[
        # innovation id, src, dest, weight, enabled
        # not necessarily ordered
        ( 3, 12,  7, True),
        ( 1,  1, 12, True),
        ( 0,  0, 12, True),
        ( 2, 12,  4, True),
        ( 4,  7,  4, True),
        (15,  2,  7, True),
        ( 6,  7, 10, True),
        ( 7, 10,  5, True),
        ( 8,  2, 10, False), # disabled
        (11,  3, 10, True),
    ]
))

assert sample.genes is not None


if st.sidebar.selectbox("View", ["Genes", "Network"]) == "Genes":

    st.write(f"""
        # Genes
        {sample.genes.n_in} input nodes, {sample.genes.n_out} output nodes

        ## Nodes
        """,

        sample.genes.nodes,

        "## Edges",

        sample.genes.edges
    )

else:
    "# Network"

    sample.express()

    assert sample.network is not None

    n = sample.network

    fig, axs = plt.subplots(1, 2)

    draw_graph(n, axs[0])

    draw_weight_matrix(n, axs[1])
    st.pyplot(fig)

    "### Nodes"

    st.write(pd.DataFrame({
        'node': [node_names(n)[i].strip('$') for i in range(n.offset, n.n_nodes)],
        'gene id': n.nodes['id'],
        'activation function': n.activation_functions(np.arange(0, n.n_nodes - n.offset))
    }))

    "## Propagation"

    x = np.array([
        st.sidebar.slider("x_0", 0., 1.),
        st.sidebar.slider("x_1", 0., 1.),
        st.sidebar.slider("x_2", 0., 1.),
    ])

    y, activation = n.apply(x, func='softmax', return_activation=True)

    st.write("Input", x)
    st.write("Output", y)

    "### Activation Graph"
    draw_graph(n, activation=activation)
    st.pyplot()
