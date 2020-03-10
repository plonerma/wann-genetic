import numpy as np

import streamlit as st
import matplotlib.pyplot as plt

import networkx as nx

from rewann.individual import Genotype, Network, Individual


class NetworkCyclicException(Exception):
    """Raised when encountering a which would lead to a cyclic network."""

sample = Individual(genes=Genotype(
    n_in=3,
    n_out=2,
    nodes=[
        # id, type, activation function
        (4, True, 0),
        (5, True, 1),
        (7, False, 2),
        (8, False, 3),
        (10, False, 4),
        (12, False, 5),
        (14, False, 6)
    ],
    edges=[
        # innovation id, src, dest, weight, enabled
        # not necessarily ordered
        ( 3, 12,  7, 1, True),
        ( 1,  1, 12, 1, True),
        ( 0,  0, 12, 1, True),
        ( 2, 12,  4, 1, True),
        ( 4,  7,  4, 1, True),
        (15,  2,  7, 1, True),
        ( 6,  7, 10, 1, True),
        ( 7, 10,  5, 1, True),
        ( 8,  2, 10, 1, False), # disabled
        (11,  3, 10, 1, True),
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

    fig, axs = plt.subplots(2, 1)

    n.draw_graph(axs[0])

    n.draw_weight_matrix(axs[1])
    st.pyplot(fig)

    "### Hidden Node IDs"
    n.params['hidden_nodes']['id']

    "## Propagation"

    x = np.array([
        st.sidebar.slider("x_0", 0., 1.),
        st.sidebar.slider("x_1", 0., 1.),
        st.sidebar.slider("x_2", 0., 1.),
    ])

    full_y = n.apply(x, return_complete=True)

    st.write("Input", x)
    st.write("Output", full_y[-n.n_out:])

    "### Activation Graph"
    n.draw_graph(activation=full_y)
    st.pyplot()
