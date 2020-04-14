import sys, os
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt


import logging

from pathlib import Path

from rewann import Environment
from rewann.environment.evolution import evaluate_inds

from rewann.vis import draw_graph, draw_weight_matrix, node_names

import pandas as pd

args = sys.argv[1:]

@st.cache(allow_output_mutation=True)
def load_env(path):
    logging.info(f"Loading env in @'{path}'")
    return Environment(path)

if len(args) > 0:
    path, *_ = args

elif len(args) == 0:
    paths = filter(lambda p: not str(p).endswith('.gitignore'),
                   Path('data').iterdir())

    paths = sorted(paths, key=os.path.getmtime, reverse=True)
    path = st.sidebar.selectbox('Path', options=paths)


env = load_env(path)

exp_view = st.sidebar.selectbox('Experiment', options=['metrics', 'params', 'log', 'population'])

if exp_view == 'log':
    "# Log"
    with open(os.path.join(path, 'execution.log'), 'r') as f:
        for l in f:
            st.write(l)

elif exp_view == 'params':
    "# Parameters"

    st.write(env.params)

elif exp_view == 'metrics':
    "# Generations"

    metrics = env.load_metrics()
    metrics.index.names = ['generation']

    options = list(sorted(metrics.columns))

    default = ['MEDIAN:kappa.median', 'MEAN:kappa.mean', 'MAX:kappa.max', 'MEDIAN:kappa.min']

    default = list(filter(lambda o: o in options, default))

    selection = st.multiselect('metrics', options=options, default=default)

    st.line_chart(data=metrics[selection])

    always_show = [
        ['MEDIAN:accuracy.median', 'MEAN:accuracy.mean', 'MAX:accuracy.max', 'MEDIAN:accuracy.min', 'MAX:accuracy.mean'],
        ['MEDIAN:log_loss.median', 'MEAN:log_loss.mean', 'MIN:log_loss.min', 'MEDIAN:log_loss.min', 'MIN:log_loss.mean'],
        ['num_no_edge_inds', 'num_no_hidden_inds', 'biggest_ind'],
        ['num_unique_individuals', 'num_individuals']
    ]

    for s in always_show:
        st.line_chart(data=metrics[s])

elif exp_view == 'population':
    pops = list(reversed(env.existing_populations()))
    gen = st.selectbox('generation', options=pops)
    pop = env.load_pop(gen)

    individuals = {f'{i}: {ind.id}': ind for i, ind in enumerate(pop)}
    i = st.selectbox('individual', options=list(individuals.keys()))

    ind = individuals[i]

    n_weights = st.slider('number of weights', 1, 1000, 100)
    n_samples = st.slider('number of samples', 1, len(env.task.x), 100)

    env.sample_weights(n_weights)
    evaluate_inds(env, [ind], n_samples=n_samples)

    ind_metrics = ind.metric_values

    ind_metrics = ind_metrics.sort_values(by=['weight'])

    fig, ax = plt.subplots()
    ind_metrics.plot(kind='line',x='weight',y='log_loss', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ind_metrics.plot(kind='line',x='weight',y='kappa', ax=ax)
    ind_metrics.plot(kind='line',x='weight',y='accuracy', ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ind_metrics[['weight']].plot(kind='hist', ax=ax)
    st.pyplot(fig)

    n = ind.network

    fig, ax = plt.subplots()
    draw_graph(n, ax)
    st.pyplot(fig)
