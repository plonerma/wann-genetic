import sys, os
import numpy as np
import streamlit as st
import altair as alt


import logging

from pathlib import Path

from rewann import Environment

import pandas as pd

args = sys.argv[1:]


def load_env(path):
    logging.info(f"Loading env in @'{path}'")
    return Environment(path)


def multiline_chart(df, xs, limit=1):
    st.altair_chart(alt.Chart(df).mark_line().encode(x='index', y=xs))


if len(args) > 0:
    path, *_ = args

elif len(args) == 0:
    paths = filter(lambda p: not str(p).endswith('.gitignore'),
                   Path('data').iterdir())

    paths = sorted(paths, key=os.path.getmtime, reverse=True)
    path = st.sidebar.selectbox('Path', options=paths)


env = load_env(path)

exp_view = st.sidebar.selectbox('Experiment', options=['metrics', 'params', 'log'])

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


    st.line_chart(data=metrics[['MEDIAN:median:accuracy', 'MEAN:mean:accuracy', 'MAX:max:accuracy', 'MIN:min:accuracy']])
    st.line_chart(data=metrics[['MEDIAN:median:kappa', 'MEAN:mean:kappa', 'MAX:max:kappa', 'MIN:min:kappa']])
    st.line_chart(data=metrics[['MEDIAN:n_hidden', 'MEAN:n_hidden', 'MAX:n_hidden', 'MIN:n_hidden']])
    st.line_chart(data=metrics[['MEDIAN:n_edges', 'MEAN:n_edges', 'MAX:n_edges', 'MIN:n_edges']])
    st.line_chart(data=metrics[['MEDIAN:age', 'MEAN:age', 'MAX:age', 'MIN:age']])

    st.line_chart(data=metrics[['num_no_edge_inds', 'num_no_hidden_inds', 'biggest_ind']])

    st.line_chart(data=metrics[['num_unique_individuals', 'num_individuals']])

    st.line_chart(data=metrics)
