import sys, os
import numpy as np
import streamlit as st

from rewann import Environment

import pandas as pd

args = sys.argv[1:]

def load_env(path):
    return Environment(path)



if len(args) > 0:
    path, *_ = args

elif len(args) == 0:
    base_path, dirs, _ = next(os.walk('data'))
    dirs = sorted(dirs)
    path = os.path.join(base_path, st.sidebar.selectbox('Path', options=dirs, index=(len(dirs)-1)))


env = load_env(path)

exp_view = st.sidebar.selectbox('Experiment', options=['log', 'params', 'metrics'])

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
    st.line_chart(data=metrics[['MEAN:mean:accuracy', 'MAX:max:accuracy', 'MIN:min:accuracy']])
    st.line_chart(data=metrics[['MEAN:mean:kappa', 'MAX:max:kappa', 'MIN:min:kappa']])
    st.line_chart(data=metrics[['MEAN:n_hidden', 'MAX:n_hidden', 'MIN:n_hidden']])
    st.line_chart(data=metrics[['MEAN:n_edges', 'MAX:n_edges', 'MIN:n_edges']])

    st.line_chart(data=metrics[['num_unique_individuals', 'num_individuals']])

    st.line_chart(data=metrics)
