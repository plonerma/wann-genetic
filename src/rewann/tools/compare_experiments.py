import json, toml
import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind

from itertools import product

import logging

from .experiment_series import ExperimentSeries

def load_series_stats(spec_path, dir_path=None):
    spec = ExperimentSeries.from_spec_file(spec_path)
    spec.discover_data_dir(dir_path)
    return spec.assemble_stats()





def load_experiment_series(spec_path, params_map=dict(), sort_by=False, data_path=None):
    if os.path.isdir(spec_path):
        spec_path = os.path.join(spec_path, 'spec.toml')

    if data_path is None:
        dir_path = os.path.dirname(os.path.realpath(spec_path))
        data_path = os.path.join(dir_path, 'data')

    df = assemble_dataframe(spec_path, data_path, params_map=params_map)

    if sort_by is not False:
        df = df.sort_values(by=sort_by)
    return df

def mean_comparison(df, group_var, group_values, measure='mean accuracy'):
    for v1, v2 in product(group_values, repeat=2):
        if v1 == v2: continue
        g1 = df[df[group_var] == v1][measure]
        g2 = df[df[group_var] == v2][measure]

        # greater-than test
        t, p = ttest_ind(g1, g2, equal_var=False)

        if t > 0 and p < 0.05:
            print(f"{group_var}={v1} is significantly ({p:.1%}) better than {group_var}={v2}")
            print (t, p)
