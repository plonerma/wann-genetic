import json, toml
import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind

from itertools import product

import logging

from .multi_spec import Specification

def assemble_dataframe(spec_path, dir_path, map_from_params=dict()):
    assert os.path.isdir(dir_path)
    _, potential_results, _ = next(os.walk(dir_path))

    spec = Specification(spec_path)

    exp = dict()

    for name in spec.generate_experiments(names_only=True):
        name = name.lower()
        exp[name] = dict(_loaded=False)
        exp[name].update(spec.name_parts)

    for dir in potential_results:
        dir = os.path.join(dir_path, dir)

        params_path = os.path.join(dir, 'params.toml')
        stats_path = os.path.join(dir, 'report', 'stats.json')


        if not os.path.exists(params_path) or not os.path.exists(stats_path):
            logging.info(f'Non-experiment directory: {dir}')
            continue

        params = toml.load(params_path)

        exp_name = params['experiment_name'].lower()

        if exp_name not in exp:
            logging.info(f'Found non-matching experiment: {exp_name}')
            continue

        if exp[exp_name]['_loaded']:
            logging.warning(f'Duplicate experiment {exp_name}. Skipping.')

        with open(stats_path, 'r') as f:
            stats = json.load(f)

        exp[exp_name].update(stats)

        for target, source in map_from_params.items():
            try:
                if isinstance(source, str):
                    v = params[source]
                else:
                    v = params
                    for k in source:
                        v = v[k]
                exp[exp_name][target] = v
            except KeyError:
                exp[exp_name][target] = np.nan

        exp[exp_name]['_loaded'] = True


    # ensure expected experiments were loaded
    data = list()
    for k, v in exp.items():
        if not v.pop('_loaded'):
            logging.info(f'Experiment {k} was not found')
        else:
            data.append(v)

    df =  pd.DataFrame(data=data)


    if len(df) == 0:
        logging.error("No data loaded.")
    elif len(data) < len(exp):
        logging.warning(f'Only {len(data)} out of {len(exp)} experiments found.')

    return df



def load_experiment_series(path, params_map=dict(), sort_by=False):
    spec_path = os.path.join(path, 'spec.toml')
    data_path = os.path.join(path, 'data')
    df = assemble_dataframe(spec_path, data_path, map_from_params=params_map)

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
