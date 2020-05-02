import json, toml
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import logging

from .multi_spec import Specification

def assemble_dataframe(spec_path, dir_path, map_from_params=dict()):
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


        if not os.path.isfile(params_path) or not os.path.isfile(stats_path):
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

    if len(data) < len(exp):
        logging.warning(f'Only {len(data)} out of {len(exp)} experiments found.')

    return pd.DataFrame(data=data)


def multivariate_analysis(args):

    df = assemble_dataframe(args.specification, args.experiments_dir,
         map_from_params=dict(
            lower_bound=['sampling', 'lower_bound'],
            upper_bound=['sampling', 'upper_bound'],
         ))

    if len(df) == 0:
        logging.error("No data loaded.")
        return

    df = df.sort_values(by='distribution')

    print ("Visualising uniform distributions")

    df['is_uniform'] = df.agg(lambda x: x['distribution'].startswith('uniform'), axis=1)

    uniform_df = df[df['is_uniform']]

    uniform_df['interval'] = uniform_df.agg('{0[lower_bound]}, {0[upper_bound]}'.format, axis=1)

    uniform_df['lower_bound_type'] = np.sign(uniform_df['lower_bound'])

    uniform_df['interval_size'] = uniform_df['upper_bound'] - uniform_df['lower_bound']

    sns.stripplot(x="interval", y="mean accuracy", data=uniform_df)
    plt.suptitle("Best Mean Accuracy (per individual, mean over sampled weights)")
    plt.show()

    sns.stripplot(x="lower_bound_type", y="mean accuracy", data=uniform_df)
    plt.suptitle("Best Mean Accuracy (per individual, mean over sampled weights)")
    plt.show()

    sns.stripplot(x="interval_size", y="mean accuracy", data=uniform_df)
    plt.suptitle("Best Mean Accuracy (per individual, mean over sampled weights)")
    plt.show()



def compare_experiment_series():
    parser = argparse.ArgumentParser(description='Experiment generation')
    parser.add_argument('specification', type=str)
    parser.add_argument('experiments_dir', type=str)

    multivariate_analysis(parser.parse_args())

if __name__ == '__main__':
    compare_experiment_series()
