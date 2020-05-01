import json, toml
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import logging

from .multi_spec import Specification

def assemble_dataframe(spec_path, dir_path):
    _, potential_results, _ = next(os.walk(dir_path))

    spec = Specification(spec_path)

    exp = dict()

    for name in spec.generate_experiments(names_only=True):
        exp[name] = dict(_loaded=False)
        exp[name].update(spec.name_parts)

    for dir in potential_results:
        dir = os.path.join(dir_path, dir)

        params_path = os.path.join(dir, 'params.toml')
        stats_path = os.path.join(dir, 'report', 'stats.json')


        if not os.path.isfile(params_path) or not os.path.isfile(stats_path):
            continue

        params = toml.load(params_path)

        exp_name = params['experiment_name']

        if exp_name not in exp:
            logging.info(f'Found non-matching experiment: {exp_name}')
            continue

        with open(stats_path, 'r') as f:
            stats = json.load(f)

        exp[exp_name].update(stats)
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


def multivariate_analysis(df):
    if len(df) == 0:
        logging.error("No data loaded.")
        return

    df = df.sort_values(by='interval')

    sns.stripplot(x="interval", y="mean accuracy", data=df)
    plt.suptitle("Best Mean Accuracy (per individual, mean over sampled weights)")
    plt.show()

    sns.stripplot(x="interval", y="max accuracy", data=df)
    plt.suptitle("Best Max Accuracy (per individual, max over sampled weights)")
    plt.show()


def compare_experiment_series():
    parser = argparse.ArgumentParser(description='Experiment generation')
    parser.add_argument('specification', type=str)
    parser.add_argument('experiments_dir', type=str)
    args = parser.parse_args()

    df = assemble_dataframe(args.specification, args.experiments_dir)
    multivariate_analysis(df)

if __name__ == '__main__':
    compare_experiment_series()
