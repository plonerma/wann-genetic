import toml
import logging
import numpy as np
import pandas as pd
import os
import subprocess
import logging

from .tasks import select_task
from .evolution import evolution

class Environment:
    from ..individual import Individual as ind_class
    from .fs_util import (default_params, setup_params,
                       dump_pop, dump_metrics, load_pop, load_metrics)

    def __init__(self, params):
        self.setup_params(params)

        # choose task
        self.task = select_task(self['task', 'name'])

        # if this is an experiment to be run, setup logger etc.
        if not 'is_report' in self or not self['is_report']:

            log_path = env_path(env, env['storage', 'log_filename'])
            logging.info (f"Check log ('{log_path}') for details.")

            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)

            if not env['debug']:
                logger.setLevel(logging.INFO)

            git_label = subprocess.check_output(["git", "describe", "--always"]).strip()
            git_label = git_label.decode('utf-8')

            logging.info(f"Current commit {git_label}")

            p = os.path.abspath(self['experiment_path'])
            logging.info(f'Saving data at {p}.')

            n_samples = len(self.task.y_true)
            logging.debug(f'{n_samples} samples in training data set.')

            # log used parameters
            params_toml = toml.dumps(self.params)
            logging.debug(f"Running experiments with the following parameters:\n{params_toml}")

            with open(env_path(env, 'params.toml'), 'w') as f:
                params = dict(env.params)
                params['is_report'] = True # mark stored params as part of a report
                toml.dump(params, f)

    def sample_weight(self):
        dist = self['sampling', 'distribution'].lower()
        if dist == 'one':
            w = 1
        elif dist == 'uniform':
            l = self['sampling', 'lower_bound']
            u = self['sampling', 'upper_bound']
            assert l is not None and u is not None

            w = np.random.uniform(l, u, size=self['sampling']['number_of_samples_per_iteration'])

        self['sampling', 'current_weight'] = w
        return w

    def run(self):
        np.random.seed(self['sampling', 'seed'])

        n = self['population', 'num_generations']
        generations = evolution(self)

        metrics = list()

        for _ in range(n):
            w = self.sample_weight()
            logging.debug(f'Sampled weight {w}')

            gen, pop = next(generations)

            logging.debug(f'Completed generation {gen}')

            gen_metrics = self.generation_metrics(gen=gen, population=pop)

            logging.info(f"#{gen} mean, max kappa: {gen_metrics['MEAN:mean:kappa']:.2}, {gen_metrics['MAX:max:kappa']:.2}")

            metrics.append(gen_metrics)

            if gen % self['storage', 'commit_population_freq'] == 0:
                self.dump_pop(gen, pop)
                self.dump_metrics(pd.DataFrame(data=metrics))

        self.dump_metrics(pd.DataFrame(data=metrics))
        self.last_population = pop

    def generation_metrics(self, population, gen=None):
        if gen is None:
            gen = self['population', 'num_generations']

        metric_names = ('n_hidden', 'n_edges', 'n_evaluations', 'age',
                        'mean:kappa', 'min:kappa', 'max:kappa', 'median:kappa',
                        'mean:accuracy', 'min:accuracy', 'max:accuracy', 'median:accuracy')
        df = pd.DataFrame(data=[
            ind.metrics(*metric_names, current_gen=gen) for ind in population
        ])

        metrics = dict(
            num_unique_individuals=len(set(population)),

            num_individuals=len(population),

            # number of inds without edges
            num_no_edge_inds=np.sum(df['n_edges'] == 0),

            # number of inds without hidden nodes
            num_no_hidden_inds=np.sum(df['n_hidden'] == 0),

            # individual with the most occurences
            biggest_ind=max([population.count(i) for i in set(population)]),
            covariance_mean_kappa_n_edges=df['n_edges'].cov(df['mean:kappa'])
        )

        for name, values in df.items():
            metrics[f'MAX:{name}'] = values.max()
            metrics[f'MIN:{name}'] = values.min()
            metrics[f'MEAN:{name}'] = values.mean()
            metrics[f'MEDIAN:{name}'] = values.median()

        return metrics

    # magic methods for direct access of parameters
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = [keys]

        d = self.params
        for k in keys: d = d[k]
        return d

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = [keys]

        *first_keys, last_key = keys
        d = self.params
        for k in first_keys:
            d[k] = d.get(k, dict())
            d = d[k]
        d[last_key] = value

    def __contains__(self, keys):
        if not isinstance(keys, tuple):
            keys = [keys]

        d = self.params

        for k in keys:
            try: d = d[k]
            except: return False
        return True
