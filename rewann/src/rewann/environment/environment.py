import toml
import logging
import numpy as np
import pandas as pd
import os

from .tasks import select_task

from .evolution import evolution

class Environment:
    from ..individual import Individual as ind_class
    from .fs_util import (default_params, setup_params, setup_logging,
                       dump_pop, dump_metrics, load_pop, load_metrics)

    def __init__(self, params):
        self.setup_params(params)

        # choose task
        task_name = self['task', 'name']
        self.task = select_task(task_name)

        # if this is an experiment to be run, setup logger etc.
        if not 'is_report' in self or not self['is_report']:
            self.setup_logging()

            p = os.path.abspath(self['experiment_path'])
            self.log.info(f'Saving data at {p}.')

            n_samples = len(self.task.y_true)
            self.log.debug(f'{n_samples} samples in training data set.')

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
            self.log.debug(f'Sampled weight {w}')

            gen, pop = next(generations)

            self.log.debug(f'Completed generation {gen}')

            gen_metrics = self.generation_metrics(gen=gen, population=pop)

            self.log.info(f"#{gen} mean, max kappa: {gen_metrics['MEAN:mean:kappa']:.2}, {gen_metrics['MAX:max:kappa']:.2}")

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
        if isinstance(keys, tuple):
            assert len(keys) > 0
            d = self.params
            for k in keys:
                d = d[k]
            return d
        else:
            return self.params[keys]

    def __setitem__(self, keys, value):
        if isinstance(keys, tuple):
            assert len(keys) > 0
            d = self.params
            for k in keys[:-1]:
                d[k] = d.get(k, dict())
                d = d[k]
            d[keys[-1]] = value
        else:
            self.params[keys] = value

    def __contains__(self, keys):
        if isinstance(keys, tuple):
            assert len(keys) > 0
            d = self.params
            for k in keys:
                try:
                    d = d[k]
                except:
                    return False
            return True
        else:
            return keys in self.params
