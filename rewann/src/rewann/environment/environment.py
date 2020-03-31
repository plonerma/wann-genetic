import toml
import logging
import numpy as np
import pandas as pd
import os

from .tasks import select_task

from .evolution import evolution

class Environment:
    from ..individual import Individual
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

    def sample_weight(self):
        dist = self['sampling', 'distribution'].lower()
        if dist == 'one':
            w = 1
        elif dist == 'uniform':
            l = self['sampling', 'lower_bound']
            u = self['sampling', 'upper_bound']
            assert l is not None and u is not None

            w = np.random.uniform(l, u)

        self['sampling', 'current_weight'] = w
        return w

    def run(self):
        np.random.seed(self['sampling', 'seed'])

        n = self['population', 'num_generations']
        generations = evolution(self)

        metrics = list()

        for gen in range(n):
            w = self.sample_weight()
            self.log.debug(f'Sampled weight {w}')

            pop = next(generations)

            self.log.debug(f'Completed generation {gen}')

            gen_metrics = self.generation_metrics(pop)
            self.log.info(f"#{gen} mean, max kappa: {gen_metrics['MEAN:mean:kappa']:.2}, {gen_metrics['MAX:max:kappa']:.2}")

            metrics.append(gen_metrics)

            if gen % self['storage', 'commit_population_freq'] == 0:
                self.dump_pop(gen, pop)
                self.dump_metrics(pd.DataFrame(data=metrics))

        self.dump_metrics(pd.DataFrame(data=metrics))
        self.last_population = pop

    def generation_metrics(self, population):

        metric_names = ('n_hidden', 'n_edges', 'n_evaluations',
                        'mean:kappa', 'min:kappa', 'max:kappa',
                        'mean:accuracy', 'min:accuracy', 'max:accuracy')
        df = pd.DataFrame(data=[
            ind.metrics(*metric_names) for ind in population
        ])

        metrics = dict(
            num_unique_individuals=len(set(population)),
            num_individuals=len(population),
        )

        for name, values in df.items():
            metrics[f'MAX:{name}'] = values.max()
            metrics[f'MIN:{name}'] = values.min()
            metrics[f'MEAN:{name}'] = values.mean()

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
