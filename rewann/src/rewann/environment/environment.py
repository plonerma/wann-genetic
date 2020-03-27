import toml
import logging
import numpy as np
import pandas as pd
import os

from .tasks import select_task

from .util import FsInterface, default_params, nested_update
from .evolution import evolution

class Environment:
    default_params = default_params

    from ..individual import Individual

    def __init__(self, params):
        # set up params based on path or dict and default parameters
        if not isinstance(params, dict):
            params_path = params
            if os.path.isdir(params_path):
                params_path = os.path.join(params_path, 'params.toml')
            assert os.path.isfile(params_path)
            params = toml.load(params_path)

            if 'is_report' in params and params['is_report']:
                params['experiment_path'] = os.path.dirname(params_path)

        self.params = dict(self.default_params)
        self.update_params(params)

        # choose task
        task_name = self['task', 'name']
        self.task = select_task(task_name)

        # ensure experiment name is defined
        if self['experiment_name'] is None:
            self['experiment_name'] = '{}_run'.format(task_name)

        # init fs interface
        self.fs = FsInterface.for_env(self)

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

    def update_params(self, params):
        self.params = nested_update(self.params, params)

    def get_metrics(self, pop):
        indiv_kappas = np.array([
            i.record.get_metrics('avg_cohen_kappa') for i in pop])

        indiv_accs = np.array([
            i.record.get_metrics('avg_accuracy') for i in pop])

        indiv_n_hidden_nodes = np.array([i.network.n_hidden for i in pop])

        indiv_n_edges = np.array([len(i.genes.edges) for i in pop])

        return dict(
            avg_kappa=np.average(indiv_kappas),
            max_kappa=np.max(indiv_kappas),
            min_kappa=np.min(indiv_kappas),

            avg_accuracy=np.average(indiv_accs),
            max_accuracy=np.max(indiv_accs),
            min_accuracy=np.min(indiv_accs),

            avg_n_hidden_nodes=np.average(indiv_n_hidden_nodes),
            max_n_hidden_nodes=np.max(indiv_n_hidden_nodes),
            min_n_hidden_nodes=np.min(indiv_n_hidden_nodes),

            avg_n_edges=np.average(indiv_n_edges),
            max_n_edges=np.max(indiv_n_edges),
            min_n_edges=np.min(indiv_n_edges),

            num_unique_individuals=len(set(pop)),
            num_individuals=len(pop),
        )

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

            gen_metrics = self.get_metrics(pop)
            self.log.info(f"#{gen} avg, max kappa: {gen_metrics['avg_kappa']:.2}, {gen_metrics['max_kappa']:.2}")

            metrics.append(gen_metrics)

            if gen % self['storage', 'commit_population_freq'] == 0:
                self.fs.dump_population(gen, pop)

        self.metrics = pd.DataFrame(data=metrics)
        self.fs.dump_metrics(self.metrics)

        self.last_population = pop

    @property
    def log(self):
        return self.fs.logger

    @property
    def path(self):
        return self.fs.base_path

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
