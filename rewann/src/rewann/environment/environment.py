import toml
import logging
import numpy as np
import pandas as pd
import os

from .tasks import select_task

from .evolution import evolution

class Environment:
    from ..individual import Individual
    from .util import (default_params, setup_params, setup_logging,
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
                self.dump_pop(gen, pop)
                self.dump_metrics(pd.DataFrame(data=metrics))

        self.dump_metrics(pd.DataFrame(data=metrics))
        self.last_population = pop

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
