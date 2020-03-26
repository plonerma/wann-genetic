import toml
import logging
import numpy as np

from .tasks import select_task

from .util import FsInterface, default_params, nested_update
from .evolution import evolution

class Environment:
    default_params = default_params

    def __init__(self, params):
        # set up params based on path or dict and default parameters
        if not isinstance(params, dict):
            params = toml.load(params)

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

        # log used parameters
        params_toml = toml.dumps(self.params)
        self.log.info(f"Running experiments with the following parameters:\n{params_toml}")

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

    def metrics(self, pop):
        indiv_kappas = np.array([
            i.performance.get_metrics('avg_cohen_kappa') for i in pop])

        return dict(
            avg_kappa=np.average(indiv_kappas),
            max_kappa=np.max(indiv_kappas)
        )

    def run(self):
        np.random.seed(self['sampling', 'seed'])

        n = self['population', 'num_generations']
        generations = evolution(self)

        for gen in range(n):
            w = self.sample_weight()
            self.log.debug(f'Sampled weight {w}')

            pop = next(generations)

            self.log.debug(f'Completed generation {gen}')

            m = self.metrics(pop)
            self.log.info(f"#{gen} avg, max kappa: {m['avg_kappa']:.2}, {m['max_kappa']:.2}")

            if not gen % self['storage', 'commit_pop_freq']:
                self.fs.commit_generation(pop)
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
            d = self.params
            for k in keys:
                d = d.get(k)
                if d is None:
                    raise KeyError()
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
