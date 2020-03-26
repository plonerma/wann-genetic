import toml
import logging
from tqdm import tqdm

from .tasks import select_task

from .util import FsInterface, default_params, nested_update
from .evolution import evolution

class Environment:
    default_params = default_params

    def __init__(self, params, root_logger=None):
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

        self.current_weight = 1

    def update_params(self, params):
        self.params = nested_update(self.params, params)

    def run(self):
        n = self['population', 'num_generations']
        for gen, pop in tqdm(zip(range(n), evolution(self)), total=n, unit='gen'):
            self.log.debug(f'Completed generation {gen}')
            self.log.warning('No sensible population performance metrics yet.')

            if not gen % self['config', 'commit_pop_freq']:
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
                    return None
            return d
        else:
            return self.params.get(keys)

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
