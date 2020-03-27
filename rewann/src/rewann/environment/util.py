import json, toml
from itertools import count, dropwhile
from io import StringIO
import os
from datetime import datetime
import pandas as pd
import collections.abc

import logging


this_directory = os.path.dirname(os.path.abspath(__file__))
default_params_path = os.path.join(this_directory, 'default.toml')
default_params = toml.load(default_params_path)

class FsInterface:
    """Provide interface for storing models on the file system."""

    @classmethod
    def for_env(cls, env):
        if not 'experiment_path' in env:
            env['experiment_path'] = FsInterface.new_path(env)

        return FsInterface(env)

    @classmethod
    def new_path(cls, env):
        name = env['experiment_name']
        base_path = env['storage', 'data_base_path']

        def possible_paths(name):
            date = str(datetime.now().date())
            for i in count():
                if base_path:
                    yield os.path.join(base_path, f'{date}_{name}_{i:03d}')
                else:
                    yield f'{date}_{name}_{i}'

        path = next(dropwhile(os.path.exists, possible_paths(name)))
        return path

    @property
    def log_path(self):
        return self.path(self.env['storage', 'log_filename'])

    def __init__(self, env):
        self.env = env
        self.base_path = env['experiment_path']
        self.gen_digits = len(str(env['population', 'num_generations']))

        if not 'is_report' in env or not env['is_report']:
            logging.info (f"Check log ('{self.log_path}') for details.")

            self.logger = logging.getLogger('experiment')
            self.logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(self.log_path)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

            if not env['debug']:
                self.logger.setLevel(logging.INFO)

            with open(self.path('params.toml'), 'w') as f:
                params = dict(env.params)
                params['is_report'] = True # mark stored params as part of a report
                toml.dump(params, f)

            # log used parameters
            params_toml = toml.dumps(env.params)
            self.logger.info(f"Running experiments with the following parameters:\n{params_toml}")

    def path(self, *parts):
        p = os.path.join(self.base_path, *parts)
        dir = os.path.dirname(p)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return p

    def gen_path(self, i):
        assert i < 10**self.gen_digits
        fstr = f'{{:0{self.gen_digits}d}}.json'
        return self.path('gen', fstr.format(i))

    def dump_population(self, gen, population):
        with open(self.gen_path(gen), 'w') as f:
            json.dump([i.serialize() for i in population], f)

    def load_population(self, gen):
        with open(gen_path(gen), 'r') as f:
            pop = json.load(f)
            return [self.env.Individual.deserialize(i) for i in pop]

    def dump_metrics(self, metrics):
        metrics.to_json(self.path('metrics.json'))

    def load_metrics(self):
        print (self.path('metrics.json'))
        return pd.read_json(self.path('metrics.json'))

    def get_stored_populations(self):
        populations = list()

        for dir, _, files in os.walk(self.path('gen')):
            for f in files:
                gen, _ = f.split('.')
                populations.append(int(gen))
        return sorted(populations)


def nested_update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth#3233356"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
