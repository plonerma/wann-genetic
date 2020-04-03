import json, toml
from itertools import count, dropwhile
from io import StringIO
import os
from datetime import datetime
import pandas as pd
from collections.abc import Mapping

import logging

import io
import numpy as np
from base64 import b64encode, b64decode


this_directory = os.path.dirname(os.path.abspath(__file__))
default_params_path = os.path.join(this_directory, 'default.toml')
default_params = toml.load(default_params_path)


def derive_path(env):
    if not 'experiment_path' in env:
        name = env['experiment_name']
        base_path = env['storage', 'data_base_path']
        def possible_paths(name):
            date = str(datetime.now().date())
            for i in count():
                if base_path:
                    yield os.path.join(base_path, f'{date}_{name}_{i:03d}')
                else:
                    yield f'{date}_{name}_{i}'

        env['experiment_path'] = next(dropwhile(os.path.exists, possible_paths(name)))
    return env['experiment_path']

def env_path(env, *parts):
    p = os.path.join(env['experiment_path'], *parts)
    dir = os.path.dirname(p)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return p

def ind_path(env, i):
    # maximum number of digits needed
    digits = len(str(env['population', 'num_generations'] * env['population', 'size']))
    fstr = f'{{:0{digits}d}}.json'
    return env_path(env, 'individuals', fstr.format(i))

def gen_path(env, i):
    digits = len(str(env['population', 'num_generations']))
    fstr = f'{{:0{digits}d}}.json'
    return env_path(env, 'gen', fstr.format(i))

def serialize_array(a : np.array):
    # Encoding as b64 is not optimal, but it allows including it in a json file
    encodedBytes = b64encode(a.tobytes())
    return str(encodedBytes, 'utf-8')

def deserialize_array(s : str, dtype : np.dtype):
    bytes = b64decode(s)
    return np.frombuffer(bytes, dtype)

def dump_ind(env, ind):
    with open(ind_path(env, ind.id), 'w') as f:
        data = dict(
            birth=ind.birth, id=ind.id,
            edges=serialize_array(ind.genes.edges),
            nodes=serialize_array(ind.genes.nodes),
        )

        if env['storage']['include_prediction_records'] and ind.prediction_records:
            cm_list, weight_list = ind.prediction_records
            data['record'] = dict(
                n_classes=cm_list[0].shape[0],
                cm_stack=[serialize_array(cm) for cm in cm_list],
                weights=weight_list,
            )

        json.dump(data, f)
    return ind.id

def load_ind(env, i):
    with open(ind_path(env, i), 'r') as f:
        data = json.load(f)

    Genotype = env.ind_class.Genotype
    p = dict(genes=Genotype(
        edges=deserialize_array(data['edges'], dtype=list(Genotype.edge_encoding)),
        nodes=deserialize_array(data['nodes'], dtype=list(Genotype.node_encoding)),
        n_in=env.task.n_in, n_out=env.task.n_out
    ))

    if 'record' in data:
        cm_list = list()
        n_classes = data['record']['n_classes']
        for cm in data['record']['cm_stack']:
            cm = deserialize_array(data['record']['cm_stack'], dtype=int)
            s = cm.shape[0]
            cm.reshape((n_classes, n_classes))
            cm_list.append(cm)
        weights = data['record']['weights']

        p['prediction_records'] = cm_list, weights
    return env.ind_class(**p)

def dump_pop(env, gen, population):
    with open(gen_path(env, gen), 'w') as f:
        json.dump([dump_ind(env, i) for i in population], f)

def load_pop(env, gen):
    with open(gen_path(env, gen), 'r') as f:
        pop = json.load(f)
        return [load_ind(env, i) for i in pop]

def dump_metrics(env, metrics):
    metrics.to_json(env_path(env, 'metrics.json'))

def load_metrics(env):
    return pd.read_json(env_path(env, 'metrics.json'))

def existing_populations(env):
    populations = list()
    for dir, _, files in os.walk(env.path('gen')):
        for f in files:
            gen, _ = f.split('.')
            populations.append(int(gen))
    return sorted(populations)

def setup_logging(env):
    log_path = env_path(env, env['storage', 'log_filename'])
    logging.info (f"Check log ('{log_path}') for details.")

    env.log = logging.getLogger('experiment')
    env.log.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    env.log.addHandler(fh)

    if not env['debug']:
        env.log.setLevel(logging.INFO)

    with open(env_path(env, 'params.toml'), 'w') as f:
        params = dict(env.params)
        params['is_report'] = True # mark stored params as part of a report
        toml.dump(params, f)

    # log used parameters
    params_toml = toml.dumps(env.params)
    env.log.info(f"Running experiments with the following parameters:\n{params_toml}")

def setup_params(env, params):
    # set up params based on path or dict and default parameters
    if not isinstance(params, dict):
        params_path = params
        if os.path.isdir(params_path):
            params_path = os.path.join(params_path, 'params.toml')
        assert os.path.isfile(params_path)
        params = toml.load(params_path)

        if 'is_report' in params and params['is_report']:
            params['experiment_path'] = os.path.dirname(params_path)

    env.params = nested_update(dict(env.default_params), params)

    derive_path(env)

    # ensure experiment name is defined
    if env['experiment_name'] is None:
        env['experiment_name'] = '{}_run'.format(task_name)

def nested_update(d, u):
    """ Update nested parameters.

    Source:
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth#3233356"""
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
