import json, toml
from itertools import count, dropwhile
from io import StringIO
import os
from datetime import datetime
import pandas as pd
from collections.abc import Mapping
import copy
from contextlib import contextmanager

import logging

import io
import numpy as np
import h5py


this_directory = os.path.dirname(os.path.abspath(__file__))
default_params_path = os.path.join(this_directory, 'default.toml')
default_params = toml.load(default_params_path)


def derive_path(env):
    if not 'experiment_path' in env:
        name = env['experiment_name']
        base_path = env['storage', 'data_base_path'] or './'
        date = str(datetime.now().date())
        paths = (os.path.join(base_path, f'{date}_{name}_{i:03d}') for i in count())

        env['experiment_path'] = next(dropwhile(os.path.exists, paths))
    return env['experiment_path']

def env_path(env, *parts):
    p = os.path.join(env['experiment_path'], *parts)
    dir = os.path.dirname(p)
    if not os.path.exists(dir): os.makedirs(dir)  # make sure dir exists
    return p

@contextmanager
def open_data(env, mode='r'):
    env.data_file = h5py.File(env.env_path('data.hdf5'), mode)
    yield env.data_file
    env.data_file.close()

# Storing and retrieving individuals

def ind_key(env, i):
    # maximum number of digits needed
    digits = len(str(env['population', 'num_generations'] * env['population', 'size']))
    return str(i).zfill(digits)

inds_group_key = 'individuals'

def store_ind(env, ind):
    if inds_group_key in env.data_file:
        inds_group = env.data_file[inds_group_key]
    else:
        inds_group = env.data_file.create_group(inds_group_key)
    ki = ind_key(env, ind.id)
    if ki in inds_group:
        return inds_group[ki]

    data = inds_group.create_group(ki)

    data['id'] = ind.id
    data['birth'] = ind.birth
    if ind.parent is not None:
        data['parent'] = ind.parent
    data.create_dataset('edges', data=ind.genes.edges)
    data.create_dataset('nodes', data=ind.genes.nodes)

    return data

def load_ind(env, i):
    inds_group = env.data_file[inds_group_key]
    return ind_from_hdf(env, inds_group[ind_key(env, i)])

def ind_from_hdf(env, data):
    return env.ind_class(
        genes=env.ind_class.Genotype(
            edges=data['edges'][()],
            nodes=data['nodes'][()],
            n_in=env.task.n_in,
            n_out=env.task.n_out,
        ),
        parent=data.get('parent')[()],
        id=data.get('id')[()],
        birth=data.get('birth')[()]
    )

# storing and retrieving a generation

def gen_key(env, i):
    digits = len(str(env['population', 'num_generations']))
    return str(i).zfill(digits)

gens_group_key = 'generations'

def store_gen(env, gen, population=None):
    if gens_group_key in env.data_file:
        gens_group = env.data_file[gens_group_key]
    else:
        gens_group = env.data_file.create_group(gens_group_key)

    gen_data = gens_group.create_group(gen_key(env, gen))

    gen_data['id'] = gen

    if population is not None:
        store_pop(env, gen_data, population)

def store_pop(env, gen_data, population):
    for ind in population:
        store_ind(env, ind)
    # store ids
    ids = [ind.id for ind in population]
    gen_data['individuals'] = np.array(ids, dtype=int)

def load_gen(env, i):
    gens_group = env.data_file[gens_group_key]
    return gens_group[gen_key(env, i)]

def load_pop(env, gen):
    if isinstance(gen, str):
        gen = int(gen)
    if isinstance(gen, int):
        gen = load_gen(env, gen)
    if not 'individuals' in gen:
        return None
    inds = gen['individuals']
    return [load_ind(env, i) for i in inds]

def existing_populations(env):
    gens_group = env.data_file[gens_group_key]

    return sorted([
        gen for gen in gens_group.keys()
        if 'individuals' in gens_group[gen]
    ])

def store_metrics(env, metrics):
    metrics.to_json(env_path(env, 'metrics.json'))

def load_metrics(env):
    return pd.read_json(env_path(env, 'metrics.json'))


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


    env.params = nested_update(copy.deepcopy(env.default_params), params)

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
