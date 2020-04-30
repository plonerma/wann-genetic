import json, toml
from itertools import count, dropwhile
from io import StringIO
import os
from datetime import datetime
import pandas as pd
from collections.abc import Mapping
import copy
from contextlib import contextmanager
from time import time
import logging

import io
import numpy as np
import h5py

try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata


this_directory = os.path.dirname(os.path.abspath(__file__))
default_params_path = os.path.join(this_directory, 'default.toml')
default_params = toml.load(default_params_path)

def get_version():
    return metadata.version('rewann')


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
    data = inds_group[ind_key(env, i)]
    if data is None:
        raise IndexError(f"Individual {i} is not stored in data.")
    return ind_from_hdf(env, data)

def ind_from_hdf(env, data):
    p = data.get('parent')
    p = None if p is None else p[()]
    return env.ind_class(
        genes=env.ind_class.Genotype(
            edges=data['edges'][()],
            nodes=data['nodes'][()],
            n_in=env.task.n_in,
            n_out=env.task.n_out,
        ),
        parent=p,
        id=data.get('id')[()],
        birth=data.get('birth')[()]
    )

# storing and retrieving a generation

def make_index(raw):
    """Utility for retrieving pandas dataframes.

    Source: https://gist.github.com/RobbieClarken/9ea7ceaaa3765f536d95
    """
    index = raw.astype('U')
    if index.ndim > 1:
        return pd.MultiIndex.from_tuples(index.tolist())
    else:
        return pd.Index(index)


def gen_key(env, i):
    digits = len(str(env['population', 'num_generations']))
    return str(i).zfill(digits)

gens_group_key = 'generations'

def store_gen(env, gen, population=None, indiv_metrics=None):
    if gens_group_key in env.data_file:
        gens_group = env.data_file[gens_group_key]
    else:
        gens_group = env.data_file.create_group(gens_group_key)

    gen_data = gens_group.create_group(gen_key(env, gen))

    gen_data['id'] = gen

    if population is not None:
        store_pop(env, gen_data, population)
    if indiv_metrics is not None:
        df = indiv_metrics
        # https://gist.github.com/RobbieClarken/9ea7ceaaa3765f536d95
        dataset = gen_data.create_dataset('indiv_metrics', data=df.values)
        dataset.attrs['index'] = np.array(df.index.tolist(), dtype='S')
        dataset.attrs['columns'] = np.array(df.columns.tolist(), dtype='S')

def store_pop(env, gen_data, population):
    for ind in population:
        store_ind(env, ind)
    # store ids
    ids = [ind.id for ind in population]
    gen_data['individuals'] = np.array(ids, dtype=int)

def load_gen(env, gen):
    if isinstance(gen, str):
        gen = int(gen)
    if isinstance(gen, int):
        gens_group = env.data_file[gens_group_key]
        return gens_group[gen_key(env, gen)]
    return gen

def load_pop(env, gen, ids_only=False):
    gen = load_gen(env, gen)
    if not 'individuals' in gen:
        return None
    inds = gen['individuals']
    if ids_only:
        return inds
    else:
        return [load_ind(env, i) for i in inds]

def store_hof(env):
    for ind in env.hall_of_fame:
        store_ind(env, ind)

    ids = np.empty(env['population', 'hof_size'], dtype=int)
    ids[:len(env.hall_of_fame)] = [ind.id for ind in env.hall_of_fame]
    ids[len(env.hall_of_fame):] = np.nan

    if 'hall_of_fame' not in env.data_file:
        hof_data = env.data_file.create_dataset('hall_of_fame', data=ids)
    else:
        env.data_file['/hall_of_fame'][...] = ids

def load_hof(env):
    if 'hall_of_fame' not in env.data_file:
        return None
    ids = env.data_file['hall_of_fame'][...]
    env.hall_of_fame = [load_ind(env, i) for i in ids]
    return env.hall_of_fame


def load_indiv_metrics(env, gen):
    gen = load_gen(env, gen)
    if not 'indiv_metrics' in gen:
        return None
    # https://gist.github.com/RobbieClarken/9ea7ceaaa3765f536d95
    dataset = gen['indiv_metrics']
    index = make_index(dataset.attrs['index'])
    columns = make_index(dataset.attrs['columns'])
    df = pd.DataFrame(data=dataset[...], index=index, columns=columns)
    return df

def stored_generations(env):
    gens_group = env.data_file[gens_group_key]
    return sorted(gens_group.keys())

def stored_populations(env):
    gens_group = env.data_file[gens_group_key]
    return [
        gen for gen in stored_generations(env)
        if 'individuals' in gens_group[gen]]

def stored_indiv_metrics(env):
    gens_group = env.data_file[gens_group_key]
    return [
        gen for gen in stored_generations(env)
        if 'indiv_metrics' in gens_group[gen]]

def store_gen_metrics(env, metrics):
    metrics.to_json(env_path(env, 'metrics.json'))

def load_gen_metrics(env):
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

    env.params = copy.deepcopy(env.default_params)

    nested_update(env.params, params)

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


class TimeStore:
    t0 = None
    total = 0
    dt = None

    def start(self):
        self.t0 = time()

    def stop(self):
        self.dt = time() - self.t0
        self.total += self.dt
