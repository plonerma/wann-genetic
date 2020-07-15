import json, toml
from itertools import count, dropwhile
from io import StringIO
import os
from datetime import datetime
import pandas as pd
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
    return metadata.version('wann_genetic')


def derive_path(env):
    """Set experiment data path based on experiment name, data, and available names."""
    if 'experiment_path' not in env:
        name = env['experiment_name']
        base_path = env['storage', 'data_base_path'] or './'
        date = str(datetime.now().date())
        paths = (os.path.join(base_path, f'{date}_{name}_{i:03d}') for i in count())

        env['experiment_path'] = next(dropwhile(os.path.exists, paths))
    return env['experiment_path']


def env_path(env, *parts):
    """Get path string relative to experiment subdirectory."""
    p = os.path.join(env['experiment_path'], *parts)
    dir = os.path.dirname(p)
    if not os.path.exists(dir):  # make sure dir exists
        os.makedirs(dir)
    return p


@contextmanager
def open_data(env, mode='r'):
    """Open hdf5-file that contains the evolved individuals.

    Parameters
    ----------
    mode : str
        'r' opens file for reading, 'w' fo writing.
    """
    env.data_file = h5py.File(env.env_path('data.hdf5'), mode)
    yield env.data_file
    env.data_file.close()


def ind_key(env, i):
    """Key in hdf data file for individual with id i."""
    # maximum number of digits needed
    digits = len(str(env['population', 'num_generations'] * env['population', 'size']))
    return str(i).zfill(digits)


inds_group_key = 'individuals'


def store_ind(env, ind):
    """Store an individual in the hdf5 data file."""
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
    data['mutations'] = ind.mutations
    if ind.parent is not None:
        data['parent'] = ind.parent
    data.create_dataset('edges', data=ind.genes.edges)
    data.create_dataset('nodes', data=ind.genes.nodes)

    return data


def load_ind(env, i):
    """Load individual with id i from the hdf5 data file."""
    inds_group = env.data_file[inds_group_key]
    data = inds_group[ind_key(env, i)]
    if data is None:
        raise IndexError(f"Individual {i} is not stored in data.")
    return ind_from_hdf(env, data)


def ind_from_hdf(env, data):
    """Create individual from h5py dataset."""
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
        birth=data.get('birth')[()],
        mutations=(data.get('mutations')[()] if 'mutations' in data else None),
    )


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
    """Hdf5 key for generation i."""
    digits = len(str(env['population', 'num_generations']))
    return str(i).zfill(digits)


gens_group_key = 'generations'


def store_gen(env, gen, population=None, indiv_measurements=None):
    """Store a generation in the hdf5 data file.

    Parameters
    ----------
    population : [wann_genetic.Individual], optional
        List of individuals to store for this generation.
    indiv_measurements
        Individuals measurements to store for this generation.
    """
    if gens_group_key in env.data_file:
        gens_group = env.data_file[gens_group_key]
    else:
        gens_group = env.data_file.create_group(gens_group_key)

    gen_data = gens_group.create_group(gen_key(env, gen))

    gen_data['id'] = gen

    if population is not None:
        # store individuals
        ids = store_pop(env, gen_data, population)
        # store ids
        gen_data['individuals'] = np.array(ids, dtype=int)

    if indiv_measurements is not None:
        df = indiv_measurements
        # https://gist.github.com/RobbieClarken/9ea7ceaaa3765f536d95
        dataset = gen_data.create_dataset('indiv_measurements', data=df.values)
        dataset.attrs['index'] = np.array(df.index.tolist(), dtype='S')
        dataset.attrs['columns'] = np.array(df.columns.tolist(), dtype='S')


def store_pop(env, gen_data, population):
    """Store a list of individuals in the hdf5 data file."""
    for ind in population:
        store_ind(env, ind)

    return [ind.id for ind in population]


def load_gen(env, gen):
    """Load a generations metadata from the hdf5 data file.

    Parameters
    ----------
    env : wann_genetic.Environment
    gen : int
    """
    if isinstance(gen, str):
        gen = int(gen)
    if isinstance(gen, int):
        gens_group = env.data_file[gens_group_key]
        return gens_group[gen_key(env, gen)]
    return gen


def load_pop(env, gen, ids_only=False):
    """Load a generations population from the hdf5 data file.

    Parameters
    ----------
    env : wann_genetic.Environment
    gen: int
    ids_only : bool, optional
        Only return the ids of the individuals in this generation instead of
        loading the complete individuals.
    """
    gen = load_gen(env, gen)
    if 'individuals' not in gen:
        return None
    inds = gen['individuals']
    if ids_only:
        return inds
    else:
        return [load_ind(env, i) for i in inds]


def store_hof(env):
    """Store the hall of fame in the hdf5 data file."""
    for ind in env.hall_of_fame:
        store_ind(env, ind)

    ids = np.empty(env['population', 'hof_size'], dtype=int)
    ids[:len(env.hall_of_fame)] = [ind.id for ind in env.hall_of_fame]
    ids[len(env.hall_of_fame):] = np.nan

    if 'hall_of_fame' not in env.data_file:
        env.data_file.create_dataset('hall_of_fame', data=ids)
    else:
        env.data_file['/hall_of_fame'][...] = ids


def load_hof(env):
    """Load the hall of fame from the hdf5 data file."""
    if 'hall_of_fame' not in env.data_file:
        return None
    ids = env.data_file['hall_of_fame'][...]
    env.hall_of_fame = [load_ind(env, i) for i in ids]
    return env.hall_of_fame


def load_indiv_measurements(env, gen):
    """Load the measurements for the individuals of a generation."""
    gen = load_gen(env, gen)
    if 'indiv_measurements' not in gen:
        return None
    # https://gist.github.com/RobbieClarken/9ea7ceaaa3765f536d95
    dataset = gen['indiv_measurements']
    index = make_index(dataset.attrs['index'])
    columns = make_index(dataset.attrs['columns'])
    df = pd.DataFrame(data=dataset[...], index=index, columns=columns)
    return df


def stored_generations(env):
    """Return a list of generations that metadata are available for."""
    gens_group = env.data_file[gens_group_key]
    return sorted(gens_group.keys())


def stored_populations(env):
    """Return a list of generations that populations are available for."""
    gens_group = env.data_file[gens_group_key]
    return [
        gen for gen in stored_generations(env)
        if 'individuals' in gens_group[gen]]


def stored_indiv_measurements(env):
    """Return a list of generations that individuals measurements are available for."""
    gens_group = env.data_file[gens_group_key]
    return [
        gen for gen in stored_generations(env)
        if 'indiv_measurements' in gens_group[gen]]


def store_gen_metrics(env, metrics):
    """Store generation metrics from metrics.json."""
    metrics.to_json(env_path(env, 'metrics.json'))


def load_gen_metrics(env):
    """Load generation metrics from metrics.json."""
    return pd.read_json(env_path(env, 'metrics.json'))


def setup_params(env, params):
    """Set up parameters.

    Based on default parameters create complete tree of parameters, get a new
    subdirectory to write in, and derive objectives tuples from the parameters.
    """
    # set up params based on path or dict and default parameters
    if not isinstance(params, dict):
        params_path = params
        if os.path.isdir(params_path):
            params_path = os.path.join(params_path, 'params.toml')
        assert os.path.isfile(params_path)
        params = toml.load(params_path)

        if 'is_report' in params and params['is_report']:
            params['experiment_path'] = os.path.dirname(params_path)

    env.update_params(env.default_params)
    env.update_params(params)

    # ensure experiment name is defined
    if 'experiment_name' not in env:
        env['experiment_name'] = '{}_run'.format(env['task', 'name'])

    derive_path(env)

    def signed_metric(m):
        if m[0] == '-':
            return (m[1:], -1)
        else:
            return (m, 1)

    objs = list(map(signed_metric, env['selection', 'objectives']))

    assert len(objs) > 0

    env.objectives = list(zip(*objs))

    env.hof_metric = signed_metric(env['selection', 'hof_metric'])


class TimeStore:
    t0 = None
    total = 0
    dt = None

    def start(self):
        self.t0 = time()

    def stop(self):
        self.dt = time() - self.t0
        self.total += self.dt
