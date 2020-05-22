"""Tools for generating and anlysing series of experiments."""



import os
import toml, json
import logging

import pandas as pd

from itertools import product
from collections import OrderedDict
from collections.abc import Mapping
from functools import reduce
from typing import Iterable, Collection, Sequence

from rewann.util import ParamTree



class Variable:
    """Represents a variable in an experiment series."""
    def __init__(self, key: tuple, values: Sequence, fmt: str=""):
        self.key = key
        self.values = values
        self.fmt = fmt

    def value_name(self, i : int):
        """Return name part for value with index `i` of this variation."""
        v = self.values[i]

        if isinstance(v, dict) and '_fmt' in v:
            fmt = v['_fmt']
        else:
            fmt = self.fmt

        if not fmt:
            return str(v)
        elif isinstance(v, dict):
            return fmt.format(**v)
        elif isinstance(v, list):
            return fmt.format(*v)
        else:
            return fmt.format(v)

    def value(self, i : int):
        value = self.values[i]
        if isinstance(value, Mapping):
            return {k:v for k,v in value.items() if k != '_fmt'}
        else:
            return value

    def set_value(self, params : ParamTree, i : int):
        params.update_params_at(self.key, self.value(i))

    def iter_indices(self):
        return range(len(self.values))



class ExperimentSeries:
    """Represents an experiment series.

    Can be used to generate experiment files and read in data from previously
    executed experiments.
    """

    experiment_paths = None

    def __init__(self, spec : Mapping, base_params=dict(), data_path=None):
        self.variables = OrderedDict()
        self.base_params = base_params
        self.name_fstr = spec['experiment_name']
        self.data_path = data_path

        self.init_variables(spec)

    @classmethod
    def from_spec_file(cls, spec_path : str, data_path=None):
        """Create experiment series from toml specification file."""
        if os.path.isdir(spec_path):
            spec_path = os.path.join(spec_path, 'spec.toml')

        assert os.path.isfile(spec_path)
        spec = toml.load(spec_path)

        dir = os.path.dirname(os.path.realpath(spec_path))

        base_params_path = os.path.join(
            dir, spec.get('base_params', 'base.toml'))

        base_params = toml.load(base_params_path)

        if data_path is None:
            data_path = os.path.join(dir, 'data')

        return cls(spec=spec, base_params=base_params, data_path=data_path)

    def init_variables(self, spec : Mapping):
        """Initialize the variables."""
        for name, var in spec.items():
            if isinstance(var, dict):
                key = var['key']

                if isinstance(key, list):
                    key = tuple(key)
                elif isinstance(key, str):
                    key = (key,)
                else:
                    assert isinstance(key, tuple)

                self.variables[name] = Variable(key, var['values'],
                                                var.get('fmt', None))
            else:
                if not name in ('experiment_name', 'base_params'):
                    logging.warning(f'Key {name} unkown. Skipping.')

    def var_names(self) -> Iterable:
        """Return an iterable containing all variable names (in order)."""
        return self.variables.keys()

    def vars(self) -> Iterable:
        """Return an iterable containing all variables (in order)."""
        return self.variables.values()

    def num_configurations(self) -> Iterable:
        """Return the product of the number of values of each variable."""
        return reduce(lambda p, var: p*len(var.values), self.vars(), 1)

    def configurations(self) -> Iterable:
        """Return iteratator containing all the possible configurations."""
        return product(*[var.iter_indices() for var in self.vars()])

    def configuration_name(self, c : Collection[int]) -> str:
        """Get full experiment name for a given configuration."""
        assert len(c) == len(self.variables)

        return self.name_fstr.format(**{
            name: var.value_name(i)
            for name, var, i in zip(self.var_names(), self.vars(), c)
        })

    def configuration_params(self, c : Collection[int], use_base=True):
        params = ParamTree()

        if use_base:
            params.update_params(self.base_params)

        for var, i in zip(self.variables.values(), c):
            var.set_value(params, i)

        params['experiment_name'] = self.configuration_name(c)
        return params

    def flat_values(self, c : Collection[int]) -> Mapping:
        """Flatten all values.

        If a variable has a dict value, it will use multiple fields with keys
        `<var_name>/<dict-key>`. This is useful for building pandas data
        frames.
        """
        values = dict()

        for var_name, var, i in zip(self.var_names(), self.vars(), c):
            value = var.value(i)
            if isinstance(value, dict):
                for k, v in value.items():
                    values[f"{var_name}/{k}"] = v
            else:
                values[var_name] = value
            values[f"{var_name}/_name"] = var.value_name(i)


        return values

    def create_experiment_files(self, outdir : str):
        """Generate parameters file for this series of experiments.

        Parameters
        ----------
        outdir : str
            path to directory were the the parameter files will be saved
        """
        # ensure the build directory exists
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        n = self.num_configurations()

        logging.info(f'Generating {n} experiment.')

        for con in self.configurations():
            params = self.configuration_params(con)

            slug = params['experiment_name'].lower().replace(' ', '_')
            file_path = os.path.join(outdir, f"{slug}.toml")

            logging.info(f'Saving {file_path}')

            with open(file_path, 'w') as f:
                toml.dump(params, f)
        logging.info(f'Generated {n} files.')

    def discover_data_dir(self, data_path=None):
        """Check directory for existing experiments.

        Checks whether found experiments match any expected experiment name.
        """
        if data_path is None:
            data_path = self.data_path

        assert data_path is not None

        paths = dict()
        for c in self.configurations():
            paths[self.configuration_name(c).lower()] = None

        _, potential_results, _ = next(os.walk(data_path))

        for dir in potential_results:
            dir = os.path.join(data_path, dir)

            params_path = os.path.join(dir, 'params.toml')

            if not os.path.exists(params_path):
                logging.info(f'Skipping non-experiment directory: {dir}')
                continue

            params = toml.load(params_path)

            exp_name = params['experiment_name'].lower()

            if exp_name not in paths:
                logging.info(f'Found non-matching experiment: {exp_name}')
                continue

            if paths[exp_name] is not None:
                logging.critical(f'Skipping duplicate experiment {exp_name}.')
                continue

            paths[exp_name] = dir

        n_found = sum((1 for p in paths.values() if p is not None))
        n_total = self.num_configurations()

        if n_found == 0:
            logging.error("No data loaded.")
        elif n_found < n_total:
            logging.warning(f'Only {n_found} out of {n_total} experiments found.')

        self.experiment_paths = paths

    def assemble_stats(self):
        """Load stats for available experiments."""
        if self.experiment_paths is None:
            try:
                self.discover_data_dir()
            except AssertionError:
                raise RuntimeError("Load set correct experiments data path first.")

        data = list()

        for c in self.configurations():
            c_name = self.configuration_name(c).lower()
            if self.experiment_paths[c_name] is not None:

                dir = self.experiment_paths[c_name]

                stats_path = os.path.join(dir, 'report', 'stats.json')
                if not os.path.exists(stats_path) or not os.path.isfile(stats_path):
                    logging.warning(f"Experiment {c_name} in dir {dir} has not stats.")
                    continue

                with open(stats_path, 'r') as f:
                    stats = json.load(f)

                stats.update(self.flat_values(c))
                data.append(stats)
        return pd.DataFrame(data)
