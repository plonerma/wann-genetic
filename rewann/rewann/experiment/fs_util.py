import toml
from itertools import count, dropwhile
from hashfs import HashFS
from io import StringIO
import os
from datetime import datetime

import collections.abc

import logging

this_directory = os.path.dirname(os.path.abspath(__file__))
default_params_path = os.path.join(this_directory, 'default.toml')


class FsInterface:
    """Provide interface for storing models on the file system."""

    @classmethod
    def new_path(cls, name, *args, base_path='', **kwargs):
        def possible_paths(name):
            date = str(datetime.now().date())
            for i in count():
                if base_path:
                    yield os.path.join(base_path, f'{date}_{name}_{i:03d}')
                else:
                    yield f'{date}_{name}_{i}'

        path = next(dropwhile(os.path.exists, possible_paths(name)))
        return cls(path, *args, **kwargs)

    def __init__(self, path, gen_digits=4):
        self.base_path = path
        self.gen_digits = gen_digits
        self.hashfs = HashFS(os.path.join(self.base_path, 'objects'),
                             depth=1, width=2, algorithm='sha256')
        logging.basicConfig(filename=os.path.join(self.base_path, 'objects'),
                            filemode='a',
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            level=logging.DEBUG)

    def path(self, *parts):
        p = os.path.join(self.base_path, *parts)
        dir = os.path.dirname(p)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return p

    def put(self, data, path=None):
        if path is not None:
            with open(path, 'w') as f:
                toml.dump(data, f)
            return path
        else:
            stream = StringIO()
            toml.dump(data, stream)
            return self.hashfs.put(stream).id

    def get(self, path):
        fileio = fs.open(address.abspath)
        return toml.load(fileio)

    def gen_path(self, i):
        assert i < 10**self.gen_digits
        fstr = f'{{:0{self.gen_digits}d}}.toml'
        return self.path('gen', fstr.format(i))

    def commit_generation(self, gen, include_population=False):
        d = dict()

        #if gen.performance:
        #    d['performance'] = gen.performance

        if include_population:
            d['individuals'] = list()
            for i in gen.individuals:
                h = self.put(i.serialize())
                d['individuals'].append(h)

        return self.put(
            # save generation data there
            data=d,
            # find first free path
            path=next(dropwhile(os.path.isfile, (self.gen_path(i) for i in count())))
        )

    def retrieve_generation(self, i=None, path=None):
        if path is None:
            assert i is not None
            path = self.gen_path(i)
        d = self.get(path)
        return Population.deserialize(d)


def nested_update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth#3233356"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
