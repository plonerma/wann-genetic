""" Command Line Script for generating a series of experiments based on
    specification and base parameters for multivariate testing. """


from argparse import ArgumentParser
import toml
import os
import logging
from collections import namedtuple
import copy


from .multi_spec import Specification


def generate_experiments():
    parser = ArgumentParser(description="Generate a series of experiments "
                            "based on specification and base parameters for "
                            "multivariate testing.")

    parser.add_argument('specification', type=str)

    parser.add_argument('--build_dir', '-o', type=str, default='build')

    args = parser.parse_args()

    spec = Specification(args.specification)

    # ensure the build directory exists
    if not os.path.exists(args.build_dir):
        os.makedirs(args.build_dir)

    for params in spec.generate_experiments():

        slug = params['experiment_name'].lower().replace(' ', '_')
        file_path = os.path.join(args.build_dir, f"{slug}.toml")

        print(f'Saving {file_path}')

        with open(file_path, 'w') as f:
            toml.dump(params, f)


if __name__ == '__main__':
    generate_experiments()
