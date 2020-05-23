""" Command Line Script for generating a series of experiments based on
    specification and base parameters for multivariate testing. """


from argparse import ArgumentParser

import logging

from .experiment_series import ExperimentSeries



def generate_experiments():
    parser = ArgumentParser(description="Generate a series of experiments "
                            "based on specification and base parameters for "
                            "multivariate testing.")

    parser.add_argument('specification', type=str)

    parser.add_argument('--build_dir', '-o', type=str, default='build')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    spec = ExperimentSeries.from_spec_file(args.specification).create_experiment_files(args.build_dir)
