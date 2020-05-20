import argparse
import numpy as np
import logging

from rewann import Environment


def run_experiment():
    """Execute an experiment (see :doc:`cli`)."""
    parser = argparse.ArgumentParser(description='Post Optimization')

    parser.add_argument('path', type=str, help='path to experiment specification')

    parser.add_argument('--comment', type=str, help='add comment field to params.', default=None)

    args = parser.parse_args()
    env = Environment(args.path)

    if args.comment is not None:
        env['comment'] = args.comment
    env.run()

    logging.info(f'Completed excution.')
