import sys
import numpy as np
import logging

from rewann import Environment


def run_experiment():
    args = sys.argv[1:]
    if len(args) != 1:
        print ("usage: run_experiment 'path'")
        return

    path, = args

    exp = Environment(params=path)
    exp.run()

    logging.info(f'Completed excution.')
