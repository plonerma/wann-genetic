import sys
import numpy as np
import logging

from rewann import Environment


def run_experiment():
    args = sys.argv[1:]

    if len(args) == 1:
        path, = args
        comment = None
    elif len(args) == 2:
        path, comment = args
    else:
        print ("usage: run_experiment 'path'")
        return

    exp = Environment(params=path)
    if comment is not None:
        exp['comment'] = comment
    exp.run()

    logging.info(f'Completed excution.')
