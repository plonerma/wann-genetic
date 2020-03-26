import sys
import numpy as np

from . import Environment

def run_experiment():
    args = sys.argv[1:]
    if len(args) != 1:
        print ("usage: run_experiment <path>")
        return

    path, = args


    exp = Environment(params=path)
    exp.run()

    indiv_kappas = np.array([i.performance.get_metrics('avg_cohen_kappa') for i in exp.last_population])
    exp.log.info(indiv_kappas)

    avg_kappa = np.average(indiv_kappas)
    exp.log.info(f'Average kappa score: {avg_kappa}')
    max_kappa = np.max(indiv_kappas)
    exp.log.info(f'Max kappa score: {max_kappa}')
