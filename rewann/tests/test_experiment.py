import logging
import sys
from rewann import Environment
import toml
import numpy as np

def test_experiment(tmp_path):
    params = toml.load('tests/test_experiment.toml')
    params['experiment_path'] = tmp_path
    exp = Environment(params=params)
    exp.run()
    # Assert that there is at least moderate agreement between predicted and true classifications

    indiv_kappas = np.array([i.record.get_metrics('avg_cohen_kappa') for i in exp.last_population])
    exp.log.info(indiv_kappas)

    avg_kappa = np.average(indiv_kappas)
    exp.log.info(f'Average kappa score: {avg_kappa}')
    max_kappa = np.max(indiv_kappas)
    exp.log.info(f'Max kappa score: {max_kappa}')
    assert avg_kappa > 0.1 and max_kappa > 0.4
