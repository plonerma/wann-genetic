import logging
import sys
from rewann import Environment
import toml
import numpy as np

import logging

def test_experiment(tmp_path):
    params = toml.load('tests/test_experiment.toml')
    params['experiment_path'] = tmp_path
    exp = Environment(params=params)
    exp.run()
    # Assert that there is at least moderate agreement between predicted and true classifications

    metrics = exp.generation_metrics(exp.last_population)
    max_kappa = metrics['MAX:kappa.max']
    mean_kappa = metrics['MEAN:kappa.mean']

    logging.info(f'Mean kappa score: {mean_kappa}')
    logging.info(f'Max kappa score: {max_kappa}')
    assert mean_kappa > 0 and max_kappa > 0.4
