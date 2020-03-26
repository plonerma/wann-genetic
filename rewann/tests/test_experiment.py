import logging
import sys
from rewann import Environment
import toml
import numpy as np

# setup logging
root = logging.getLogger()
root.setLevel(logging.WARNING)

sh = logging.StreamHandler()

sh.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

sh.setLevel(logging.WARNING)

# streamlit registers a handler -> overwrite that
root.handlers = [sh]


def test_experiment(tmp_path):
    params = toml.load('tests/test_experiment.toml')
    params['experiment_path'] = tmp_path
    exp = Environment(params=params, root_logger=root)
    exp.run()
    # Assert that there is at least moderate agreement between predicted and true classifications
    assert np.average(np.array([i.performance.get_metrics('cohen_kappa') for i in exp.last_population])) > 0.4
