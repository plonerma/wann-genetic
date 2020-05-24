import logging
import sys
from rewann import Environment
import toml
import numpy as np

import pytest

import logging

def experiment_test(params_path, tmp_path, mean_treshold=0.0, max_treshold=0.4):
    logging.info("Starting training")
    params = toml.load(params_path)
    params['experiment_path'] = tmp_path
    exp = Environment(params=params)
    exp.run()
    # Assert that there is at least moderate agreement between predicted and true classifications

    logging.info("Starting evaluation")

    metrics = exp.population_metrics(exp.hall_of_fame)
    max_kappa = metrics['MAX:kappa.max']
    mean_kappa = metrics['MEAN:kappa.mean']

    logging.info(f'Mean kappa score: {mean_kappa}')
    logging.info(f'Max kappa score: {max_kappa}')
    assert mean_kappa > mean_treshold and max_kappa > max_treshold

@pytest.mark.slow
def test_layer_agnostic(tmp_path):
    experiment_test('tests/test_layer_agnostic.toml', tmp_path)

@pytest.mark.slow
def test_layer_based(tmp_path):
    experiment_test('tests/test_layer_based.toml', tmp_path)

@pytest.mark.slow
def test_negative_edges(tmp_path):
    experiment_test('tests/test_neg_edges.toml', tmp_path)

@pytest.mark.slow
def test_recurrent(tmp_path):
    experiment_test('tests/test_recurrent_echo.toml', tmp_path, max_treshold=0.2)

#@pytest.mark.slow
def test_torch(tmp_path):
    experiment_test('tests/test_torch.toml', tmp_path)
