import numpy as np

from rewann import RecurrentGenes
from rewann.individual.numpy import RecurrentIndividual as NumpyInd
from rewann.individual.torch import RecurrentIndividual as TorchInd


ff_genes = RecurrentGenes(
    n_in=1,
    n_out=2,
    nodes=[  # one in, one bias, one out

        # id, out, activation function
        (2, True, 0),  # active if in is active
        (3, True, 0),  # active if in is not active
    ],
    edges=[
        # id, src, dst, sgn, enabled, recurrent
        (  0,   0,   2,   1, True, False),
        (  1,   0,   3,  -1, True, False),
        (  2,   1,   3,   1, True, False),
    ]
)
ff_inputs = np.array([
    # sequences, sequence elements, values
    [[0], [1]],
    [[1], [0]]
])

ff_expected_outputs = np.array([[[1, 0], [0, 1]]])

ff_weights = np.array([1])


def test_numpy_ff():
    i = NumpyInd(genes=ff_genes)
    i.express()

    outputs = i.get_measurements(
        weights=ff_weights, x=ff_inputs,
        measures=['predictions'])['predictions']

    valid = ff_expected_outputs >= 0

    assert np.all(
        outputs[valid] == ff_expected_outputs[valid]
    )


def test_torch_ff():
    i = NumpyInd(genes=ff_genes)
    i.express()
    outputs = i.get_measurements(
        weights=ff_weights, x=ff_inputs,
        measures=['predictions'])['predictions']

    valid = ff_expected_outputs >= 0

    assert np.all(
        outputs[valid] == ff_expected_outputs[valid]
    )


re_genes = RecurrentGenes(
    n_in=1,
    n_out=2,
    nodes=[  # one in, one bias, one out

        # id, out, activation function
        (2, True, 0),  # same as ff with one timestep delay
        (3, True, 0),
    ],
    edges=[
        # id, src, dst, sgn, enabled, recurrent
        (  0,   0,   2,   1, True   , True),
        (  1,   0,   3,  -1, True   , True),
        (  2,   1,   3,   1, True   , False),
    ]
)
re_inputs = np.array([
    # sequences, sequence elements, values
    [[0], [1]],
    [[1], [0]]
])

re_expected_outputs = np.array([[
    [-1, 1],
    [-1, 0]
]])

re_weights = np.array([1])


def test_numpy_re():
    i = NumpyInd(genes=re_genes)
    i.express()

    ms = i.get_measurements(
        weights=re_weights, x=re_inputs, measures=['predictions', 'raw'])

    predictions = ms['predictions']

    valid = re_expected_outputs >= 0

    assert np.all(
        predictions[valid] == re_expected_outputs[valid]
    )


def test_torch_re():
    i = TorchInd(genes=re_genes)
    i.express()

    ms = i.get_measurements(
        weights=re_weights, x=re_inputs, measures=['predictions', 'raw'])
    predictions = ms['predictions']

    valid = re_expected_outputs >= 0

    assert np.all(
        predictions[valid] == re_expected_outputs[valid]
    )
