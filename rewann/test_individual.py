import numpy as np

from rewann.individual import Individual, Network, Genotype

sample = Individual(genes=Genotype(
    n_in=3,
    n_out=2,
    nodes=[
        # id, type, activation function
        (4, True, 0),
        (5, True, 0),
        (7, False, 0),
        (8, False, 0),
        (10, False, 0),
        (12, False, 0),
        (14, False, 0)
    ],
    edges=[
        # innovation id, src, dest, enabled
        ( 3, 12,  7, True), # not necessarily ordered
        ( 1,  1, 12, True),
        ( 0,  0, 12, True),
        ( 2, 12,  4, True),
        ( 4,  7,  4, True),
        (15,  2,  7, True),
        ( 6,  7, 10, True),
        ( 7, 10,  5, True),
        ( 8,  2, 10, False), # disabled
        (11,  3, 10, True),
    ]
))

def test_serialization():
    s = sample.serialize()
    x = Individual.deserialize(s)

    assert sample.genes == x.genes


def test_gene_expression():
    sample.express()

    expected_weight_matrix = np.array([
        # Inputs + Bias
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        # Hidden nodes
        [0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1]
    ], dtype=np.float)


    assert np.all(sample.network.weight_matrix == expected_weight_matrix)
