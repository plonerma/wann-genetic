import numpy as np
import json

from wann_genetic import Individual

sample = Individual(genes=Individual.Genotype(
    n_in=3,
    n_out=2,
    nodes=[
        # id, type, activation function
        (4, True, 0),
        (5, True, 0),
        (7, False, 0),
        (10, False, 0),
        (12, False, 0),
    ],
    edges=[
        # innovation id, src, dest, sign, enabled
        ( 3, 12,  7, 1, True), # not necessarily ordered
        ( 1,  1, 12,-1, True),
        ( 0,  0, 12, 1, True),
        ( 2, 12,  4, 1, True),
        ( 4,  7,  4, 1, True),
        (15,  2,  7, 1, True),
        ( 6,  7, 10, 1, True),
        ( 7, 10,  5,-1, True),
        ( 8,  2, 10, 1, False), # disabled
        (11,  3, 10,-1, True),
    ]
))

def test_gene_expression():
    sample.express()

    expected_weight_matrix = np.array([
        # Inputs + Bias      (ids)
        [ 1, 0, 0, 0, 0 ],  #  0
        [-1, 0, 0, 0, 0 ],  #  1
        [ 0, 1, 0, 0, 0 ],  #  2
        [ 0, 0,-1, 0, 0 ],  #  3
        # Hidden nodes
        [ 0, 1, 0, 1, 0 ],  # 12
        [ 0, 0, 1, 1, 0 ],  #  7
        [ 0, 0, 0, 0,-1 ],  # 10

        #12  7 10  4  5 (ids)

    ], dtype=np.float)


    print(expected_weight_matrix.shape)
    print(expected_weight_matrix)
    print(sample.network.weight_matrix.shape)
    print(sample.network.weight_matrix)

    assert np.all(sample.network.weight_matrix == expected_weight_matrix)
