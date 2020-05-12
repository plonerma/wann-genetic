from rewann.individual import RecurrentIndividual as Ind
import numpy as np
import logging

def test_ff():
    i = Ind(genes=Ind.Genotype(
        n_in=1,
        n_out=2,
        nodes=[ # one in, one bias, one out

            # id, out, activation function
            (2, True, 0), # active if in is active
            (3, True, 0), # active if in is not active
        ],
        edges=[
            # id, src, dst, sgn, enabled, recurrent
            (  0,   0,   2,   1, True   , False),
            (  1,   0,   3,  -1, True   , False),
            (  2,   1,   3,   1, True   , False),
        ]
    ))

    i.express()
    inputs = np.array([
        # sequences, sequence elements, values
            [[0], [1]],
            [[1], [0]]
        ])

    outputs = i.apply(inputs, weights=np.array([1]), func='argmax')
    expected_outputs = np.array([[[1, 0], [0, 1]]])
    valid = expected_outputs >= 0


    assert np.all(
        outputs[valid] == expected_outputs[valid]
    )


def test_re():
        i = Ind(genes=Ind.Genotype(
            n_in=1,
            n_out=2,
            nodes=[ # one in, one bias, one out

                # id, out, activation function
                (2, True, 0), # same as ff with one timestep delay
                (3, True, 0),
            ],
            edges=[
                # id, src, dst, sgn, enabled, recurrent
                (  0,   0,   2,   1, True   , True),
                (  1,   0,   3,  -1, True   , True),
                (  2,   1,   3,   1, True   , False),
            ]
        ))

        i.express()
        inputs = np.array([
            # sequences, sequence elements, values
                [[0], [1]],
                [[1], [0]]
            ])

        outputs = i.apply(inputs, weights=np.array([1]), func='argmax')
        expected_outputs = np.array([[[-1, 0], [-1, 1]]])
        valid = expected_outputs >= 0

        logging.debug(i.apply(inputs, weights=np.array([1]), func='none'))

        logging.debug(outputs[valid])

        logging.debug(expected_outputs[valid])

        assert np.all(
            outputs[valid] == expected_outputs[valid]
        )
