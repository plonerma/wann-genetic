import numpy as np
from rewann.util import get_array_field


def rearrange_matrix(m, indices):
    """Rearrange matrix `m` according to provided indices."""
    # rearrange
    i_rows, i_cols = indices
    m = m[i_rows, :]
    m = m[:, i_cols]
    return m


def num_used_activation_functions(nodes, available_funcs):
    prefix = 'n_nodes_with_act_func_'

    values = dict()
    for func in available_funcs:
        name = prefix + func[0]
        values[name] = 0

    unique, counts = np.unique(nodes['func'], return_counts=True)
    for func_id, num in zip(unique, counts):
        name = prefix + available_funcs[func_id][0]
        values[name] = num

    return values
