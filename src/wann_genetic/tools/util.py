from tabulate import tabulate
import logging


def get_initial_population(p):
    v = p('population', 'initial_enabled_edge_prob')  # store first, so it has been accessed
    if p('population', 'initial_genes') == 'full':
        return f"fully connected\n(edges have {v:0.0%} probability of being enabled)"
    else:
        return "without any connections"

def get_tournament_size(p):
    v = p('selection', 'tournament_size')
    if p('selection', 'use_tournaments'):
        return v


default_table_rows = {
    # population
    'Population size': ('population', 'size'),
    'Number of generations': ('population', 'num_generations'),
    'Initial population': get_initial_population,
    'Size of the hall of fame': ('population', 'hof_size'),

    # selection
    'Elite ratio\n(ratio of individuals to surive without mutation)': lambda p: f"{p('selection', 'elite_ratio'):0.0%}",
    'Culling ratio\n(ratio of individuals to exclude from selection)': lambda p: f"{p('selection', 'culling_ratio'):0.0%}",
    'Number of individuals in a tournament': get_tournament_size,

    # mutations
    'New edge mutation strategy': ('mutation', 'new_edge', 'strategy'),

    # sampling
    'Number of weights per generation': ('sampling', 'num_weights_per_iteration'),
    'Number of training samples per generation': lambda p: p('sampling', 'num_samples_per_iteration') if p('sampling', 'num_samples_per_iteration')>0 else 'all',
    'Number of evaluation iterations required to enter hall of fame\n(if to few, individual will be more thoroughly evaluted)': ('sampling', 'hof_evaluation_iterations'),
}

default_ignored_keys = [
    ('config', None), # None : catch-all
    ('postopt', None),
    ('storage', None),
    ('debug', ),
    ('sampling', 'seed'),
    ('sampling', 'post_init_seed'),
]

default_headers = ['Parameter', 'Value']

def hyperparam_table(params, tablefmt='grid', table_mapping=default_table_rows, headers=default_headers, ignore_unused_keys=False, ignored_keys=default_ignored_keys):
    accessed_keys = set()

    def access_params(*key):
        accessed_keys.add(key)
        return params[key]

    table = list()
    for name, key in table_mapping.items():
        if callable(key):
            value = key(access_params)
        else:
            value = access_params(*key)

        if value is not None:
            table.append([name, value])

    table = tabulate(table, tablefmt=tablefmt, headers=headers)

    if not ignore_unused_keys:
        for k in params.nested_keys():
            if k not in accessed_keys:
                ignored = False

                for ignored_k in ignored_keys:
                    if ((k == ignored_k)  # key is listed explicitly
                        # key is caught by catch-all ignore key
                        or (ignored_k[-1] is None and ignored_k[:-1] == k[:len(ignored_k)-1])):
                        ignored = True
                        break

                if not ignored:
                    logging.warning(f'Key {k} was not used.')
    return table
