from functools import partial
import numpy as np
import logging


def update_hall_of_fame(env, pop):
    """Update the hall of fame."""
    if not env.hall_of_fame:
        elite = pop[:env.elite_size]
    else:
        # best inds in pop (excluding those already in hof)
        elite = list()

        inds = iter(pop)
        while len(elite) < env.elite_size:
            i = next(inds)
            if i not in env.hall_of_fame:
                elite.append(i)

    hof_size = env['population', 'hof_size']

    metric, metric_sign = env.hof_metric

    # make sure elite is properly evaluated

    n_evals = env['sampling', 'hof_evaluation_iterations']
    eval_size = env['sampling', 'num_weight_samples_per_iteration']

    required_evaluations = [
        max(0, n_evals - int(ind.measurements['n_evaluations'] / eval_size))
        for ind in elite
    ]

    for i in range(max(required_evaluations)):
        inds = [
            ind for ind, revs in zip(elite, required_evaluations)
            if revs > i
        ]
        weights = env.sample_weights()
        make_measurements(env, inds, weights=weights)


    candidates = env.hall_of_fame + elite

    if len(candidates) <= hof_size:
        env.hall_of_fame = candidates

    else:
        # sort candidates
        scores = np.array([metric_sign * ind.get_data(metric) for ind in candidates])

        env.hall_of_fame = [
            candidates[i] for i in np.argsort(-scores)[:hof_size]
        ]
    return env.hall_of_fame

def make_measurements(env, pop, weights):
    measurements = evaluate_inds(env, pop, weights, measures=env['selection', 'recorded_metrics'])
    for ind, measurements in zip(pop, measurements):
        ind.record_measurements(weights=weights, measurements=measurements)

def evaluate_inds(env, pop, weights, test=False, measures=['log_loss'], n_samples=None):
    """Use the process pool to evaluate a list of individuals.

    Parameters
    ----------
    env : rewann.Environment
        Environment to use for process pool, weight sampling, and task data.
    pop : list
        List of individuals to evaluate
    weights : np.ndarray
        Sampled weights.
    test : bool.optional
        If true, test sampels are used.
    measures : [str]
        Which measurements to get from the network.
    n_samples : int, optional
        How many samples to use. Defaults to sampling/num_training_samples_per_iteration

    Returns
    --------
    dict
        Dictionary containing the measurements that were made.
    """

    express_inds(env, pop)

    if n_samples is None:
        n_samples = env['sampling', 'num_training_samples_per_iteration']

    x, y_true = env.task.get_data(test=test, samples=n_samples)

    return env.pool_map((
        lambda network: network.get_measurements(
            weights=weights,
            x=x, y_true=y_true,
            measures=measures
        )), pop)


def express_inds(env, pop):
    """Express inds that have not been expressed yet.

    Convert genes into neural network for all new individuals.


    Parameters
    ----------
    env : rewann.Environment
    pop : [rewann.Individual]
    """
    inds_to_express = list(filter(lambda i: i.network is None, pop))

    networks = env.pool_map(env.ind_class.Phenotype.from_genes, (i.genes for i in inds_to_express))
    for ind, net in zip(inds_to_express, networks):
        ind.network = net


def get_objective_values(ind, objectives):
    """Get measurements of an individual for the specified objectives.

    Parameters
    ----------
    ind : rewann.Individual
        The individual to get the measurements for.
    objectives : tuple
        Tuple (metric_names, sign) that specifies the objectives.
        `metric_names` is a list of the measurements to get, sign specifies
        whether the objective is to be maximized (positive) or minimized
        (negative).

    Returns
    -------
    np.ndarray
        Signed measurements for the individual.
    """
    metric_names, signs = objectives

    return [
        s*m for s,m in zip(signs, ind.get_data(*metric_names, as_list=True))
    ]
