from itertools import count
import numpy as np
import logging
from rewann import Individual

def get_objective_values(ind : Individual, objectives):
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

    measurements = ind.measurements(*metric_names, as_list=True)

    return [s*v for v,s in zip(measurements, np.sign(signs))]

def rank_individuals(population, objectives, return_order=False):
    """Rank individuals by multiple objectives using NSGA-sort.

    Parameters
    ----------
    population : List[rewann.Individual]
        List of individuals to rank.
    objectives : tuple
        Tuple (metric_names, sign) that specifies the objectives.
        `metric_names` is a list of the measurements to get, sign specifies
        whether the objective is to be maximized (positive) or minimized
        (negative).
    return_order : bool, optional
        Return a ranked ordering instead of returning the a rank for each individual.

    Returns
    -------
    numpy.ndarray
        Depending on whether return_order is set, the function returns a ranking or
        an ordering according to the objectives.
    """
    try:
        values = [get_objective_values(ind, objectives) for ind in population]
        objectives = np.array(values, dtype=float)
    except Exception as e:
        logging.warning(values)
        logging.warning(population[0]._measurements)
        raise e


    # compute fronts
    front_list = []
    ix = np.arange(len(population))

    # if dm[i, j], then i dominates j
    domination_matrix = dominates(objectives, *np.meshgrid(ix, ix, indexing='ij'))

    unassigned = np.ones(len(population), dtype=bool)

    front_counter = count(1)


    while np.any(unassigned):
        # all individuals that aren't dominated (except of indivs in prior fronts)
        front, = np.where((np.sum(domination_matrix, axis=0) == 0) & unassigned)

        # remove front from domination matrix
        domination_matrix[front, :] = False

        # mark individual in front as assigned
        unassigned[front] = False

        front_list.append(front)

    # sort within fronts

    for front_index, front in enumerate(front_list):
        # caculate crowding_distance
        dist = np.sum(crowding_distances(objectives[front]), axis=-1)
        # sort
        front_list[front_index] = front[np.argsort(-dist)]

        # store front_index for later inspection
        for i in front_list[front_index]:
            population[i].front = front_index


    order = np.hstack(front_list)
    if return_order:
        return order
    else:
        rank = np.empty(len(population), dtype=int)
        rank[order] = ix
        return rank

def dominates(objectives : np.ndarray, i, j):
    """Pareto dominance

    :math:`i` dominates :math:`j` if it is just as good as :math:`j` in all
    objective and at least slightly better in one.

    Parameters
    ----------
    objectives : np.ndarray
        Signed objective measurements for the individuals in the population.
    i
        Index (or indices if `i` is numpy.ndarray) of individual(s) :math:`i`.
    j
        Index (or indices if `j` is numpy.ndarray) of individual(s) :math:`j`.
    """
    return np.all(objectives[i] >= objectives[j], axis=-1) & np.any(objectives[i] > objectives[j], axis=-1)

def crowding_distances(front_objectives):
    """Calculate the crowding distance."""
    # Order by objective value
    n_inds, n_objs = front_objectives.shape


    key = np.argsort(front_objectives, axis=0)
    obj_key = np.arange(n_objs)

    sorted_obj = np.empty((n_inds + 2, n_objs), dtype=float)

    sorted_obj[[0, -1]] = np.inf  # set bounds to inf
    sorted_obj[1:-1] = front_objectives[key, obj_key]

    #warnings.filterwarnings("ignore", category=RuntimeWarning) # inf on purpose

    prevDist = np.abs(sorted_obj[1:-1]-sorted_obj[:-2])
    nextDist = np.abs(sorted_obj[1:-1]-sorted_obj[2:])

    crowd = prevDist+nextDist

    max, min = sorted_obj[-2], sorted_obj[1]

    objs_to_normalize = (max != min)

    crowd[:, objs_to_normalize] = crowd[:, objs_to_normalize] / (max-min)[objs_to_normalize]

    # Restore original order
    dist = np.empty(front_objectives.shape)
    dist[key, obj_key] = crowd[...]

    return dist
