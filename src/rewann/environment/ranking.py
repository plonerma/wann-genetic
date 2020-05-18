from itertools import count
import numpy as np
import logging

def get_objective_values(ind, objectives):
    metric_names, signs = objectives

    measurements = ind.measurements(*metric_names, as_list=True)

    return [s*v for v,s in zip(measurements, signs)]

def rank_individuals(population, objectives, return_order=False, return_fronts=False):
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

def dominates(objectives, i, j):
    return np.all(objectives[i] >= objectives[j], axis=-1) & np.any(objectives[i] > objectives[j], axis=-1)

def crowding_distances(front_objectives):
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
