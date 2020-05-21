"""Implementation of an evolutionary algorithm"""

import numpy as np
from itertools import count
from functools import partial
import logging

from .ranking import rank_individuals

class InnovationRecord(set):
    """Keeps track of edge and node counts.

    Edge ids need to be tracked if crossover should ever be implemented.
    """
    @classmethod
    def empty(cls, start_id):
        """Initialize empty innovation record.

        start_id : ids of new nodes (hidden) start at start_id.
        """
        self = cls()
        self.node_counter = count(start_id)
        self.edge_counter = count(1) # 0 are initial edges
        self.individual_counter = count(0)
        self.generation = 0
        return self

    def next_ind_id(self):
        return next(self.individual_counter)

    def next_edge_id(self):
        return next(self.edge_counter)

    def next_node_id(self):
        return next(self.node_counter)

    def edge_exists(self, src, dest):
        return (src, dest) in self

    def add_edge(self, src, dest, **args):
        # disregard args for now
        # potentially switch to dicts and store args as value
        self.add((src, dest))

def apply_networks(params, x):
    network, weights = params
    return network.apply(x=x, weights=weights, func='raw')


def evaluate_inds(env, pop, n_samples=-1, record_raw=False,
                  use_test_samples=False):
    """Use the process pool to evaluate a list of individuals.

    Parameters
    ----------
    env : rewann.Environment
        environment to use for process pool, weight sampling, and task data
    pop : list
        list of individuals to evaluate
    n_samples : int
        number of samples to use (-1 for all -> should not be used with
        synthetic tasks)
    record_raw : bool
        use the record_raw parameter in Individual class (if true, stores all
        predictions instead of calculating recoreded measures online)
    use_test_samples : bool
        use samples dedicated for testing (option is passed to tasks `get_data`
        function)
    """

    x, y_true = env.task.get_data(test=use_test_samples, samples=n_samples)

    apply_func=partial(apply_networks, x=x)

    #logging.debug('expressing individuals')

    express_inds(env, pop)

    weights = env['sampling', 'current_weight']

    #logging.debug(f'Applying networks ({n_samples})')

    results = env.pool_map(apply_func, [(ind.network, weights) for ind in pop])

    #logging.debug('recording metrics')

    for y_raw, ind in zip(results, pop):
        ind.record_measurements(
            weights=weights,
            y_true=y_true, y_raw=y_raw,
            record_raw=record_raw)

def express_inds(env, pop):
    """Express inds that have not been expressed yet.

    Convert genes into neural network for all new individuals.


    Parameters
    ----------
    env : rewann.Environment
    pop : list
    """
    inds_to_express = list(filter(lambda i: i.network is None, pop))

    networks = env.pool_map(env.ind_class.Phenotype.from_genes, (i.genes for i in inds_to_express))
    for ind, net in zip(inds_to_express, networks):
        ind.network = net

def create_initial_population(env):
    """Create initial population based on parameters in `env`.

    Parameters
    ----------
    env : rewann.Environment
    """
    initial = env['population', 'initial_genes']

    if initial == 'empty':
        base_ind = env.ind_class.empty_initial(env.task.n_in, env.task.n_out)
        express_inds(env, [base_ind])
        return [base_ind]*env['population', 'size']

    elif initial == 'full':
        pop = list()
        prob_enabled = env['population', 'initial_enabled_edge_probability']
        for i in range(env['population', 'size']):
            pop.append(
                env.ind_class.full_initial(env.task.n_in, env.task.n_out,
                    id=i, prob_enabled=prob_enabled,
                    negative_edges_allowed=env['population', 'enable_edge_signs']))
            # disable some edges
        express_inds(env, pop)
        evaluate_inds(env, pop, n_samples=env['sampling', 'num_training_samples_per_iteration'])
        order = rank_individuals(pop, objectives=env.objectives, return_order=True)
        pop = [pop[i] for i in order]
        return pop
    else:
        raise RuntimeError(f'Unknown initial genes type {initial}')

def evolution(env):
    """Main function for evolutionary algorithm.

    Parameters
    ----------
    env : rewann.Environment
    """
    pop = create_initial_population(env)

    if env['sampling', 'post_init_seed'] is not False:
        env.seed(env['sampling', 'post_init_seed'])

    # first hidden id after ins, bias, & outs
    innov = InnovationRecord.empty(env.task.n_in + 1 + env.task.n_out)

    logging.debug('Created initial population.')

    while True:
        innov.generation += 1

        logging.debug('evolving next generation')

        pop = evolve_population(env, pop, innov)

        evaluate_inds(env, pop, n_samples=env['sampling', 'num_training_samples_per_iteration'])
        order = rank_individuals(pop, objectives=env.objectives, return_order=True)
        pop = [pop[i] for i in order]

        # yield next generation
        yield innov.generation, pop

def evolve_population(env, pop, innov):
    """Rank population, apply tournaments if enabled, and mutate surivers.

    Parameters
    ----------
    env : rewann.Environment
        Environment to use for params
    pop : list
        current population
    innov: InnovationRecord
        used for tracking ids etc.
    """
    pop_size = env['population', 'size']
    culling_size = int(np.floor(env['selection', 'culling_ratio'] * pop_size))

    # population is assumed to be ordered

    # Elitism (best `elite_size` individual surive without mutation)
    new_pop = pop[:env.elite_size]

    num_places_left = pop_size - len(new_pop)
    winner = None

    if env['selection', 'use_tournaments']:
        participants = np.random.randint(
            len(pop) - culling_size, # last `culling_size` individuals are ignored
            size=(num_places_left, env['selection', 'tournament_size']))

        winner = np.argmin(participants, axis=-1)

    else:
        winner = np.arange(num_places_left)

    # Breed child population
    for i in winner:
        # Mutation only: take only fittest parent
        child = pop[i].mutation(env, innov)

        assert child is not None
        new_pop.append(child)

    return new_pop

def update_hof(env, pop):
    """Update the hall of fame.

    Parameters
    ----------
    env : rewann.Environment
    pop : iterable
    """
    # best inds excluding indivs alread in hall of fame
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
        max(0, n_evals - int(ind.measurements('n_evaluations') / eval_size))
        for ind in elite
    ]

    for i in range(max(required_evaluations)):
        inds = [
            ind for ind, revs in zip(elite, required_evaluations)
            if revs > i
        ]
        env.sample_weights()
        evaluate_inds(env, inds, n_samples=env['sampling', 'num_training_samples_per_iteration'])


    candidates = env.hall_of_fame + elite

    if len(candidates) <= hof_size:
        env.hall_of_fame = candidates

    else:
        # sort candidates
        scores = np.array([metric_sign * ind.measurements(metric) for ind in candidates])

        env.hall_of_fame = [
            candidates[i] for i in np.argsort(-scores)[:hof_size]
        ]
    return env.hall_of_fame
