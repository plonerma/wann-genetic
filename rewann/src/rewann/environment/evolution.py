""" Reimplementation of population stuff -> don't use seperate class, but build
    iterator (iterating over the generations)."""

import numpy as np
from itertools import count
from functools import partial
import logging

from .ranking import rank_individuals

class InnovationRecord(set):
    @classmethod
    def empty(cls, start_id):
        """Initialize empty innovation record.

        start_id : ids of new nodes (hidden) start at start_id.
        """
        self = cls()
        self.node_counter = count(start_id)
        self.edge_counter = count(0)
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

def evaluate_inds(env, pop, n_samples=-1, reduce_values=True):
    if n_samples < 0 or n_samples > len(env.task.x):
        x = env.task.x
        y_true = env.task.y_true
    else:
        chosen_samples = np.random.choice(len(env.task.x), size=n_samples, replace=False)
        x = env.task.x[chosen_samples]
        y_true = env.task.y_true[chosen_samples]

    apply_func=partial(apply_networks, x=x)

    logging.debug('expressing individuals')

    express_inds(env, pop)

    weights = env['sampling', 'current_weight']

    logging.debug(f'Applying networks ({n_samples})')

    results = env.pool_map(apply_func, [(ind.network, weights) for ind in pop])

    logging.debug('recording metrics')

    for y_probs, ind in zip(results, pop):
        ind.record_metrics(weights, y_true, y_probs, reduce_values=reduce_values)

def express_inds(env, pop):
    inds_to_express = list(filter(lambda i: i.network is None, pop))
    networks = env.pool_map(env.ind_class.Network.from_genes, (i.genes for i in inds_to_express))
    for ind, net in zip(inds_to_express, networks):
        ind.network = net

def apply_networks(params, x):
    network, weights = params
    return network.apply(x=x, weights=weights)

def create_initial_population(env):
    if not env['population', 'initial_with_edges']:
        base_ind = env.ind_class.base(n_in, n_out)
        express_inds(env, [base_ind])
        return [base_ind]*env['population', 'size']

    else:
        pop = list()
        prob_enabled = env['population', 'initial_enabled_edge_probability']
        for i in range(env['population', 'size']):
            pop.append(
                env.ind_class.full_initial(env.task.n_in, env.task.n_out,
                                           id=i, prob_enabled=prob_enabled))
            # disable some edges
        express_inds(env, pop)
        evaluate_inds(env, pop, n_samples=env['sampling', 'num_training_samples_per_iteration'])
        order = rank_individuals(pop, return_order=True)
        pop = [pop[i] for i in order]
        return pop

def evolution(env):
    pop = create_initial_population(env)

    # first hidden id after ins, bias, & outs
    innov = InnovationRecord.empty(env.task.n_in + 1 + env.task.n_out)

    logging.debug('Created initial population.')

    while True:
        innov.generation += 1

        logging.debug('evolving next generation')

        pop = evolve_population(env, pop, innov)

        evaluate_inds(env, pop, n_samples=env['sampling', 'num_training_samples_per_iteration'])
        order = rank_individuals(pop, return_order=True)
        logging.debug(order)
        pop = [pop[i] for i in order]

        # yield next generation
        yield innov.generation, pop

def evolve_population(env, pop, innov):
    pop_size = env['population', 'size']
    elite_size = int(np.floor(env['selection', 'elite_ratio'] * pop_size))
    culling_size = int(np.floor(env['selection', 'culling_ratio'] * pop_size))

    # population is assumed to be ordered

    # Elitism (best `elite_size` individual surive without mutation)
    new_pop = pop[:elite_size]

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
