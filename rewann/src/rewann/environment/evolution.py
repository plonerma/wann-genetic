""" Reimplementation of population stuff -> don't use seperate class, but build
    iterator (iterating over the generations)."""

import numpy as np
from itertools import count
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

def evolution(env):
    # initial population
    n_in, n_out = env.task.n_in, env.task.n_out

    base_ind = env.ind_class.base(n_in, n_out)

    population = [base_ind]*env['population', 'size']

    base_ind.evaluate(env)

    # first hidden id after ins, bias, & outs
    h = n_in + n_out + 1

    innov = InnovationRecord.empty(h)

    logging.debug('Created initial population.')

    while True:
        innov.generation += 1

        # evolve & evaluate
        population = evolve_population(env, population, innov)
        #logging.debug('Evolved population.')

        # make sure to only evaluate once, even if an individual happens to
        # appear mutiple times
        for ind in set(population):
            ind.evaluate(env)

        # yield next generation
        yield innov.generation, population

def evolve_population(env, pop, innov):
    elite_size = env['population', 'elite_size']

    rank = rank_individuals(pop)

    # Elitism (best `elite_size` individual surive without mutation)
    if len (pop) > elite_size:
        elite = np.argpartition(rank, elite_size)
        new_pop = [pop[i] for i in elite[:elite_size]]
    else:
        new_pop = pop

    pop_size = env['population', 'size']
    tournament_size = env['population', 'tournament_size']
    num_tournaments = pop_size - len(new_pop)

    # Tournament selection
    participants = np.random.randint(len(pop), size=(num_tournaments, tournament_size))

    scores = np.empty(participants.shape)

    # simplicity tournament
    score = rank[participants]

    winner = np.argmin(scores, axis=-1)

    # Breed child population
    for i in winner:
        # Mutation only: take only fittest parent
        child = pop[i].mutation(env, innov)

        assert child is not None
        new_pop.append(child)

    return new_pop
