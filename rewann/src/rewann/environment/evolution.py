""" Reimplementation of population stuff -> don't use seperate class, but build
    iterator (iterating over the generations)."""

import numpy as np
from itertools import count

from ..individual import Individual

class InnovationRecord(set):
    @classmethod
    def empty(cls, start_id):
        """Initialize empty innovation record.

        start_id : ids of new nodes (hidden) start at start_id.
        """
        self = cls()
        self.node_counter = count(start_id)
        self.edge_counter = count(0)
        return self

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

def evolution(env, ind_class=Individual):
    # initial population
    n_in, n_out = env.task.n_in, env.task.n_out

    base_ind = ind_class.base(n_in, n_out)

    population = [base_ind]*env['population', 'size']

    base_ind.evaluate(env)

    # first hidden id after ins, bias, & outs
    h = n_in + n_out + 1

    innov = InnovationRecord.empty(h)

    env.log.debug('Created initial population.')

    while True:
        # evolve & evaluate
        population = evolve_population(env, population, innov)
        #env.log.debug('Evolved population.')

        # make sure to only evaluate once, even if an individual happens to
        # appear mutiple times
        for ind in set(population):
            ind.evaluate(env)

        # yield next generation
        yield population

def evolve_population(env, pop, innov):

    new_pop = list()

    # most basic version for now
    fitness = np.array([
        i.fitness for i in pop
    ])

    rank = np.argsort(-fitness)

    pop_size = env['population', 'size']
    elite_size = env['population', 'elite_size']
    tournament_size = env['population', 'tournament_size']
    num_tournaments = pop_size - elite_size


    # Elitism
    for i in range(elite_size):
        new_pop.append(pop[rank[i % len(pop)]])

    # Tournament selection
    tournament_participants = np.random.randint(len(pop),
        size=(num_tournaments, tournament_size))

    tournament_scores = fitness[tournament_participants]

    # Breed child population
    for i in range(num_tournaments):
        # Mutation only: take only fittest parent
        j = np.argmax(tournament_scores[i, :])
        winner = tournament_participants[i, j]
        child = pop[winner].mutation(env, innov)
        assert child is not None
        new_pop.append(child)

    return new_pop
