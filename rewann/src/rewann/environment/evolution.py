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

    population = [ind_class.base(n_in, n_out)]

    for ind in population:
        ind.evaluate(env)

    # first hidden id after ins, bias, & outs
    h = n_in + n_out + 1

    innov = InnovationRecord.empty(h)

    env.log.debug('Created initial population.')

    while True:
        # evolve & evaluate
        population = evolve_population(env, population, innov)
        #env.log.debug('Evolved population.')

        for ind in population:
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

    n_new_children = env['population', 'size']

    # Elitism
    for i in range(env['population', 'elite_size']):
        new_pop.append(pop[i % len(pop)])

    n_new_children -= env['population', 'elite_size']

    # Tournament selection
    pot_parents = np.random.randint(len(pop),
        size=(n_new_children, env['population', 'tournament_size']))

    pot_parents_fitness = fitness[pot_parents]

    # Breed child population
    for i in range(n_new_children):
        # Mutation only: take only highest fit parent
        j = np.argmax(pot_parents_fitness[i, :])
        parent = pot_parents[i, j]
        child = pop[parent].mutation(env, innov)
        assert child is not None
        new_pop.append(child)

    return new_pop
