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
    evaluate_population(env, population)

    # first hidden id after ins, bias, & outs
    h = n_in + n_out + 1

    innov = InnovationRecord.empty(h)

    env.log.debug('Created initial population.')

    while True:
        # evolve & evaluate
        population = evolve_population(env, population, innov)
        #env.log.debug('Evolved population.')

        evaluate_population(env, population)
        #env.log.debug('Evaluated population.')

        # yield next generation
        yield population

def evaluate_population(env, population):
    for ind in population:
        ind.evaluation(env)

def evolve_population(env, pop, innov):
    #env.log.debug(f"Evolving population.")
    # all individuals need to be evaluated to evolve to next generation

    assert all(i.performance is not None for i in pop)

    new_pop = list()

    # most basic version for now
    ind_performances = np.array([
        i.performance.accuracy for i in pop
    ])

    rank = np.argsort(-ind_performances)

    n_new_children = env['population', 'size']

    # Elitism
    for i in range(env['population', 'elite_size']):
        if i >= len(pop):
            i %= len(pop)
        new_pop.append(pop[i])

    n_new_children -= env['population', 'elite_size']

    # Tournament selection
    pot_parents = np.random.randint(len(pop),
        size=(n_new_children, env['population', 'tournament_size']))

    pot_parents_fitness = ind_performances[pot_parents]

    # Breed child population
    for i in range(n_new_children):
        # Mutation only: take only highest fit parent
        j = np.argmax(pot_parents_fitness[i, :])
        parent = pot_parents[i, j]
        child = pop[parent].mutation(env, innov)
        new_pop.append(child)
        assert child is not None

    return new_pop
