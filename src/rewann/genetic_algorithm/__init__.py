"""Implementation of an evolutionary algorithm"""

from itertools import count

import numpy as np

from .genetic_operations import mutation
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
        self.edge_counter = count(1)  # 0 are initial edges
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


class GeneticAlgorithm:
    population = None
    hall_of_fame = None

    def __init__(self, env):
        self.env = env

        # first hidden id after ins, bias, & outs
        self.innov = InnovationRecord.empty(env.task.n_in + 1 + env.task.n_out)

    def ask(self):
        if self.population is None:
            self.create_initial_pop()
        else:
            self.evolve_population()

        return self.population

    def tell(self, obj_values):
        self.rank_population(obj_values)

    def rank_population(self, obj_values):
        order = rank_individuals(self.population, obj_values, return_order=True)
        self.population = [self.population[i] for i in order]

    def create_initial_pop(self):
        """Create initial population based on parameters."""
        env = self.env

        initial = env['population', 'initial_genes']

        if initial == 'empty':
            base_ind = env.ind_class.empty_initial(env.task.n_in, env.task.n_out)
            self.population = [base_ind]*env['population', 'size']

        elif initial == 'full':
            self.population = list()
            prob_enabled = env['population', 'initial_enabled_edge_probability']
            for i in range(env['population', 'size']):
                self.population.append(
                    env.ind_class.full_initial(
                        env.task.n_in, env.task.n_out,
                        id=i, prob_enabled=prob_enabled,  # disable some edges
                        negative_edges_allowed=env['population', 'enable_edge_signs']))

        else:
            raise RuntimeError(f'Unknown initial genes type {initial}')

    def evolve_population(self):
        """Apply tournaments if enabled, and mutate surivers.

        Population is assumed to be ordered.
        """
        env = self.env
        pop = self.population

        pop_size = env['population', 'size']
        culling_size = int(np.floor(env['selection', 'culling_ratio'] * pop_size))

        # Elitism (best `elite_size` individual surive without mutation)
        new_pop = pop[:env.elite_size]

        num_places_left = pop_size - len(new_pop)
        winner = None

        if env['selection', 'use_tournaments']:
            participants = np.random.randint(
                len(pop) - culling_size,  # last `culling_size` individuals are ignored
                size=(num_places_left, env['selection', 'tournament_size']))

            winner = np.argmin(participants, axis=-1)

        else:
            winner = np.arange(num_places_left)

        # Breed child population
        for i in winner:
            # Mutation only: take only fittest parent
            child = self.mutate(pop[i])

            assert child is not None
            new_pop.append(child)

        self.population = new_pop

    def mutate(self, individual):
        return mutation(individual, self.env, self.innov)
