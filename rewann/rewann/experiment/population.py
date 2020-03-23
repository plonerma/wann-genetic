import numpy as np
from itertools import count

from ..tasks import Performance

class Population:
    """Store information about current set of individuals.

    When serialized can be thought of as a generation.
    """

    from ..individual import Individual

    def __init__(self, individuals=None, performance=None):
        self.individuals = individuals
        self.performance = performance

        max_node_id = -1
        max_edge_id = -1
        if individuals:
            for i in individuals:
                if len(i.genes.nodes):
                    max_node_id = np.max([np.max(i.genes.nodes['id']), max_node_id])
                if len(i.genes.edges):
                    max_edge_id = np.max([np.max(i.genes.edges['id']), max_edge_id])

        self.node_counter = count(max_node_id + 1)
        self.edge_counter = count(max_edge_id + 1)

    def populate(self, inds):
        if self.individuals is None:
            self.individuals = list()
        self.individuals += inds

    def evaluate(self, task):
        if self.performance is None:
            p = Performance(task.n_classes)
            for ind in self.individuals:
                ind_p = ind.evaluation(task)
                p += ind_p

            self.performance = p
            self.performance.normalize()

    def next_node_id(self):
        return next(self.node_counter)

    def next_edge_id(self):
        return next(self.edge_counter)

    @classmethod
    def initial(cls, n_in, n_out, params):
        pop = cls(individuals=[cls.Individual.base(n_in, n_out)])
        return pop

    def evolve(self, params):
        new_pop = list()
        cur_pop = self.individuals

        assert all(i.performance is not None for i in self.individuals)


        # most basic version for now
        ind_performances = np.array([
            i.performance.accuracy for i in self.individuals
        ])

        rank = np.argsort(-ind_performances)

        n_new_children = params['population', 'size']

        # Elitism
        for i in range(params['population', 'elite_size']):
            if i >= len(cur_pop):
                i %= len(cur_pop)
            new_pop.append(cur_pop[i])

        n_new_children -= params['population', 'elite_size']

        # Tournament selection
        pot_parents = np.random.randint(len(cur_pop),
            size=(n_new_children, params['population', 'tournament_size']))

        pot_parents_fitness = ind_performances[pot_parents]

        # Breed child population
        for i in range(n_new_children):
            # Mutation only: take only highest fit parent
            j = np.argmax(pot_parents_fitness[i, :])
            parent = pot_parents[i, j]
            child = cur_pop[parent].mutation(self, params)
            new_pop.append(child)
            assert child is not None

        # replace individuals in population
        self.individuals = new_pop
        self.performance = None
