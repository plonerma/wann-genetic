from wann_genetic.individual import IndividualBase, RecurrentIndividualBase


class Individual(IndividualBase):
    from .ffnn import Network as Phenotype


class RecurrentIndividual(RecurrentIndividualBase):
    from .rnn import Network as Phenotype
