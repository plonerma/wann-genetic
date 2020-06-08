from rewann.individual import IndividualBase


class Individual(IndividualBase):
    from .ffnn import Network as Phenotype


class RecurrentIndividual(IndividualBase):
    from .rnn import Network as Phenotype
