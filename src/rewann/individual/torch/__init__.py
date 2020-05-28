from rewann.individual import IndividualBase


class Individual(IndividualBase):
    from .ffnn import Network as Phenotype
