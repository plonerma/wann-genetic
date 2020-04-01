import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from ..util import deserialize_array, serialize_array

from .metrics import apply_metrics

import logging

def expressed(func):
    def exp_func(self, *args, **kwargs):
        self.express()
        return func(self, *args, **kwargs)
    return exp_func

class Individual:
    """ Collection of representations of an individual.

    Will contain genes (genotype), network (phenotype), and performance statistics (fitness).
    """

    genes = None
    network = None
    prediction_records = None

    from .genes import Genotype
    from .network import Network

    from .genetic_operations import mutation

    def __init__(self, genes=None, network=None, prediction_records=None, id=None, birth=None):
        self.genes = genes
        self.network = network
        self.prediction_records = prediction_records
        self.id = id
        self.birth = birth

    # Translations

    def express(self):
        """Translate genes to network."""
        if self.network is None:
            assert self.genes is not None
            self.network = self.Network.from_genes(self.genes)

    @expressed
    def apply(self, *args, **keargs):
        return self.network.apply(*args, **keargs)

    @expressed
    def evaluate(self, env):
        if self.prediction_records is None:
            # stack for storing confusion matrices
            cm_list = list()

            # weights used during evaluations
            weight_list = list()
        else:
            cm_list, weight_list = self.prediction_records

        shared_weight = env['sampling', 'current_weight']

        y_pred=self.apply(env.task.x, func='argmax', w=shared_weight)

        cm = confusion_matrix(env.task.y_true, y_pred,
                              labels=list(range(env.task.n_out)),
                              normalize='all')

        cm_list.append(cm)
        weight_list.append(shared_weight)

        self.prediction_records = cm_list, weight_list

    @property
    def fitness(self):
        fitness_metric = 'median:accuracy'
        f = self.get_prediction_metrics(fitness_metric)[fitness_metric]
        return f

    @expressed
    def metrics(self, *metrics, current_gen=None):
        if not metrics:
            metrics = ('max:kappa', 'mean:kappa', 'min:kappa',
                       'n_hidden', 'n_edges')

        metrics = list(metrics)

        ind_metrics = dict(
            n_hidden=self.network.n_hidden,
            n_edges=len(self.genes.edges),
            n_evaluations=len(self.prediction_records[0]),
        )
        if current_gen is not None:
            ind_metrics['age'] = current_gen - self.birth
        m = dict()
        for k, v in ind_metrics.items():
            if k in metrics:
                metrics.remove(k)
                m[k] = v

        pm = self.get_prediction_metrics(*metrics)
        for k in metrics:
            m[k] = pm[k]
        return m

    def get_prediction_metrics(self, *metrics):
        cm_list, weight_list = data=self.prediction_records
        values = dict(
            cm_stack=np.array(cm_list), weights=np.array([weight_list])
        )
        return apply_metrics(values, metrics)

    @classmethod
    def base(cls, *args, **kwargs):
        return cls(genes=cls.Genotype.base(*args, **kwargs), birth=0, id=0)

    # Serialization

    def serialize(self, include_prediction_records=False):
        d = dict(
            genes=self.genes.serialize(),
            birth=self.birth, id=self.id)
        if include_prediction_records and self.prediction_records:
            cm_list, weight_list = data=self.prediction_records
            d['record'] = dict(
                n_classes=cm_list[0].shape[0],
                cm_stack=[serialize_array(cm) for cm in cm_list],
                weights=weight_list,
            )
        return d

    @classmethod
    def deserialize(cls, d : dict):
        p = dict(genes=cls.Genotype.deserialize(d['genes']))
        if 'record' in d:
            cm_list = list()
            n_classes = d['record']['n_classes']
            for cm in d['record']['cm_stack']:
                cm = deserialize_array(d['record']['cm_stack'], dtype=int)
                s = cm.shape[0]
                cm.reshape((n_classes, n_classes))
                cm_list.append(cm)
            weights = d['record']['weights']

            p['prediction_records'] = cm_list, weights
        return cls(**p)
