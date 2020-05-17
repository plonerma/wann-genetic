import toml
import logging
import numpy as np
import pandas as pd
import os
import subprocess
import logging


from multiprocessing import Pool


from rewann.individual import Individual, RecurrentIndividual

from .tasks import select_task
from .evolution import evolution, update_hof
from .util import get_version, TimeStore
from .params import ParamTree

from rewann.postopt import Report

class Environment(ParamTree):
    from .util import (default_params, setup_params, open_data,
                       store_gen, store_gen_metrics, load_pop, load_gen_metrics,
                       stored_populations, stored_indiv_metrics, store_hof, load_hof,
                       load_indiv_metrics,
                       env_path)

    def __init__(self, params):
        super().__init__()

        self.setup_params(params)

        self.metrics = list()
        self.pool= None
        self.data_file = None

        self.hall_of_fame = list()

        use_test_samples = True

        # choose task
        self.task = select_task(self['task', 'name'])

        # choose adequate type of individuals
        if self.task.is_recurrent:
            self.ind_class = RecurrentIndividual
        else:
            self.ind_class = Individual

        self.ind_class.recorded_metrics = self['selection', 'recorded_metrics']

        # only use enabeld activations functions
        funcs = self.ind_class.Network.available_act_functions
        if self['population', 'enabled_activation_functions'] != 'all':
            self.ind_class.Network.available_act_functions = [
                funcs[i] for i in self['population', 'enabled_activation_functions']
            ]

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self['sampling', 'seed'])

    @property
    def elite_size(self):
        return int(np.floor(self['selection', 'elite_ratio'] * self['population', 'size']))

    def sample_weights(self, n=None):
        if n is None:
            n = self['sampling']['num_weight_samples_per_iteration']

        dist = self['sampling', 'distribution'].lower()

        if dist == 'one':
            w = 1

        elif dist == 'uniform':
            l = self['sampling', 'lower_bound']
            u = self['sampling', 'upper_bound']
            assert l is not None and u is not None

            w = np.random.uniform(l, u, size=n)

        elif dist == 'lognormal':
            m = self['sampling', 'mean']
            s = self['sampling', 'sigma']
            assert m is not None and s is not None

            w = np.random.lognormal(m, s, size=n)

        elif dist == 'normal':
            m = self['sampling', 'mean']
            s = self['sampling', 'sigma']
            assert m is not None and s is not None

            w = np.random.normal(m, s, size=n)

        else:
            raise RuntimeError(f'Distribution {dist} not implemented.')
        self['sampling', 'current_weight'] = w
        return w

    def setup_pool(self, n=None):
        if n is None:
            n = self['config', 'num_workers']
        if n == 1:
            self.pool = None
        else:
            self.pool = Pool(n)

    def pool_map(self, func, iter):
        if self.pool is None:
            return map(func, iter)
        else:
            return self.pool.imap(func, iter)

    def setup_optimization(self):
        log_path = self.env_path(self['storage', 'log_filename'])
        logging.info (f"Check log ('{log_path}') for details.")

        logger = logging.getLogger()

        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        if self['debug']:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logging.info(f"Package version {get_version()}")

        p = os.path.abspath(self['experiment_path'])
        logging.info(f'Saving data at {p}.')

        logging.debug('Loading training dataset')
        self.task.load_training()

        # log used parameters
        params_toml = toml.dumps(self.params)
        logging.debug(f"Running experiments with the following parameters:\n{params_toml}")

        with open(self.env_path('params.toml'), 'w') as f:
            params = dict(self.params)
            params['is_report'] = True # mark stored params as part of a report
            toml.dump(params, f)

        self.seed()


    def run(self):
        # set up logging, write params
        self.setup_optimization()

        # set up pool of workers
        self.setup_pool()

        with self.open_data('w'):
            # run optimization
            self.optimize()

        if self['postopt', 'run_postopt']:
            with self.open_data('r'):
                # evaluate individuals in hall of fame
                self.post_optimization()
        if self.pool is not None:
            logging.info('Closing pool')
            self.pool.close()

    def optimize(self):
        generations = evolution(self)

        logging.info("Starting evolutionary algorithm")

        ts = TimeStore()

        ts.start()
        for _ in range(self['population', 'num_generations']):

            ts.start()

            w = self.sample_weights()
            logging.debug(f'Sampled weight {w}')

            gen, pop = next(generations)

            logging.debug('Updating hall of fame')
            self.hall_of_fame = update_hof(self, pop)

            ts.stop()

            avg = (ts.total / gen)
            expected_time = (self['population', 'num_generations'] - gen) * avg
            logging.info(f'Completed generation {gen}; {ts.dt:.02}s elapsed, {avg:.02}s avg, {ts.total:.02}s total. '
                         f'Expected time remaining: {expected_time:.02}s')

            self.store_data(gen, pop)
        self.last_population = pop
        self.store_hof()

    def post_optimization(self):
        r = Report(self).run_evaluations( # run evaluations on test data
            num_weights=self['postopt', 'num_weights'],
            num_samples=self['postopt', 'num_samples'] # all
        )

        if self['postopt', 'compile_report']:
            r.compile() # plot metrics, derive stats
        else:
            r.compile_stats() # at least derive and store stats

    def store_data(self, gen, pop):
        gen_metrics = self.population_metrics(gen=gen, population=pop)
        gen_metrics, indiv_metrics = self.population_metrics(gen=gen, population=pop, return_indiv_metrics=True)

        metric, metric_sign = self.hof_metric
        p = ("MAX" if metric_sign > 0 else "MIN")
        metric_value = gen_metrics[f"{p}:{metric}"]

        logging.info(f"#{gen} {p}:{metric}: {metric_value:.2}")

        self.metrics.append(gen_metrics)

        commit_freq = self['storage', 'commit_elite_freq']
        if (commit_freq > 0 and gen % commit_freq == 0):
            self.store_gen(gen, population=pop[:self.elite_size], indiv_metrics=indiv_metrics)

        commit_freq = self['storage', 'commit_metrics_freq']
        if (commit_freq > 0 and gen % commit_freq == 0):
            self.store_gen_metrics(pd.DataFrame(data=self.metrics))

    def population_metrics(self, population, gen=None, return_indiv_metrics=False, reduced_values=True):
        if gen is None:
            gen = self['population', 'num_generations']

        base_metrix = ['n_hidden', 'n_enabled_edges', 'n_total_edges',
                       'n_evaluations', 'age', 'front', 'n_layers', 'n_mutations']


        prefixed_metrics = self.ind_class.recorded_metrics


        prefixes = {'max': np.max, 'mean': np.mean, 'min': np.min}

        if reduced_values:
            metric_names = base_metrix + [
                f'{m}.{p}' for p in prefixes for m in prefixed_metrics
            ]
            individual_metrics = pd.DataFrame(data=[
                ind.metrics(*metric_names, current_gen=gen) for ind in population
            ])

        else:
            metric_names = base_metrix + prefixed_metrics
            individual_metrics = [
                ind.metrics(*metric_names, current_gen=gen) for ind in population
            ]
            for im in individual_metrics:
                im.update({
                    f'{pm}.{pf}': pfunc(im[pm])
                    for pm in prefixed_metrics
                    for pf, pfunc in prefixes.items()
                })
                for pm in prefixed_metrics:
                    del im[pm]

            individual_metrics = pd.DataFrame(data=individual_metrics)


        metrics = dict(
            num_unique_individuals=len(set(population)),

            num_individuals=len(population),

            # number of inds without edges
            num_no_edge_inds=np.sum(individual_metrics['n_enabled_edges'] == 0),

            # number of inds without hidden nodes
            num_no_hidden_inds=np.sum(individual_metrics['n_hidden'] == 0),

            # individual with the most occurences
            biggest_ind=max([population.count(i) for i in set(population)]),
        )

        for name, values in individual_metrics.items():
            metrics[f'Q_0:{name}'] = metrics[f'MIN:{name}'] = values.min()
            metrics[f'Q_1:{name}'] = np.quantile(values, .25)
            metrics[f'Q_2:{name}'] = metrics[f'MEDIAN:{name}'] = values.median()
            metrics[f'Q_3:{name}'] = np.quantile(values, .75)
            metrics[f'Q_4:{name}'] = metrics[f'MAX:{name}'] = values.max()

            metrics[f'MEAN:{name}'] = values.mean()
            metrics[f'STD:{name}'] = values.std()

        if return_indiv_metrics:
            return metrics, individual_metrics
        else:
            return metrics
