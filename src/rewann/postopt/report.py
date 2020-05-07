import os
from functools import partial
import argparse

from tabulate import tabulate

tabulate = partial(tabulate, tablefmt='pipe')

from matplotlib import pyplot as plt

import numpy as np

import logging

import json

from .vis import draw_graph
from rewann.environment.util import load_ind, load_hof
from rewann.environment.evolution import evaluate_inds

class Report:
    def __init__(self, env, report_name='report'):
        self.path = partial(env.env_path, report_name)
        self.env = env

        self.elements = list()

        self.fig, self.ax = plt.subplots()

    def rel_path(self, abspath):
        return os.path.relpath(abspath, start=self.path())

    def add(self, *elements):
        self.elements += elements

    def add_image(self, path, caption=""):
        self.add(f"![{caption}]({self.rel_path(path)})\n")

    def add_fig(self, name, caption=""):
        p = self.path('media', f'{name}.svg')
        plt.savefig(p)
        self.add_image(p, caption=caption)
        plt.clf()

    def write_main_doc(self, doc_name='index.md'):
        with open(self.path(doc_name), 'w') as f:
            print('\n\n'.join(self.elements), file=f)

    def write_stats(self, stats, fname='stats.json'):
        with open(self.path(fname), 'w') as f:
            json.dump(stats, f)

    def add_gen_metrics(self):
        metrics = self.env.load_gen_metrics()
        metrics.index.names = ['generation']

        for bm in ('log_loss.mean', 'accuracy.mean', 'kappa.mean'):
            mean = metrics[f'MEAN:{bm}']
            median = metrics[f'MEDIAN:{bm}']
            min = metrics[f'MIN:{bm}']
            max = metrics[f'MAX:{bm}']
            gen = metrics.index
            try:
                q1 = metrics[f'Q_1:{bm}']
                q3 = metrics[f'Q_3:{bm}']
            except KeyError:
                logging.warning('Quantiles not included in measurements.')
                q1, q3 = None, None

            plt.plot(gen, median, '-', label='median', linewidth=.25, color='tab:blue')
            plt.plot(gen, mean, '-', label='mean', linewidth=.75, color='tab:green')
            plt.legend()

            if q1 is not None:
                plt.fill_between(gen, q1, q3, alpha=0.2, fc='tab:blue')
            plt.fill_between(gen, min, max, alpha=0.1, fc='tab:blue')
            caption = f"{bm} over generations"
            plt.suptitle(caption)
            self.add_fig(f'gen_metrics_{bm}', caption)

    def add_ind_info(self, ind):
        self.add(f"### Individual {ind.id}\n")
        metrics = ind.metric_values

        self.add(tabulate([
            ('mean log_loss:', metrics['log_loss'].mean()),
            ('mean accuracy:', metrics['accuracy'].mean()),
            ('mean kappa:', metrics['kappa'].mean()),
            ('number of edges', len(ind.genes.edges)),
            ('number of hidden nodes', ind.network.n_hidden),
            ('number of layers', ind.network.n_layers),
            ('birth', ind.birth),
        ], ['key', 'value']))

        # plot graphs
        for m in ('log_loss', 'accuracy', 'kappa'):
            plt.plot(metrics['weight'], metrics[m])
            caption = f'Metric {m}'
            plt.xlabel('weight')
            plt.ylabel(m)
            self.add_fig(f'metric_{m}_{ind.id}', caption)

        self.add('#### Network')

        # plot network
        draw_graph(ind.network, plt.gca())
        caption = f"Network of individual {ind.id}"
        self.add_fig(f'network_{ind.id}', caption)
        plt.clf()

    def compile_stats(self):
        stat_funcs = [('mean', np.mean), ('max', np.max)]
        hof_metrics = [ind.metric_values for ind in self.env.hall_of_fame]

        stats = list()

        for measure in ('accuracy', 'kappa'):
            for func_name, func in stat_funcs:
                descr = f'{func_name} {measure}'
                measures = [func(m[measure]) for m in hof_metrics]
                i = np.argmax(measures)
                indiv_id = self.env.hall_of_fame[i].id
                value = measures[i]
                stats.append((f'{func_name} {measure}', value, indiv_id))

        self.add(
            "## Best results in hall of fame",
            tabulate(stats, ['measure', 'value', 'individual'])
        )

        self.write_stats({
            key: value for key, value, *_ in stats
        })

    def compile(self):
        self.add(f"# Report {self.env['experiment_name']}")

        self.compile_stats()

        # add generation metrics
        self.add_gen_metrics()

        self.add("## Individuals in hall of fame")

        for ind in self.env.hall_of_fame:
            self.add_ind_info(ind)

        self.write_main_doc()

    def run_evaluations(self, num_weights=100, num_samples=1000):
        self.env.sample_weights(num_weights)

        self.env.load_hof()
        self.env.task.load_test()

        logging.info('Evaluating indivs in hall of fame.')

        evaluate_inds(self.env, self.env.hall_of_fame,
                      n_samples=num_samples,
                      reduce_values=False,
                      use_test_samples=True)
        return self

def compile_report():
    from rewann import Environment

    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Post Optimization')
    parser.add_argument('experiment_path', type=str, help='path to experiment')
    parser.add_argument('--num_weights', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=1000)

    args = parser.parse_args()
    env = Environment(args.experiment_path)

    with env.open_data('r'):
        Report(env).run_evaluations(
            num_weights=args.num_weights, num_samples=args.num_samples
        ).compile()
