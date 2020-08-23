import os
from functools import partial
import argparse

from matplotlib import pyplot as plt

import sklearn
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import pandas as pd

import logging

import json

from .vis import draw_graph
from wann_genetic.environment.util import load_ind
from wann_genetic.environment.evaluation_util import express_inds
from wann_genetic.tasks import ClassificationTask

from tabulate import tabulate

tabulate = partial(tabulate, tablefmt='pipe')


class Report:
    def __init__(self, env, report_name='report'):
        self.path = partial(env.env_path, report_name)
        self.env = env
        self.elements = list()
        self.fig, self.ax = plt.subplots()

        self._gen_metrics = None

    @property
    def gen_metrics(self):
        if self._gen_metrics is None:
            self._gen_metrics = self.env.load_gen_metrics()
            self._gen_metrics.index.names = ['generation']
        return self._gen_metrics

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

    def add_gen_line_plot(self, measures):
        metrics = self.gen_metrics
        metrics.plot(y=measures)

        measures = ', '.join(measures)

        caption = f"{measures} over generations"
        plt.suptitle(caption)
        self.add_fig(f'gen_metrics_{measures}', caption)

    def add_gen_quartiles_plot(self, measure):
        metrics = self.gen_metrics
        mean = metrics[f'MEAN:{measure}']
        median = metrics[f'MEDIAN:{measure}']
        min = metrics[f'MIN:{measure}']
        max = metrics[f'MAX:{measure}']
        gen = metrics.index
        try:
            q1 = metrics[f'Q_1:{measure}']
            q3 = metrics[f'Q_3:{measure}']
        except KeyError:
            logging.warning('Quartiles not included in measurements.')
            q1, q3 = None, None

        plt.plot(gen, median, '-', label='median', linewidth=.25, color='tab:blue')
        plt.plot(gen, mean, '-', label='mean', linewidth=.75, color='tab:green')
        plt.legend()

        if q1 is not None:
            plt.fill_between(gen, q1, q3, alpha=0.2, fc='tab:blue')
        plt.fill_between(gen, min, max, alpha=0.1, fc='tab:blue')
        caption = f"{measure} over generations"
        plt.suptitle(caption)
        self.add_fig(f'gen_metrics_{measure}', caption)

    def add_gen_metrics(self, measures=None):
        if measures is None:
            measures = [f'{m}.mean' for m in self.env['selection', 'recorded_metrics']]

        for bm in measures:
            self.add_gen_quartiles_plot(bm)

    def add_ind_info(self, ind):
        env = self.env

        self.add(f"### Individual {ind.id}\n")

        data = dict(ind.raw_measurements)

        if 'predictions' in data:
            del data['predictions']
        if 'y_true' in data:
            del data['y_true']

        measurements = pd.DataFrame(data=data)
        measurements = measurements.sort_values(by=['weight'])

        self.add(tabulate([
                (f'mean {m}:', measurements[m].mean())
                for m in env['selection', 'recorded_metrics']
            ] +
            [
                ('number of edges', len(ind.genes.edges)),
                ('number of hidden nodes', ind.network.n_hidden),
                ('number of layers', ind.network.n_layers),
                ('birth', ind.birth),
                ('number of mutations', ind.mutations),
            ], ['key', 'value']))

        # plot graphs
        for m in env['selection', 'recorded_metrics']:
            plt.plot(measurements['weight'], measurements[m])
            caption = f'Metric {m}'
            plt.xlabel('weight')
            plt.ylabel(m)
            self.add_fig(f'metric_{m}_{ind.id}', caption)

        if isinstance(env.task, ClassificationTask):
            self.add('#### Confusion matrix')

            cms = [
                sklearn.metrics.confusion_matrix(ind.raw_measurements['y_true'], pred, normalize='all', labels=list(range(len(env.task.y_labels))))
                for pred in ind.raw_measurements['predictions']
            ]

            cmd = ConfusionMatrixDisplay(np.mean(cms, axis=0), display_labels=env.task.y_labels)
            cmd.plot(ax=plt.gca())
            self.add_fig(f'confusion_matrix_{ind.id}')

        self.add('#### Network')

        # plot network
        self.add_network(ind)

    def add_network(self, ind, labels=None):
        draw_graph(ind.network, labels=labels)
        caption = f"Network of individual {ind.id}"
        self.add_fig(f'network_{ind.id}', caption)
        plt.clf()

    def compile_stats(self):
        pop_stat_funcs = [('MIN', np.argmin), ('MAX', np.argmax)]
        hof_measurements = [ind.measurements for ind in self.env.hall_of_fame]

        stats = list()
        json_stats = dict()

        for measure in hof_measurements[0].keys():

            measures = [m[measure] for m in hof_measurements]

            max_i, max_v = np.argmax(measures), np.max(measures)
            min_i, min_v = np.argmin(measures), np.min(measures)
            mean_v = np.mean(measures)

            stats += [
                (f'MAX:{measure}', max_v, self.env.hall_of_fame[max_i].id),
                (f'MIN:{measure}', min_v, self.env.hall_of_fame[min_i].id),
                (f'MEAN:{measure}', mean_v, ''),
            ]

            json_stats.update({
                f'MAX:{measure}': float(max_v),
                f'ARGMAX:{measure}': (int(max_i), int(self.env.hall_of_fame[max_i].id)),
                f'MIN:{measure}': float(min_v),
                f'ARGMIN:{measure}': (int(min_i), int(self.env.hall_of_fame[min_i].id)),
                f'MEAN:{measure}': float(mean_v),
            })

        self.add(
            "## Best results in hall of fame",
            tabulate(stats, ['measure', 'value', 'individual'])
        )

        self.write_stats(json_stats)

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
        env = self.env
        env.load_hof()
        env.task.load_test(env=env)

        express_inds(env, env.hall_of_fame)

        logging.info('Evaluating indivs in hall of fame.')

        measures = env['selection', 'recorded_metrics']

        if isinstance(env.task, ClassificationTask):
            measures = measures + ['predictions']

        x, y_true = env.task.get_data(test=True, samples=num_samples)
        weights = env.sample_weights(num_weights)

        measurements = env.pool_map(
            partial(env.ind_class.Phenotype.get_measurements,
                    weights=weights,
                    x=x, y_true=y_true,
                    measures=measures),
            [ind.network for ind in env.hall_of_fame])

        for ind, m in zip(env.hall_of_fame, measurements):
            ind.measurements = dict()
            for k, v in m.items():
                if k == 'predictions':
                    continue
                for fname, func in [('min', np.min), ('mean', np.mean), ('max', np.max)]:
                    ind.measurements[f'{k}.{fname}'] = func(v)

            m.update({
                'y_true': y_true[~np.isnan(y_true)],
                'weight': weights,
            })
            ind.raw_measurements = m

        metric, metric_sign = env.hof_metric

        env.hall_of_fame = sorted(
            env.hall_of_fame,
            key=lambda ind: -metric_sign*ind.measurements[metric])

        return self


def compile_report():
    from wann_genetic import Environment

    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Post Optimization')
    parser.add_argument('--path', type=str, help='path to experiment', default='.')
    parser.add_argument('--weights', '-w', type=int, default=100)
    parser.add_argument('--samples', '-s', type=int, default=1000)

    args = parser.parse_args()
    env = Environment(args.path)

    with env.open_data('r'):
        Report(env).run_evaluations(
            num_weights=args.weights, num_samples=args.samples
        ).compile()


def draw_network():
    from wann_genetic import Environment

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Post Optimization')
    parser.add_argument('id', type=int, help='id of the network')

    parser.add_argument('--path', '-p', type=str, default='.',
                        help='path to experiment')

    parser.add_argument('--function_names', '-l', action='store_true',
                        help='use names of activation as node labels')
    parser.add_argument('--function_plots', '-g', action='store_true',
                        help='use plot functions on nodes')
    parser.add_argument('--names', '-n', action='store_true',
                        help='use names of nodes as labels')

    args = parser.parse_args()
    env = Environment(args.path)

    with env.open_data('r'):
        ind = load_ind(env, args.id)
        express_inds(env, [ind])

        if args.names:
            labels = 'names'
        elif args.function_names:
            labels = 'func_names'
        elif args.function_plots:
            labels = 'func_plots'
        else:
            labels = None

        Report(env).add_network(ind, labels=labels)
        logging.info(f'Plotted network for individual {ind.id}')


def plot_gen_quartiles():
    from wann_genetic import Environment

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Post Optimization')
    parser.add_argument('--path', '-p', type=str, default='.',
                        help='path to experiment')
    parser.add_argument('measure', type=str, default='accuracy', nargs='+')

    args = parser.parse_args()
    env = Environment(args.path)

    Report(env).add_gen_metrics(args.measure)


def plot_gen_lines():
    from wann_genetic import Environment

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Post Optimization')
    parser.add_argument('--path', '-p', type=str, default='.',
                        help='path to experiment')
    parser.add_argument('measure', type=str, default='accuracy', nargs='+')

    args = parser.parse_args()
    env = Environment(args.path)

    Report(env).add_gen_line_plot(args.measure)
