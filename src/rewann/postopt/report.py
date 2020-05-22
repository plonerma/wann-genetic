import os
from functools import partial
import argparse

from tabulate import tabulate

tabulate = partial(tabulate, tablefmt='pipe')

from matplotlib import pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np

import logging

import json

from .vis import draw_graph
from rewann.environment.util import load_ind, load_hof
from rewann.environment.evolution import evaluate_inds, express_inds
from rewann.environment.tasks import ClassificationTask

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
            measures = [f'{m}.mean' for m in self.env.ind_class.recorded_measures]

        for bm in measures:
            self.add_gen_quartiles_plot(bm)

    def add_ind_info(self, ind):
        self.add(f"### Individual {ind.id}\n")
        measurements = ind.measurements_df(*ind.recorded_measures)

        self.add(tabulate([
            (f'mean {m}:', measurements[m].mean())
            for m in ind.recorded_measures
        ] +
        [
            ('number of edges', len(ind.genes.edges)),
            ('number of hidden nodes', ind.network.n_hidden),
            ('number of layers', ind.network.n_layers),
            ('birth', ind.birth),
            ('number of mutations', ind.mutations),
        ], ['key', 'value']))

        # plot graphs
        for m in ind.recorded_measures:
            plt.plot(measurements['weight'], measurements[m])
            caption = f'Metric {m}'
            plt.xlabel('weight')
            plt.ylabel(m)
            self.add_fig(f'metric_{m}_{ind.id}', caption)

        if isinstance(self.env.task, ClassificationTask):
            self.add('#### Confusion matrix')

            cmd = ConfusionMatrixDisplay(np.mean(ind.measurements('cm'), axis=0), display_labels=self.env.task.y_labels)
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
        measures = self.env.ind_class.recorded_measures
        stat_funcs = [('mean', np.mean), ('max', np.max), ('min', np.min)]
        hof_metrics = [ind.measurements(*measures, as_dict=True) for ind in self.env.hall_of_fame]

        stats = list()

        for measure in measures:
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
                      record_raw=True,
                      use_test_samples=True)

        metric, metric_sign = self.env.hof_metric
        metric, m_func = metric.split('.')

        m_func = m_func.lower()
        m_func = dict(
            mean=np.mean,
            max=np.max,
            min=np.min
        ).get(m_func, np.mean)

        self.env.hall_of_fame = sorted(self.env.hall_of_fame,
            key=lambda ind: -metric_sign*m_func(ind.measurements(metric))
        )

        return self

def compile_report():
    from rewann import Environment

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
    from rewann import Environment

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
    from rewann import Environment

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Post Optimization')
    parser.add_argument('--path', '-p', type=str, default='.',
                        help='path to experiment')
    parser.add_argument('measure', type=str, default='accuracy', nargs='+')

    args = parser.parse_args()
    env = Environment(args.path)

    Report(env).add_gen_metrics(args.measure)

def plot_gen_lines():
    from rewann import Environment

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Post Optimization')
    parser.add_argument('--path', '-p', type=str, default='.',
                        help='path to experiment')
    parser.add_argument('measure', type=str, default='accuracy', nargs='+')

    args = parser.parse_args()
    env = Environment(args.path)

    Report(env).add_gen_line_plot(args.measure)
