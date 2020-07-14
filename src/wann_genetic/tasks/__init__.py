from .base import ClassificationTask, RecurrentTask

from .image import mnist_256, digit_raw

from .rnn import EchoTask, AddingTask, CopyTask

from .name_origin import NameOriginTask


def load_iris(test=False):
    from sklearn import datasets

    dataset = datasets.load_iris()
    return dataset['data'], dataset['target'], dataset['target_names']


available_tasks = {
    'iris': ClassificationTask(n_in=4, n_out=3, load_func=load_iris),
    'mnist256': ClassificationTask(
                    n_in=256, n_out=10,
                    load_func=mnist_256),
    'digits': ClassificationTask(
                    n_in=64, n_out=10,
                    load_func=digit_raw),

    'echo': EchoTask(),
    'adding': AddingTask(),
    'copy': CopyTask(),
    'name_origin': NameOriginTask(),
}


def select_task(task_name):
    return available_tasks[task_name]
