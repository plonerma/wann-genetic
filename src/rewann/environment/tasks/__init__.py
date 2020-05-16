from .base import ClassificationTask, RecurrentTask

from .image import mnist_256, digit_raw

from .rnn import EchoTask, AddingTask

from functools import partial


def load_iris():
    from sklearn import datasets

    dataset = datasets.load_iris()
    return dataset['data'], dataset['target']


available_tasks = {
    'iris': ClassificationTask(n_in=4, n_out=3, train_loader=load_iris),
    'mnist256': ClassificationTask(
                    n_in=256, n_out=10,
                    train_loader=mnist_256,
                    test_loader=partial(mnist_256, load_test=True)),
    'digits': ClassificationTask(
                    n_in=64, n_out=10,
                    train_loader=digit_raw,
                    test_loader=partial(digit_raw, load_test=True)),

    'echo20': EchoTask(20),
    'adding': AddingTask(8)
}

def select_task(task_name):
    return available_tasks.get(task_name, available_tasks.get('iris'))
