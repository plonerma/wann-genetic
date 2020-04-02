from .base import ClassificationTask

from .image import mnist_256, digit_raw


def iris():
    from sklearn import datasets

    dataset = datasets.load_iris()
    return ClassificationTask(dataset['data'], dataset['target'],
                              n_in =len(dataset['feature_names']),
                              n_out=len(dataset['target_names']))



available_tasks = {
    'iris': iris,
    'mnist256' : mnist_256,
    'digits': digit_raw
}

def select_task(task_name):
    task = available_tasks.get(task_name, available_tasks.get('iris'))
    return task()
