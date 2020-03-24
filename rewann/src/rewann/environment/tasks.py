class ClassificationTask:
    def __init__(self, samples, labels, n_in, n_out):
        assert len(samples) and len(labels)
        self.n_in = n_in
        self.n_out = n_out
        self.x = samples
        self.y_true = labels

from sklearn import datasets

iris = datasets.load_iris()

available_tasks = {
    'iris': ClassificationTask(iris['data'], iris['target'],
                               n_in =len(iris['feature_names']),
                               n_out=len(iris['target_names'])),
}

def select_task(task_name):
    return available_tasks.get(task_name, available_tasks.get('iris'))
