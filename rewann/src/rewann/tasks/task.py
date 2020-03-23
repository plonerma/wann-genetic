from .performance import Performance

class ClassificationTask:
    from .performance import Performance

    def __init__(self, n_classes, n_dims):
        self.n_dims = n_dims
        self.n_classes = n_classes

    def evaluate_individual(self, ind):
        """Iterate through batches to be used for evaluation."""
        performance = Performance(self.n_classes)

        for x, y_true in self.batches:
            y_pred = ind.apply(x, func='argmax')
            performance += (y_true, y_pred)

        return performance

class RLTask:
    def evaluate_individual(self, ind):
        self.setup_env()
        while keep_going:
            sample = self.get_input()
            output = ind.apply(sample)
            self.evaluate_output(output)

        return performance


from sklearn import datasets

class SklearnClassificationTask(ClassificationTask):
    def __init__(self, samples, labels, n_classes, n_dims, batch_size=None):
        super().__init__(n_classes, n_dims)

        if batch_size is None:
            self.batches = [(samples, labels)]
        else:
            self.batches = [
                (samples[i : i + batch_size], labels[i : i + batch_size])
                for i in range(0, samples.shape[0], batch_size)
            ]


IrisTask = SklearnClassificationTask(
    *datasets.load_iris(return_X_y=True), n_classes=3, n_dims=4
)
