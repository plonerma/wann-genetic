class ClassificationTask:
    def __init__(self, samples, labels, n_in, n_out):
        assert len(samples) and len(labels)
        self.n_in = n_in
        self.n_out = n_out
        self.x = samples
        self.y_true = labels
