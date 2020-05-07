class Task:
    def load_training(self):
        pass

    def load_test(self):
        pass

    def get_data(self):
        raise NotImplementedError()

class ClassificationTask(Task):
    x, y_true, test_x, test_y_true = None, None, None, None

    def __init__(self, n_in, n_out, train_loader, test_loader=None):
        self.n_in = n_in
        self.n_out = n_out

        self.train_loader = train_loader
        self.test_loader = train_loader if test_loader is None else test_loader

    def load_training(self):
        if self.x is None:
            self.x, self.y = self.train_loader()

    def load_test(self):
        if self.test_x is None:
            self.test_x, self.test_y = self.test_loader()

    def get_data(self, samples, test=False):
        if samples:
            if self.test_x is None:
                logging.warning("Evaluation on test data requires loading test data.")
            x = self.test_x
            y = self.test_y

        else:
            if self.x is None:
                logging.warning("Evaluation on training data requires loading training data.")

            x = self.x
            y = self.y

        if samples > 0 and samples < len(x):
            chosen = np.random.choice(len(x), size=samples, replace=False)
            x = x[chosen]
            y = y[chosen]

        return x, y
