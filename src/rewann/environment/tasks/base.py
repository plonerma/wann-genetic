class ClassificationTask:
    x, y_true, test_x, test_y_true = None, None, None, None

    def __init__(self, n_in, n_out, train_loader, test_loader=None):
        self.n_in = n_in
        self.n_out = n_out

        self.train_loader = train_loader
        self.test_loader = train_loader if test_loader is None else test_loader

    def load_training(self):
        if self.x is None:
            self.x, self.y_true = self.train_loader()

    def load_test(self):
        if self.test_x is None:
            self.test_x, self.test_y_true = self.test_loader()
