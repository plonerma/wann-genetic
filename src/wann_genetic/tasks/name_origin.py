"""This code is derived from the pytorch tutorial on nlp.

https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

import numpy as np

from wann_genetic.tasks import RecurrentTask, ClassificationTask

class NameOriginTask(RecurrentTask, ClassificationTask):
    all_letters = string.ascii_letters + " .,;'"
    n_in = len(all_letters)
    n_out = 18

    def __init__(self):
        pass

    def unicodeToAscii(self, s):
        """Turn a Unicode string to plain ASCII,
        thanks to https://stackoverflow.com/a/518232/2809427"""

        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def readNames(self, filename):
        """Read a file and split into lines."""

    def load(self, *, env, test=False):
        languages = {}

        no_files = True

        for filename in glob.glob('name_origin_data/*.txt'):
            no_files = False
            language = os.path.splitext(os.path.basename(filename))[0]
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            languages[language] = [self.unicodeToAscii(line) for line in lines]

        if no_files:
            raise RuntimeError('Data not found.')

        all_languages = sorted(list(languages.keys()))

        assert len(all_languages) == self.n_out

        self._y_labels = all_languages

        seed = env['task', 'sample_order_seed']

        test_portion = env['task', 'test_portion']

        np.random.seed(seed)

        data = list()
        for lang in all_languages:
            # select random subset of available data
            selection = np.random.rand(len(languages[lang])) < (1-test_portion)

            assert selection.size >= 2

            if not np.any(selection):
                selection[np.random.randint(selection.size)] = True
            elif np.all(selection):
                selection[np.random.randint(selection.size)] = True


            if test:
                # use the other part
                selection = ~selection

            for sample, selected in zip(languages[lang], selection):
                # only use selected samples
                if selected:
                    data.append((all_languages.index(lang), sample))

        self.data = data

    def get_data(self, samples, test=False):
        data = self.data

        if samples > 0 and samples < len(data):
            chosen = np.random.choice(len(data), size=samples, replace=False)
            data = [data[i] for i in chosen]

        max_sample_length = 0
        for _, sample in data:
            max_sample_length = max(len(sample), max_sample_length)

        # build x and y arrays from data
        x = np.zeros((len(data), max_sample_length, self.n_in))
        y = np.full((len(data), max_sample_length), np.nan, dtype=float)

        for i, d in enumerate(data):
            lang, sample = d

            for j, c in enumerate(sample):
                c = self.all_letters.index(c)
                x[i, j, c] = 1

            y[i, len(sample) - 1] = lang

        return x, y
