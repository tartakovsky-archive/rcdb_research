import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._validation import indexable, _num_samples


class WalkForwardCV(BaseCrossValidator):
    def __init__(self, n_splits, test_size, gap_size=.0, expanding=True):
        self.n_splits = int(n_splits)
        assert 0 < gap_size + test_size < 1 and test_size > 0, "Not enough train part"
        self.test_size = test_size
        self.gap_size = gap_size
        self.expanding = expanding

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater"
                f" than the number of samples: n_samples={n_samples}."
            )

        indices = np.arange(n_samples)

        n_sample = n_samples // self.n_splits

        n_test = int(n_sample * self.test_size)
        n_gap = int(n_sample * self.gap_size)


        train_start = 0
        train_end = n_samples - self.n_splits * (n_test + n_gap)  # needs to use all dataset

        n_train = train_end

        for split_number in range(1, self.n_splits + 1):
            test_start = train_end + n_gap

            if split_number == self.n_splits:
                test_end = n_samples
            else:
                test_end = test_start + n_test

            yield indices[train_start:train_end], indices[test_start:test_end]
            train_end = test_end

            if not self.expanding:
                train_start = train_end - n_train
