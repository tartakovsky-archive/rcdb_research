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

    @staticmethod
    def get_n_wf_split(n_samples, offset_pct, n_splits):
        a = offset_pct * (n_splits - 1)
        return (n_samples / a) / (1.0 + 1.0 / a)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater"
                f" than the number of samples: n_samples={n_samples}."
            )

        indices = np.arange(n_samples)

        n_wf_split = self.get_n_wf_split(
            n_samples, self.test_size, self.n_splits
        )

        n_test = int(n_wf_split * self.test_size)
        n_gap = int(n_wf_split * self.gap_size)
        n_train = int(n_wf_split - n_test - n_gap)

        # print(f"n_wf_split={n_wf_split}, n_test={n_test}, n_gap={n_gap}, n_train={n_train}")

        train_start = 0
        for split_number in range(1, self.n_splits + 1):
            train_end = train_start + n_train
            test_start = train_end + n_gap

            if split_number == self.n_splits:
                test_end = n_samples - 1
            else:
                test_end = test_start + n_test

            print(train_start, train_end, test_start, test_end)
            yield indices[0 if self.expanding else train_start:train_end], indices[test_start:test_end]

            train_start = train_start + n_test
