import copy

import numpy as np
import scipy.stats as stats


class Transform:
    def transform(self, *args, **kwargs):
        raise NotImplementedError


class LogTransform(Transform):
    def transform(self, v):
        return np.exp(v)


class QuantizeTransform(Transform):
    def __init__(self, q: float):
        self.q = q

    def transform(self, v: np.ndarray) -> np.ndarray:
        return v // self.q * self.q  # np.round(v / self.q) * self.q


class DiscretizeTransform(Transform):
    def __init__(self, step: int):
        if type(step) != int:
            raise ValueError("Discretize step is `int` only")
        self.step = step

    def transform(self, v: np.ndarray) -> np.ndarray:
        return (v // self.step * self.step).astype(np.int)


class Sampler:
    def __init__(self, name, seed=None, transforms=None, **kwargs):
        self.name = name
        self.seed = seed
        self.kwargs = kwargs
        self.transforms = transforms if transforms else []

    @staticmethod
    def uniform(low: float, high: float, *args, **kwargs):
        loc = low
        scale = high - loc
        return Sampler("uniform", *args, loc=loc, scale=scale, **kwargs)

    @staticmethod
    def normal(loc: float = 0, scale: float = 3, *args, **kwargs):
        return Sampler("norm", loc=loc, scale=scale, *args, **kwargs)

    @staticmethod
    def truncnormal(low: float = 0, high: float = 1, loc: float = None, scale: float = None, *args, **kwargs):
        return Sampler("truncnormal", low=low, high=high, loc=loc, scale=scale, *args, **kwargs)

    ##########

    def __get_distribution__(self):
        if self.name == "truncnormal":
            low = self.kwargs['low']
            high = self.kwargs['high']

            if 'scale' not in self.kwargs:
                # why 7 works? replace with something logical
                scale = (high - low) / 7
            else:
                scale = self.kwargs['scale']

            if 'loc' not in self.kwargs:
                loc = (high + low) / 2
            else:
                loc = self.kwargs['loc']

            d = stats.truncnorm(
                (low - loc) / scale,
                (high - loc) / scale,
                loc=loc,
                scale=scale
            )
        else:
            d = getattr(stats, self.name)(**self.kwargs)

        return d

    def draw(self, n: int = 1):
        rs = np.random.RandomState(self.seed)
        d = self.__get_distribution__()
        val = d.rvs(size=n, random_state=rs)

        if self.transforms:
            for t in self.transforms:
                val = t.transform(val)

        return val[0] if n == 1 else val

    def cp(self):
        return copy.deepcopy(self)

    def log(self):
        obj = self.cp()
        obj.transforms.append(LogTransform())
        return obj

    def quantize(self, q: float):
        obj = self.cp()
        obj.transforms.append(QuantizeTransform(q=q))
        return obj

    def discretize(self, step: int):
        obj = self.cp()
        obj.transforms.append(DiscretizeTransform(step=step))
        return obj
