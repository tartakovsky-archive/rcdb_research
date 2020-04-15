from __future__ import annotations

from typing import Union
from dataclasses import dataclass

import numpy as np


# binomial,
# normal,
# qnormal, # quantized normal, see hyperopt https://github.com/hyperopt/hyperopt/wiki/FMin
# lognormal,
# qlognormal,
# uniform,
# quniform,
# loguniform,
# qloguniform,
# discrete_uniform,
# exponential,
# reverse_exponential,


# Base class
class Distribution:
    def sample(self, n: int) -> Union[np.ndarray, float, int]:
        raise NotImplementedError


#
# @dataclass
# class DiscreteUniform(Distribution):
#     start: int
#     end: int
#     step: int = 1
#
#     def sample(self, n: int = 1) -> Union[np.ndarray, int]:
#         # TODO: Add support for step size
#         draw = np.random.randint(self.start, self.end, n)
#         return draw[0] if n == 1 else draw


##########
# Uniform
##########
@dataclass
class Uniform(Distribution):
    low: float
    high: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.uniform(self.low, self.high, n)
        return draw[0] if n == 1 else draw


@dataclass
class LogUniform(Distribution):
    low: float
    high: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.uniform(self.low, self.high, n)
        draw = np.exp(draw)
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedUniform(Distribution):
    low: float
    high: float
    q: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.uniform(self.low, self.high, n)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedLogUniform(Distribution):
    low: float
    high: float
    q: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.uniform(self.low, self.high, n)
        draw = np.exp(draw)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


##########
# Normal
##########
@dataclass
class Normal(Distribution):
    mean: float
    std: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.normal(self.mean, self.std, n)
        return draw[0] if n == 1 else draw


@dataclass
class LogNormal(Distribution):
    mean: float
    std: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.normal(self.mean, self.std, n)
        draw = np.exp(draw)
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedNormal(Distribution):
    mean: float
    std: float
    q: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.normal(self.mean, self.std, n)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedLogNormal(Distribution):
    mean: float
    std: float
    q: float
    seed: int = None

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        self.rs = getattr(self, 'rs', np.random.RandomState(self.seed))
        draw = self.rs.normal(self.mean, self.std, n)
        draw = np.exp(draw)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


class Sampler:
    ##########
    # Uniform
    ##########
    @staticmethod
    def uniform(low: float, high: float) -> Uniform:
        return Uniform(low=low, high=high)

    @staticmethod
    def loguniform(low: float, high: float) -> LogUniform:
        return LogUniform(low=low, high=high)

    @staticmethod
    def quniform(low: float, high: float, q: float = 0.0001) -> QuantizedUniform:
        return QuantizedUniform(low=low, high=high, q=q)

    @staticmethod
    def qloguniform(low: float, high: float, q: float = 0.0001) -> QuantizedLogUniform:
        return QuantizedLogUniform(low=low, high=high, q=q)

    ##########
    # Normal
    ##########
    @staticmethod
    def normal(mean: float, std: float) -> Normal:
        return Normal(mean=mean, std=std)

    @staticmethod
    def lognormal(mean: float, std: float) -> LogNormal:
        return LogNormal(mean=mean, std=std)

    @staticmethod
    def qnormal(mean: float, std: float, q: float = 0.0001) -> QuantizedNormal:
        return QuantizedNormal(mean=mean, std=std, q=q)

    @staticmethod
    def qlognormal(mean: float, std: float, q: float = 0.0001) -> QuantizedLogNormal:
        return QuantizedLogNormal(mean=mean, std=std, q=q)

# Sampler.duniform(0, 10).sample(2)
