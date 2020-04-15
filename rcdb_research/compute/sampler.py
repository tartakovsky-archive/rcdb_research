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

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.uniform(self.low, self.high, n)
        return draw[0] if n == 1 else draw


@dataclass
class LogUniform(Distribution):
    low: float
    high: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.uniform(self.low, self.high, n)
        draw = np.exp(draw)
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedUniform(Distribution):
    low: float
    high: float
    q: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.uniform(self.low, self.high, n)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedLogUniform(Distribution):
    low: float
    high: float
    q: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
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

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.mean, self.std, n)
        return draw[0] if n == 1 else draw


@dataclass
class LogNormal(Distribution):
    mean: float
    std: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.mean, self.std, n)
        draw = np.exp(draw)
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedNormal(Distribution):
    mean: float
    std: float
    q: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.mean, self.std, n)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedLogNormal(Distribution):
    mean: float
    std: float
    q: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.mean, self.std, n)
        draw = np.exp(draw)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


class Sampler:
    ##########
    # Uniform
    ##########
    @staticmethod
    def uniform(low: float, high: float, seed: int = None) -> Uniform:
        return Uniform(low=low, high=high, seed=seed)

    @staticmethod
    def loguniform(low: float, high: float, seed: int = None) -> LogUniform:
        return LogUniform(low=low, high=high, seed=seed)

    @staticmethod
    def quniform(low: float, high: float, q: float = 0.0001, seed: int = None) -> QuantizedUniform:
        return QuantizedUniform(low=low, high=high, q=q, seed=seed)

    @staticmethod
    def qloguniform(low: float, high: float, q: float = 0.0001, seed: int = None) -> QuantizedLogUniform:
        return QuantizedLogUniform(low=low, high=high, q=q, seed=seed)

    ##########
    # Normal
    ##########
    @staticmethod
    def normal(mean: float, std: float, seed: int = None) -> Normal:
        return Normal(mean=mean, std=std, seed=seed)

    @staticmethod
    def lognormal(mean: float, std: float, seed: int = None) -> LogNormal:
        return LogNormal(mean=mean, std=std, seed=seed)

    @staticmethod
    def qnormal(mean: float, std: float, q: float = 0.0001, seed: int = None) -> QuantizedNormal:
        return QuantizedNormal(mean=mean, std=std, q=q, seed=seed)

    @staticmethod
    def qlognormal(mean: float, std: float, q: float = 0.0001, seed: int = None) -> QuantizedLogNormal:
        return QuantizedLogNormal(mean=mean, std=std, q=q, seed=seed)

