from __future__ import annotations

from typing import Union
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats


# Base class
class Distribution:
    def sample(self, n: int) -> Union[np.ndarray, float, int]:
        raise NotImplementedError


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


@dataclass
class DiscreteUniform(Distribution):
    start: int
    end: int
    step: int = 1
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, int]:
        values = np.arange(self.start, self.end, self.step)
        draw = self.rs.choice(values, n)
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


@dataclass
class TruncatedNormal(Distribution):
    mean: float
    std: float
    start: float
    end: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)
        self.distribution = stats.truncnorm(
            (self.start - self.mean) / self.std,
            (self.end - self.mean) / self.std,
            loc=self.mean,
            scale=self.std
        )

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.distribution.rvs(n, random_state=self.rs)
        return draw[0] if n == 1 else draw


@dataclass
class DiscreteNormal(Distribution):
    start: int
    end: int
    step: int = 1
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, int]:
        x = np.arange(self.start, self.end, self.step)

        x_upper = x + 0.5
        x_lower = x - 0.5
        prob = stats.norm.cdf(x_upper, scale=3) - stats.norm.cdf(x_lower, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1

        draw = self.rs.choice(x, n, p=prob)
        return draw[0] if n == 1 else draw
