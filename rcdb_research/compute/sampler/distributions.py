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
    low: int
    high: int
    step: int = 1
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, int]:
        values = np.arange(self.low, self.high, self.step)
        draw = self.rs.choice(values, n)
        return draw[0] if n == 1 else draw


##########
# Normal
##########
@dataclass
class Normal(Distribution):
    loc: float
    scale: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.loc, self.scale, n)
        return draw[0] if n == 1 else draw


@dataclass
class LogNormal(Distribution):
    loc: float
    scale: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.loc, self.scale, n)
        draw = np.exp(draw)
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedNormal(Distribution):
    loc: float
    scale: float
    q: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.loc, self.scale, n)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


@dataclass
class QuantizedLogNormal(Distribution):
    loc: float
    scale: float
    q: float
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.rs.normal(self.loc, self.scale, n)
        draw = np.exp(draw)
        draw = np.round(draw / self.q) * self.q
        return draw[0] if n == 1 else draw


@dataclass
class TruncatedNormal(Distribution):
    low: float
    high: float
    loc: float
    scale: float

    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)
        self.distribution = stats.truncnorm(
            (self.low - self.loc) / self.scale,
            (self.high - self.loc) / self.scale,
            loc=self.loc,
            scale=self.scale
        )

    def sample(self, n: int = 1) -> Union[np.ndarray, float]:
        draw = self.distribution.rvs(n, random_state=self.rs)
        return draw[0] if n == 1 else draw


@dataclass
class DiscreteNormal(Distribution):
    low: int = -100
    high: int = 100
    loc: float = None
    scale: float = None
    step: int = 1
    seed: int = None

    def __post_init__(self):
        self.rs = np.random.RandomState(self.seed)
        if self.loc is None:
            self.loc = (self.high + self.low) / 2
        if self.scale is None:
            self.scale = (self.high - self.low) / 7  # why 7 works? replace with something locingful

    def sample(self, n: int = 1) -> Union[np.ndarray, int]:
        # based on https://stackoverflow.com/a/37412692
        x = np.arange(self.low, self.high, self.step)

        x_upper = x + 0.5
        x_lower = x - 0.5
        p_upper = stats.norm.cdf(x_upper, loc=self.loc, scale=self.scale)
        p_lower = stats.norm.cdf(x_lower, loc=self.loc, scale=self.scale)
        prob = p_upper - p_lower
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1

        draw = self.rs.choice(x, n, p=prob)
        return draw[0] if n == 1 else draw
