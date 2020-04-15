from __future__ import annotations

from .distributions import Uniform, LogUniform, QuantizedUniform, QuantizedLogUniform, DiscreteUniform
from .distributions import Normal, LogNormal, QuantizedNormal, QuantizedLogNormal, TruncatedNormal, DiscreteNormal


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

    @staticmethod
    def duniform(start: int, end: int, step: int, seed: int = None) -> DiscreteUniform:
        return DiscreteUniform(start=start, end=end, step=step, seed=seed)

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

    @staticmethod
    def dnormal(start: int, end: int, step: int, seed: int = None) -> DiscreteNormal:
        return DiscreteNormal(start=start, end=end, step=step, seed=seed)

    @staticmethod
    def truncnormal(mean: float, std: float, start: float, end: float, seed: int = None) -> TruncatedNormal:
        return TruncatedNormal(mean=mean, std=std, start=start, end=end, seed=seed)
