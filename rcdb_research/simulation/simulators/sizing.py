from typing import Callable


class Fee:
    @staticmethod
    def bitfinex() -> Callable:

        def fee_fn() -> dict:
            return dict(maker=-0.2/100, taker=-0.2/100)

        return fee_fn

    @staticmethod
    def bitmex() -> Callable:

        def fee_fn() -> dict:
            return dict(maker=0.025/100, taker=-0.075/100)

        return fee_fn

    @staticmethod
    def binance() -> Callable:

        def fee_fn() -> dict:
            return dict(maker=-0.075/100, taker=-0.075/100)

        return fee_fn

    @staticmethod
    def custom(maker: float, taker: float) -> Callable:

        def fee_fn() -> dict:
            return dict(maker=maker, taker=taker)

        return fee_fn


class Slippage:
    @staticmethod
    def percent(percent: float) -> Callable:

        def slippage_fn() -> float:
            return percent

        return slippage_fn


class MarketImpact:
    @staticmethod
    def percent(percent: float) -> Callable:

        def impact_fn() -> float:
            return percent

        return impact_fn
