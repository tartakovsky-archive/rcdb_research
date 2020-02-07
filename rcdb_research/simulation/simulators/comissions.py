from typing import Callable


class PositionSizing:
    @staticmethod
    def flat(units: float) -> Callable:

        def sizing_fn() -> float:
            return units

        return sizing_fn

    @staticmethod
    def percent(percent: float) -> Callable:

        def sizing_fn(current_equity: float) -> float:
            return current_equity * percent

        return sizing_fn

    @staticmethod
    def kelly(win_size: float, loss_size: float, divider: float = 1) -> Callable:

        def sizing_fn(current_equity: float, win_proba: float) -> float:
            kelly = win_proba / loss_size - (1 - win_proba) / win_size
            kelly = kelly / divider

            return current_equity * kelly

        return sizing_fn
