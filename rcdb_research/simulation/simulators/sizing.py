from typing import Callable


class PositionSizing:
    @staticmethod
    def percent(percent: float, threshold: float = 0.5, direction: str = 'both') -> Callable[[float], float]:

        supported_directions = ['pos', 'neg', 'both']
        if direction not in supported_directions:
            raise ValueError(
                f'{direction} direction is not supported. Should be one of the following: {supported_directions}'
            )

        def sizing_fn(proba: float) -> float:
            if direction == 'pos':
                return percent if proba > threshold else 0
            elif direction == 'neg':
                return -percent if (1 - proba) > threshold else 0
            else:
                return percent if proba > threshold else -percent if (1 - proba) > threshold else 0

        return sizing_fn

    @staticmethod
    def kelly(win_size: float, loss_size: float, divider: float = 1, direction: str = 'both') -> Callable[[float], float]:

        supported_directions = ['pos', 'neg', 'both']
        if direction not in supported_directions:
            raise ValueError(
                f'{direction} direction is not supported. Should be one of the following: {supported_directions}'
            )

        def kelly_fn(win_size: float, loss_size: float, win_proba: float) -> float:
            return win_proba / loss_size - (1 - win_proba) / win_size

        def sizing_fn(proba: float) -> float:
            pos_kelly = kelly_fn(win_size, loss_size, proba)
            neg_kelly = -kelly_fn(win_size, loss_size, 1 - proba)

            if direction == 'pos':
                kelly = pos_kelly if pos_kelly > 0 else 0
            elif direction == 'neg':
                kelly = neg_kelly if neg_kelly < 0 else 0
            else:
                kelly = pos_kelly if pos_kelly > 0 else neg_kelly if neg_kelly < 0 else 0

            return kelly / divider

        return sizing_fn
