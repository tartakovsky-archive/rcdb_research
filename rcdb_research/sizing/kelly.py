import os
from typing import Optional, Dict

import numpy as np

from . import mc_sizing


class Kelly:
    def __init__(self, direction: str = 'both'):
        supported_directions = ['pos', 'neg', 'both']
        if direction not in supported_directions:
            raise ValueError(
                f'{direction} direction is not supported. Should be one of the following: {supported_directions}'
            )

        self.direction = direction

    def size(self, proba: float, exp_win: float, exp_loss: float) -> float:
        return kelly(win_proba=proba, exp_win=exp_win, exp_loss=exp_loss, direction=self.direction)


class FractionalKelly(Kelly):
    def __init__(self, fraction: float = 1.0, direction: str = 'both'):
        super().__init__(direction=direction)
        self.fraction = fraction

    def size(self, proba: float, exp_win: float, exp_loss: float) -> float:
        return super().size(proba=proba, exp_win=exp_win, exp_loss=exp_loss) * self.fraction


class RiskAdjustedKelly(FractionalKelly):
    def __init__(self, max_dd: float, max_dd_proba: float = 0.001, direction: str = 'both', mc_params={}):
        super().__init__(
            fraction=estimate_kelly_fraction(max_dd=max_dd, max_dd_proba=max_dd_proba, mc_params=mc_params),
            direction=direction
        )


def kelly(win_proba: float, exp_win: float, exp_loss: float, direction='both') -> float:
    supported_directions = ['pos', 'neg', 'both']
    if direction not in supported_directions:
        raise ValueError(
            f'{direction} direction is not supported. Should be one of the following: {supported_directions}'
        )

    def kelly_fn(p: float, win: float, loss: float) -> float:
        return p / abs(loss) - (1 - p) / win

    pos_kelly = np.maximum(0, kelly_fn(p=win_proba, win=exp_win, loss=exp_loss))
    neg_kelly = -np.maximum(0, kelly_fn(p=(1 - win_proba), win=exp_win, loss=exp_loss))

    if direction == 'pos':
        return pos_kelly
    elif direction == 'neg':
        return neg_kelly
    else:
        return pos_kelly + neg_kelly


def estimate_kelly_fraction(
        max_dd: float,
        max_dd_proba: float = 0.001,
        compounded: bool = True,
        mc_params: Optional[Dict] = None,
        cache_path=None
) -> float:
    if mc_params is None:
        mc_params = {}

    mc_params = {
        'xtol': 1e-4,
        'size_upper_bound': 100,
        'n_curves': 1000,
        'n_steps': 50000,
        **mc_params
    }

    risk_preferences = dict(
        minimum_wealth=1 - max_dd,
        max_drawdown_risk=max_dd_proba
    )

    if cache_path is not None:
        cache_path = os.path.expanduser(cache_path)
        os.makedirs(cache_path, exist_ok=True)
        d = {**risk_preferences, **mc_params}
        d = {k: v for k, v in sorted(d.items(), key=lambda x: x[0])}
        path = os.path.join(cache_path, str(d))
        if os.path.exists(path):
            return float(open(path, 'r').read())

    movement_sizes = np.arange(0.015, 0.99, 0.05)
    prob_points = np.linspace(0.6, 0.8, 10)
    coefficients = []

    for movement_size in movement_sizes:
        for p in prob_points:
            mc = mc_sizing.compute_bet(
                avg_win=movement_size,
                avg_loss=movement_size,
                **mc_params,
                **risk_preferences,
                predicted_probability=p,
                compounded=compounded
            )
            ke = kelly(p, movement_size, movement_size)
            coefficients.append(mc / ke)
    result = np.mean(coefficients)

    if cache_path is not None:
        open(path, 'w').write(str(result))

    return float(result)  # noqa
