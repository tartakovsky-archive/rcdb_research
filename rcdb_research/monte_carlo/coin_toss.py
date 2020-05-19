from numpy.random import RandomState
from typing import List, Union, Optional
import numpy as np


def mc_coin_toss_game(
        p: float = 0.5,
        profit: float = 0.01,
        loss: float = -0.01,
        n_tosses: int = 500,
        n_games: int = 30,
        cumulative: bool = True,
        compounded: bool = False,
        random_state: Optional[Union[RandomState, int]] = None) -> List[np.ndarray]:
    if random_state is None:
        random_state = RandomState()
    elif isinstance(random_state, int):
        random_state = RandomState(random_state)

    def simulate_game() -> np.ndarray:
        tosses = random_state.choice([0, 1], size=n_tosses, p=[1 - p, p])
        returns = np.where(tosses == 1, profit, loss)

        if cumulative:
            cum_return = np.cumprod(returns + 1) - 1 if compounded else np.cumsum(returns)
            return cum_return

        return returns

    return [simulate_game() for _ in range(n_games)]
