import pandas as pd

from . import components, markets, holidays  # noqa


def calc_all(data: pd.DatetimeIndex, param_sets: dict = None, column_names=None) -> pd.DataFrame:
    df = pd.concat(
        [
            components.calc_all(data),
            markets.calc_all(data, param_sets[markets.PREFIX]),
            holidays.calc_all(data, param_sets[holidays.PREFIX])
        ],
        axis=1
    )
    return df
