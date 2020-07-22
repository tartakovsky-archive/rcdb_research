from itertools import combinations_with_replacement

import pytest
import numpy as np
import pandas as pd

from numpy_ext import nans
from rcdb_research import bars
from rcdb_research.bars.functions import move_feature_by_nans


@pytest.mark.parametrize(
    'func, params',
    [
        (bars.facade.time, dict(period=5)),
        (bars.facade.percent_o2c, dict(threshold=.5)),
        (bars.facade.percent, dict(threshold=.5)),
        (bars.facade.fixed, dict(threshold=.5, column='col')),
        (bars.facade.adaptive, dict(avg_per=3, window=5, column='col')),
        (bars.facade.fixed_percent_fixed_series, dict(percent_threshold=.5, series_threshold=.5, series_column='col')),
    ]
)
def test_facade_funcs_type_checks(func, params, df=pd.DataFrame(['s'])):
    with pytest.raises(ValueError):
        func(df, **params)


@pytest.fixture
def input_df():
    return pd.DataFrame(
        {
            'open': [1, 2, 3, 4, 5, 6],
            'high': [2, 3, 4, 5, 6, 7],
            'low': [1, 2, 3, 4, 5, 6],
            'close': [1, 2, 3, 4, 5, 6],
            'volume_buy': [10, 20, 30, 40, 50, 60],
            'volume_sell': [1, 0, 2, 0, 1, 3],
            'volume_quote_buy': [1, 2, 3, 4, 5, 6],
            'volume_quote_sell': [1, 2, 3, 4, 5, 6],
            'ticks_buy': [1, 2, 3, 4, 5, 6],
            'ticks_sell': [1, 2, 3, 4, 5, 6],
            'custom_col': [.1, .2, .3, .4, .5, .6]
        },
        index=pd.to_datetime(
            [
                '22/10/2019 12:00:00',
                '22/10/2019 12:00:01',
                '22/10/2019 12:00:02',

                '22/10/2019 12:00:04',
                '22/10/2019 12:00:05',
                '22/10/2019 12:00:05',
            ]
        )
    )


NANS_COMBINATIONS = [
    '-'.join(x)
    for x in set(map(lambda x: tuple(set(x)), combinations_with_replacement(['head', 'middle', 'tail'], 3)))
]


@pytest.fixture()
def ohlcv_df(ohlcv_df):
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index).tz_localize(None)
    return ohlcv_df


@pytest.fixture(
    params=NANS_COMBINATIONS,
    ids=NANS_COMBINATIONS
)
def ohlcv_df_nans(request, ohlcv_df):
    nan_indexes = {
        'head': [
            '15/05/2019 11:59:00',
            '15/05/2019 11:59:01',
            '15/05/2019 11:59:02',
        ],
        'middle': [
            '16/05/2019 16:58:10.150',
            '16/05/2019 16:58:10.200',
            '16/05/2019 16:58:10.250',
        ],
        'tail': [
            '22/10/2019 13:00:00',
            '22/10/2019 13:00:01',
            '22/10/2019 13:00:02',
        ],
    }
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index).tz_localize(None)
    dfs = [ohlcv_df]
    for part in request.param.split('-'):
        dfs.append(
            pd.DataFrame(
                nans((3, ohlcv_df.shape[-1])),
                columns=ohlcv_df.columns,
                index=pd.to_datetime(nan_indexes[part])
            )
        )

    return pd.concat(dfs).sort_index()


@pytest.mark.parametrize(
    "func, params, test_res_df",
    [
        # time
        (
            bars.time, dict(period=3),
            pd.DataFrame(
                {
                    'open': [1, 4],
                    'high': [4, 7],
                    'low': [1, 4],
                    'close': [3, 6],
                    'volume_buy': [60, 150],
                    'volume_sell': [3, 4],
                    'volume_quote_buy': [6, 15],
                    'volume_quote_sell': [6, 15],
                    'ticks_buy': [6, 15],
                    'ticks_sell': [6, 15],
                    'custom_col': [.1, .4]

                },
                index=pd.to_datetime(['22/10/2019 12:00:00', '22/10/2019 12:00:03'])
            )
        ),
        # percent_o2c
        (
            bars.percent_o2c, dict(threshold=0.5),
            pd.DataFrame(
                {
                    'open': [1, 3, 6],
                    'high': [3, 6, 7],
                    'low': [1, 3, 6],
                    'close': [2, 5, 6],
                    'volume_buy': [30, 120, 60],
                    'volume_sell': [1, 3, 3],
                    'volume_quote_buy': [3, 12, 6],
                    'volume_quote_sell': [3, 12, 6],
                    'ticks_buy': [3, 12, 6],
                    'ticks_sell': [3, 12, 6],
                    'custom_col': [.1, .3, .6]

                },
                index=pd.to_datetime(['22/10/2019 12:00:00', '22/10/2019 12:00:02', '22/10/2019 12:00:05'])
            )
        ),
        # percent
        (
            bars.percent, dict(threshold=0.5),
            pd.DataFrame(
                {
                    'open': [1, 3, 4, 6],
                    'high': [3, 4, 6, 7],
                    'low': [1, 3, 4, 6],
                    'close': [2, 3, 5, 6],
                    'volume_buy': [30, 30, 90, 60],
                    'volume_sell': [1, 2, 1, 3],
                    'volume_quote_buy': [3, 3, 9, 6],
                    'volume_quote_sell': [3, 3, 9, 6],
                    'ticks_buy': [3, 3, 9, 6],
                    'ticks_sell': [3, 3, 9, 6],
                    'custom_col': [.1, .3, .4, .6]

                },
                index=pd.to_datetime(
                    ['22/10/2019 12:00:00', '22/10/2019 12:00:02', '22/10/2019 12:00:04', '22/10/2019 12:00:05']
                )
            )
        ),
        # fixed_volume
        (
            bars.fixed_volume, dict(threshold=0.5),
            pd.DataFrame(
                {
                    'open': [1, 3, 4, 5, 6],
                    'high': [3, 4, 5, 6, 7],
                    'low': [1, 3, 4, 5, 6],
                    'close': [2, 3, 4, 5, 6],
                    'volume_buy': [30, 30, 40, 50, 60],
                    'volume_sell': [1, 2, 0, 1, 3],
                    'volume_quote_buy': [3, 3, 4, 5, 6],
                    'volume_quote_sell': [3, 3, 4, 5, 6],
                    'ticks_buy': [3, 3, 4, 5, 6],
                    'ticks_sell': [3, 3, 4, 5, 6],
                    'volume': [31, 32, 40, 51, 63],
                    'custom_col': [.1, .3, .4, .5, .6]
                },
                index=pd.to_datetime(['22/10/2019 12:00:00', '22/10/2019 12:00:02',
                                      '22/10/2019 12:00:04', '22/10/2019 12:00:05',
                                      '22/10/2019 12:00:05'])
            )
        ),
        # fixed_quote_volume
        (
            bars.fixed_quote_volume, dict(threshold=0.5),
            pd.DataFrame(
                {
                    'open': [1, 3, 4, 5, 6],
                    'high': [3, 4, 5, 6, 7],
                    'low': [1, 3, 4, 5, 6],
                    'close': [2, 3, 4, 5, 6],
                    'volume_buy': [30, 30, 40, 50, 60],
                    'volume_sell': [1, 2, 0, 1, 3],
                    'volume_quote_buy': [3, 3, 4, 5, 6],
                    'volume_quote_sell': [3, 3, 4, 5, 6],
                    'ticks_buy': [3, 3, 4, 5, 6],
                    'ticks_sell': [3, 3, 4, 5, 6],
                    'volume_quote': [6, 6, 8, 10, 12],
                    'custom_col': [.1, .3, .4, .5, .6]
                },
                index=pd.to_datetime(['22/10/2019 12:00:00', '22/10/2019 12:00:02',
                                      '22/10/2019 12:00:04', '22/10/2019 12:00:05',
                                      '22/10/2019 12:00:05'])
            )
        ),
        # fixed_ticks
        (
            bars.fixed_ticks, dict(threshold=0.5),
            pd.DataFrame(
                {
                    'open': [1, 3, 4, 5, 6],
                    'high': [3, 4, 5, 6, 7],
                    'low': [1, 3, 4, 5, 6],
                    'close': [2, 3, 4, 5, 6],
                    'volume_buy': [30, 30, 40, 50, 60],
                    'volume_sell': [1, 2, 0, 1, 3],
                    'volume_quote_buy': [3, 3, 4, 5, 6],
                    'volume_quote_sell': [3, 3, 4, 5, 6],
                    'ticks_buy': [3, 3, 4, 5, 6],
                    'ticks_sell': [3, 3, 4, 5, 6],
                    'ticks': [6, 6, 8, 10, 12],
                    'custom_col': [.1, .3, .4, .5, .6]
                },
                index=pd.to_datetime(['22/10/2019 12:00:00', '22/10/2019 12:00:02',
                                      '22/10/2019 12:00:04', '22/10/2019 12:00:05',
                                      '22/10/2019 12:00:05'])
            )
        ),
        # adaptive_volume
        (
            bars.adaptive_volume, dict(avg_per=1, window=2),
            pd.DataFrame(
                {
                    'open': [3, 4, 5, 6],
                    'high': [4, 5, 6, 7],
                    'low': [3, 4, 5, 6],
                    'close': [3, 4, 5, 6],
                    'volume_buy': [30, 40, 50, 60],
                    'volume_sell': [2, 0, 1, 3],
                    'volume_quote_buy': [3, 4, 5, 6],
                    'volume_quote_sell': [3, 4, 5, 6],
                    'ticks_buy': [3, 4, 5, 6],
                    'ticks_sell': [3, 4, 5, 6],
                    'volume': [32, 40, 51, 63],
                    'custom_col': [.3, .4, .5, .6]
                },
                index=pd.to_datetime(['22/10/2019 12:00:02', '22/10/2019 12:00:04',
                                      '22/10/2019 12:00:05', '22/10/2019 12:00:05'])
            )
        ),
        # adaptive_quote_volume
        (
            bars.adaptive_quote_volume, dict(avg_per=1, window=2),
            pd.DataFrame(
                {
                    'open': [3, 4, 5, 6],
                    'high': [4, 5, 6, 7],
                    'low': [3, 4, 5, 6],
                    'close': [3, 4, 5, 6],
                    'volume_buy': [30, 40, 50, 60],
                    'volume_sell': [2, 0, 1, 3],
                    'volume_quote_buy': [3, 4, 5, 6],
                    'volume_quote_sell': [3, 4, 5, 6],
                    'ticks_buy': [3, 4, 5, 6],
                    'ticks_sell': [3, 4, 5, 6],
                    'volume_quote': [6, 8, 10, 12],
                    'custom_col': [.3, .4, .5, .6]
                },
                index=pd.to_datetime(['22/10/2019 12:00:02', '22/10/2019 12:00:04',
                                      '22/10/2019 12:00:05', '22/10/2019 12:00:05'])
            )
        ),
        # adaptive_ticks
        (
            bars.adaptive_ticks, dict(avg_per=1, window=2),
            pd.DataFrame(
                {
                    'open': [3, 4, 5, 6],
                    'high': [4, 5, 6, 7],
                    'low': [3, 4, 5, 6],
                    'close': [3, 4, 5, 6],
                    'volume_buy': [30, 40, 50, 60],
                    'volume_sell': [2, 0, 1, 3],
                    'volume_quote_buy': [3, 4, 5, 6],
                    'volume_quote_sell': [3, 4, 5, 6],
                    'ticks_buy': [3, 4, 5, 6],
                    'ticks_sell': [3, 4, 5, 6],
                    'ticks': [6, 8, 10, 12],
                    'custom_col': [.3, .4, .5, .6]

                },
                index=pd.to_datetime(['22/10/2019 12:00:02', '22/10/2019 12:00:04',
                                      '22/10/2019 12:00:05', '22/10/2019 12:00:05'])
            )
        ),
        # fixed_percent_fixed_ticks
        (
            bars.fixed_percent_fixed_ticks, dict(percent_threshold=0.5, series_threshold=0.7),
            pd.DataFrame(
                {
                    'open': [1, 3, 6],
                    'high': [3, 6, 7],
                    'low': [1, 3, 6],
                    'close': [2, 5, 6],
                    'volume_buy': [30, 120, 60],
                    'volume_sell': [1, 3, 3],
                    'volume_quote_buy': [3, 12, 6],
                    'volume_quote_sell': [3, 12, 6],
                    'ticks_buy': [3, 12, 6],
                    'ticks_sell': [3, 12, 6],
                    'ticks': [6, 24, 12],
                    'custom_col': [.1, .3, .6]
                },
                index=pd.to_datetime(['22/10/2019 12:00:00', '22/10/2019 12:00:02', '22/10/2019 12:00:05'])
            )
        ),
        # fixed_percent_fixed_time
        (
            bars.fixed_percent_fixed_time, dict(threshold=0.05, period='2s'),
            pd.DataFrame(
                {
                    'open': [1, 4, 5],
                    'high': [4, 5, 7],
                    'low': [1, 4, 5],
                    'close': [3, 4, 6],
                    'volume_buy': [60, 40, 110],
                    'volume_sell': [3, 0, 4],
                    'volume_quote_buy': [6, 4, 11],
                    'volume_quote_sell': [6, 4, 11],
                    'ticks_buy': [6, 4, 11],
                    'ticks_sell': [6, 4, 11],
                    'custom_col': [.1, .4, .5]
                },
                index=pd.to_datetime(['22/10/2019 12:00:00', '22/10/2019 12:00:04', '22/10/2019 12:00:05'])
            )
        ),
        # adaptive_percent
        (
            bars.adaptive_percent, dict(avg_per=1, window=3),
            pd.DataFrame(
                {
                    'open': [1],
                    'high': [7],
                    'low': [1],
                    'close': [6],
                    'volume_buy': [210],
                    'volume_sell': [7],
                    'volume_quote_buy': [21],
                    'volume_quote_sell': [21],
                    'ticks_buy': [21],
                    'ticks_sell': [21],
                    'custom_col': [.1]
                },
                index=pd.to_datetime(['22/10/2019 12:00:00'])
            )
        )
    ]
)
def test_bars(func, params, test_res_df, input_df):
    if func in [bars.fixed_volume, bars.adaptive_volume]:
        input_df["volume"] = input_df.volume_buy + input_df.volume_sell

    if func in [bars.fixed_quote_volume, bars.adaptive_quote_volume]:
        input_df["volume_quote"] = input_df.volume_quote_buy + input_df.volume_quote_sell

    if func in [bars.fixed_ticks, bars.adaptive_ticks, bars.fixed_percent_fixed_ticks]:
        input_df["ticks"] = input_df.ticks_buy + input_df.ticks_sell

    res_df = func(input_df, **params)
    res_df = res_df[sorted(res_df.columns)]
    print(res_df.to_string())
    assert res_df.equals(test_res_df[sorted(test_res_df)])


@pytest.mark.parametrize(
    'func, params',
    [
        (bars.percent_o2c, dict(threshold=0.5)),
        (bars.fixed_volume, dict(threshold=20)),
        (bars.fixed_quote_volume, dict(threshold=600)),
        (bars.fixed_ticks, dict(threshold=30)),
        (bars.adaptive_volume, dict(avg_per=2, window=10)),
        (bars.adaptive_quote_volume, dict(avg_per=2, window=10)),
        (bars.adaptive_ticks, dict(avg_per=2, window=10)),
        (bars.fixed_percent_fixed_ticks, dict(percent_threshold=0.5, series_threshold=0.7)),
        (bars.fixed_percent_fixed_time, dict(threshold=0.05, period='2s')),
        (bars.adaptive_percent, dict(avg_per=1, window=3)),
    ],
    ids=[
        'percent_o2c', 'fixed_volume', 'fixed_quote_volume', 'fixed_ticks',
        'adaptive_volume', 'adaptive_quote_volume', 'adaptive_ticks',
        'fixed_percent_fixed_ticks', 'fixed_percent_fixed_time', 'adaptive_percent'
    ]
)
def test_bars_with_nans(func, params, ohlcv_df_nans: pd.DataFrame, ohlcv_df: pd.DataFrame):
    ohlcv_df["volume"] = ohlcv_df.volume_buy + ohlcv_df.volume_sell
    ohlcv_df["volume_quote"] = ohlcv_df.volume_quote_buy + ohlcv_df.volume_quote_sell
    ohlcv_df["ticks"] = ohlcv_df.ticks_buy + ohlcv_df.ticks_sell

    ohlcv_df_nans["volume"] = ohlcv_df_nans.volume_buy + ohlcv_df_nans.volume_sell
    ohlcv_df_nans["volume_quote"] = ohlcv_df_nans.volume_quote_buy + ohlcv_df_nans.volume_quote_sell
    ohlcv_df_nans["ticks"] = ohlcv_df_nans.ticks_buy + ohlcv_df_nans.ticks_sell

    res_df = func(ohlcv_df, **params)
    res_df_nan = func(ohlcv_df_nans, **params)

    assert np.array_equal(res_df.values, res_df_nan.values)
    assert res_df.index.equals(res_df_nan.index)


@pytest.mark.parametrize(
    'data, res',
    [
        (
            dict(
                a=[1, 1, None, None, 1, 1, 1, None, 1],
                f=[0, 1, 0, 0, 0, 1, 0, 0, 1]
            ),
            np.array([0, 0, 0, 1, 0, 1, 0, 0, 1])
        ),
        (
            dict(
                a=[1, None, None],
                f=[1, 0, 0]
            ),
            np.array([0, 0, 1])
        ),
        (
            dict(
                a=[1, 1, 1],
                f=[1, 0, 0]
            ),
            np.array([1, 0, 0])
        ),
        (
            dict(
                a=[None, None, 1],
                f=[0, 0, 1]
            ),
            np.array([0, 0, 1])
        ),
    ]
)
def test_move_feature_by_nans(data, res):
    df = pd.DataFrame(data)
    assert np.array_equal(
        move_feature_by_nans(df.f, df.a).values,
        res
    )


def test_all_possible_percent_bars_shapes(ohlcv_df, n_bars=5, threshold=0.005):
    ohlcv_df = ohlcv_df[:len(ohlcv_df) // 6]
    res = next(bars.all_possible_percent_bars(ohlcv_df, threshold=threshold, n_bars=n_bars))
    assert res.shape == (n_bars, len(ohlcv_df.columns))

    size = 0
    for item in bars.all_possible_percent_bars(ohlcv_df, threshold=threshold, n_bars=n_bars):
        size += 1

    assert item is None
    assert size == len(ohlcv_df)
