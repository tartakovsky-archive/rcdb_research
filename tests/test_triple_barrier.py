import pytest
import numpy as np
import pandas as pd

from rcdb_research.labeling import triple_barrier_labeling


CLOSE = pd.Series(
    [
        95.35, 95.3, 100.0, 93.04, 96.35,
        93.19, 90.13, 92.37, 91.48,
        93.48, 95.487, 60.1, 100.24,
        100.12, 100.21, 100.12, 100.55,
        30.12, 100.21, 20.12, 43.55,

    ],
    index=pd.DatetimeIndex(
        [
            np.datetime64('2018-12-31T18:00:00'),

            np.datetime64('2019-01-01T00:00:00'),
            np.datetime64('2019-01-01T06:00:00'),
            np.datetime64('2019-01-01T12:20:00'),
            np.datetime64('2019-01-01T18:30:00'),

            np.datetime64('2019-01-02T01:00:00'),
            np.datetime64('2019-01-02T06:00:00'),
            np.datetime64('2019-01-02T12:00:00'),
            np.datetime64('2019-01-02T18:00:00'),

            np.datetime64('2019-01-03T00:00:00'),
            np.datetime64('2019-01-03T06:00:00'),
            np.datetime64('2019-01-03T12:00:00'),
            np.datetime64('2019-01-03T18:00:00'),

            np.datetime64('2019-01-04T00:00:00'),
            np.datetime64('2019-01-04T06:00:00'),
            np.datetime64('2019-01-04T12:00:00'),
            np.datetime64('2019-01-04T18:00:00'),

            np.datetime64('2019-01-05T00:00:00'),
            np.datetime64('2019-01-05T06:00:00'),
            np.datetime64('2019-01-05T12:00:00'),
            np.datetime64('2019-01-05T18:00:00'),
        ],
        name="timestamp"
    ),
    name="close",
)

WINDOW = 5
TEST_LABELS = np.array(
    [
        # bars which has not history for daily volatility
        np.nan, np.nan, np.nan, np.nan, np.nan,
        -1., 1., 1., 1., -1., -1., 1., 0.,
        -1., -1., -1., -1.,
        # not enough future bars (window size is 5)
        np.nan, np.nan, np.nan, np.nan
    ]
)


def test_triple_barrier_labeling():
    res = triple_barrier_labeling(CLOSE, daily_volatility_span=3, window=WINDOW)
    assert len(res) == len(CLOSE)

    assert np.isnan(res[-(WINDOW - 1):]).all()

    assert np.array_equal(res[~np.isnan(res)], TEST_LABELS[~np.isnan(TEST_LABELS)])


@pytest.mark.parametrize(
    "kwargs, barrier_excluded",
    [
        (dict(pt_coef=np.nan), 1.),
        (dict(sl_coef=np.nan), -1.),
    ]
)
def test_border_disabling(kwargs, barrier_excluded):
    res = triple_barrier_labeling(CLOSE, daily_volatility_span=3, window=WINDOW, **kwargs)
    assert not (res == barrier_excluded).any()
