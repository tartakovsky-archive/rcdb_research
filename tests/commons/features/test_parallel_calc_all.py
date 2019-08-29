import pytest

import pandas as pd

from commons.features.parallel_calc_all import JobManager, km, t
from commons.features.transformations import symlog2_
from commons.features.numba_example import f


@pytest.fixture(scope='module')
def df(ohlcv_df):
    df = ohlcv_df[:50].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@pytest.fixture(scope='module')
def calculated_feature_df(df):
    res = df[[]].copy()
    close = df.close.values
    res['namespace1.move_mean(a=close, window_arr=10)'] = f.move_mean(a=close, window_arr=10)

    close_symlog = symlog2_(close)
    res['namespace0.feature0(a=close.symlog(), window_arr=10)'] = f.move_mean(close_symlog, window_arr=10)
    res['namespace0.feature0(a=close.symlog(), window_arr=20)'] = f.move_mean(close_symlog, window_arr=20)
    return res


@pytest.mark.parametrize(
    'n_jobs',
    [1, -1, 2]
)
def test_JobManager_basic_usage(calculated_feature_df, df, tmp_path, n_jobs):
    config = dict(
        namespace0=[
            dict(
                fn=f.move_mean,
                alias='feature0',
                pg=km(window_arr=[10, 20]),
                dm=km(a=[km.col('close').t([t.symlog()])]),
            ),
        ],
        namespace1=[
            dict(
                fn=f.move_mean,
                pg=km(window_arr=[10]),
                dm=km(a=['close']),
            ),
        ]
    )

    temp_folder = str(tmp_path.resolve()) if n_jobs != 1 else None

    jm = JobManager(df, config=config, n_jobs=n_jobs, temp_folder=temp_folder, batch_size=300)
    job_result = jm.run_job()

    assert job_result.count_results() == len(calculated_feature_df.columns)

    res_df = job_result.get_pandas()

    assert (calculated_feature_df.index == res_df.index).all()
    assert set(calculated_feature_df.columns) == set(res_df.columns)

    for col in calculated_feature_df.columns:
        assert (calculated_feature_df[col] == res_df[col]).all()


def test_JobManager_temp_folder_None_for_parallel(ohlcv_df):
    with pytest.raises(AttributeError) as ex:
        JobManager(ohlcv_df, {}, n_jobs=-1)
        assert ex.match("temp_folder is required for parallel_execution")


def test_JobManager_df_with_tz_index(ohlcv_df):
    with pytest.raises(ValueError) as ex:
        JobManager(ohlcv_df, {})
        assert ex.match("Timezones not allowed in the input DataFrame. Use `df.index = pd.to_datetime(df.index)"
                        ".tz_localize(None)` to remove tz info.")
