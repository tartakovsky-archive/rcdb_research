from config_samples.stattests import stattests_config
import pandas as pd
from commons.features.parallel_calc_all import JobManager
import os
import pytest


def calc(config, bars, n_jobs=1, tmp="/tmp/fns"):
    try:
        os.makedirs(tmp)
    except FileExistsError:
        pass
    jm = JobManager(
        bars,
        config=config,
        n_jobs=n_jobs,
        temp_folder=tmp,
        batch_size=20,
        benchmark=False,
        # name_as_dict=True
    )
    job_result = jm.run_job()
    df = job_result.get_pandas()
    return df


@pytest.mark.parametrize('feature_idx', range(-1, len(list(stattests_config.items())[0][1])))
def test(feature_idx, ohlcv_df, tmpdir):
    data = ohlcv_df.copy()
    data['close_pct_change'] = data['close'].pct_change()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.dropna()
    if feature_idx == -1:
        config = stattests_config
    else:
        prefix, features = list(stattests_config.items())[0]
        config = {prefix: [features[feature_idx]]}
    result = calc(config, data.head(400), n_jobs=2, tmp=tmpdir.mkdir('fns'))

    nans = result.isna().sum()
    faulty = nans[nans > 300]

    print(faulty)
    assert list(faulty.index) == []
