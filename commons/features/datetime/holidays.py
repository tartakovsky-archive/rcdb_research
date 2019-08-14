from datetime import timedelta, datetime
from typing import List, Dict

import pytz
import pandas as pd
import numpy as np

from . import utils
from commons.utils import calc_all_parallel

PREFIX = "dt_holidays"


def calc_all(data: pd.DatetimeIndex, param_sets: List[Dict], column_names=None, n_jobs=-1) -> pd.DataFrame:
    """
    Calculate all holidays features.
    See supported countries at holidays.utils.supported_countries()

    :param pd.DataFrame data: df with DateTimeIndex. No required columns
    :param List[Dict] param_sets: list of dict with country name. Example [{"country_name": "US"}, ...]
    :param column_names: unused template parameter
    :return: df with features
    """
    return calc_all_parallel(
        dict(datetime_holidays=calc_all_generator(data, param_sets, column_names)),
        n_jobs=n_jobs
    )


def calc_all_generator(data: pd.DatetimeIndex, param_sets: List[Dict], column_names=None) -> pd.DataFrame:
    """
    Calculate all holidays features.
    See supported countries at holidays.utils.supported_countries()

    :param pd.DataFrame data: df with DateTimeIndex. No required columns
    :param List[Dict] param_sets: list of dict with country name. Example [{"country_name": "US"}, ...]
    :param column_names: unused template parameter
    :return: df with features
    """
    if type(data) != pd.DatetimeIndex:
        raise ValueError("calc_all `data` arg expected to be `pd.DateTimeIndex`")

    ts_col = "ts"
    df = pd.DataFrame(dict(data=data.values))
    df[ts_col] = df['data']
    df.set_index('data', inplace=True)

    timestamps = data.to_pydatetime()

    supported_countries = utils.supported_countries()
    for ps in param_sets:
        name = ps["country_name"]
        if name not in supported_countries:
            raise ValueError(f"Unsupported country {name}. Choose from supported: {supported_countries}")

    calc_calls = [
        [f6, [timestamps]]
    ]

    for ps in param_sets:
        country_name = ps["country_name"]

        calc_calls += [
            [f1, [timestamps, country_name]],
            [f2, [timestamps, country_name]],
            [f3, [timestamps, country_name]],
            [f4, [timestamps, country_name]],
            [f5, [timestamps, country_name]]
        ]

    return calc_calls


def f1(timestamps: np.array, country_name: str) -> np.array:
    """
    Check if the timestamps are holidays
    :param np.array timestamps: array of datetime`s
    :param str country_name: name of the country for get instance of workalendar calendar
    :return: array of 0 and 1 for each timestamp
    """
    calendar = utils.get_holiday_calendar(country_name)
    return np.vectorize(calendar.is_holiday)(timestamps) * 1


def f2(timestamps: np.array, country_name: str) -> np.array:
    """
    Check if the timestamps are working days
    :param np.array timestamps: array of datetime`s
    :param str country_name: name of the country for get instance of workalendar calendar
    :return: array of 0 and 1 for each timestamp
    """
    calendar = utils.get_holiday_calendar(country_name)
    return np.vectorize(calendar.is_working_day)(timestamps) * 1


def f3(timestamps: np.array, country_name: str) -> np.array:
    """
    Calculate distance to the holiday in seconds
    :param np.array timestamps: array of datetime`s
    :param str country_name: name of the country for get instance of workalendar calendar
    :return: array with seconds for each timestamp
    """

    def prepare_timedelta_to_seconds(td):
        seconds = td.total_seconds()
        if seconds < 0:
            seconds = 0.0
        return seconds

    holidays = utils.get_holidays(country_name, timestamps)

    idxs = holidays.searchsorted(
        np.vectorize(datetime.date)(timestamps),
        side="left"
    )

    date_to_utcdatetime = np.vectorize(
        lambda d: datetime.combine(d, datetime.min.time()).replace(tzinfo=pytz.UTC)
    )

    holidays_td = date_to_utcdatetime(holidays)[idxs] - timestamps
    return np.vectorize(prepare_timedelta_to_seconds)(holidays_td)


def f4(timestamps: np.array, country_name: str) -> np.array:
    """
    Check if the timestamps are day before the holidays
    :param timestamps: array of datetime`s
    :param str country_name: name of the country for get instance of workalendar calendar
    :return: array of 0 and 1 for each timestamp
    """
    holidays = utils.get_holidays(country_name, timestamps)
    return utils.is_date_in_timestamp(timestamps, holidays - timedelta(days=1))


def f5(timestamps: np.array, country_name: str) -> np.array:
    """
    Check if the timestamps are day after the holidays
    :param timestamps: array of datetime`s
    :param str country_name: name of the country for get instance of workalendar calendar
    :return: array of 0 and 1 for each timestamp
    """
    holidays = utils.get_holidays(country_name, timestamps)
    return utils.is_date_in_timestamp(timestamps, holidays + timedelta(days=1))


def f6(timestamps: np.array) -> np.array:
    """
    Check if the timestamps are weekend
    :param timestamps: array of datetime`s
    :return: array of 0 and 1 for each timestamp
    """
    return np.vectorize(lambda x: int(x.weekday() > 4))(timestamps)
