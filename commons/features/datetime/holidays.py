from datetime import timedelta, datetime
from typing import List, Dict

import pytz
import pandas as pd
import numpy as np
from workalendar.registry import registry
from workalendar.core import Calendar

from . import utils


def calc_all(data: pd.DataFrame, param_sets: List[Dict], column_names=None) -> pd.DataFrame:
    """
    Calculate all holidays features.
    See supported countries at holidays.utils.supported_countries()

    :param pd.DataFrame data: df with DateTimeIndex. No required columns
    :param List[Dict] param_sets: list of dict with country name. Example [{"country_name": "US"}, ...]
    :param column_names: unused template parameter
    :return: df with features
    """
    df = data.copy()

    df["date"] = df.index.date
    years = df.date.map(lambda x: x.year).unique()

    timestamp = data.index.to_pydatetime()

    supported_countries = utils.supported_countries()
    for ps in param_sets:
        name = ps["country_name"]
        if name not in supported_countries:
            raise ValueError(f"Unsupported country {name}. Choose from supported: {supported_countries}")

    for ps in param_sets:
        country_name = ps["country_name"]
        calendar = registry.get_calendar_class(country_name)()

        holidays = np.array([])
        for year in years:
            holidays = np.concatenate(
                (holidays, np.array([x[0] for x in calendar.holidays(year)])),
                axis=None
            )

        df[f"f1{country_name}"] = f1(timestamp, calendar)
        df[f"f2{country_name}"] = f2(timestamp, calendar)
        df[f"f3{country_name}"] = f3(timestamp, holidays)
        df[f"f4{country_name}"] = f4(timestamp, holidays)
        df[f"f5{country_name}"] = f5(timestamp, holidays)

    df["f6"] = f6(timestamp)
    return df.drop("date", axis=1)


def f1(timestamp: np.array, calendar: Calendar) -> np.array:
    """
    Check if the timestamps are holidays
    :param np.array timestamp: array of datetime`s
    :param workalendar.core.Calendar calendar: instance of workalendar calendar
    :return: array of 0 and 1 for each timestamp
    """
    return np.vectorize(calendar.is_holiday)(timestamp) * 1


def f2(timestamp: np.array, calendar: Calendar) -> np.array:
    """
    Check if the timestamps are working days
    :param np.array timestamp: array of datetime`s
    :param workalendar.core.Calendar calendar: instance of workalendar calendar
    :return: array of 0 and 1 for each timestamp
    """
    return np.vectorize(calendar.is_working_day)(timestamp) * 1


def f3(timestamp: np.array, holidays: np.array) -> np.array:
    """
    Calculate distance to the holiday in seconds
    :param np.array timestamp: array of datetime`s
    :param np.array holidays: array of datetime.date`s
    :return: array with seconds for each timestamp
    """
    def prepare_timedelta_to_seconds(td):
        seconds = td.total_seconds()
        if seconds < 0:
            seconds = 0.0
        return seconds

    idxs = holidays.searchsorted(
        np.vectorize(datetime.date)(timestamp),
        side="left"
    )

    date_to_utcdatetime = np.vectorize(
        lambda d: datetime.combine(d, datetime.min.time()).replace(tzinfo=pytz.UTC)
    )

    holidays_td = date_to_utcdatetime(holidays)[idxs] - timestamp
    return np.vectorize(prepare_timedelta_to_seconds)(holidays_td)


def f4(timestamp: np.array, holidays: np.array) -> np.array:
    """
    Check if the timestamps are day before the holidays
    :param timestamp: array of datetime`s
    :param holidays: array of datetime.date`s
    :return: array of 0 and 1 for each timestamp
    """
    return utils.is_date_in_timestamp(timestamp, holidays - timedelta(days=1))


def f5(timestamp: np.array, holidays: np.array) -> np.array:
    """
    Check if the timestamps are day after the holidays
    :param timestamp: array of datetime`s
    :param holidays: array of datetime.date`s
    :return: array of 0 and 1 for each timestamp
    """
    return utils.is_date_in_timestamp(timestamp, holidays + timedelta(days=1))


def f6(timestamp: np.array) -> np.array:
    """
    Check if the timestamps are weekend
    :param timestamp: array of datetime`s
    :return: array of 0 and 1 for each timestamp
    """
    return np.vectorize(lambda x: int(x.weekday() > 4))(timestamp)
