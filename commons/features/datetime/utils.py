from datetime import timedelta

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from workalendar.registry import registry


def is_date_in_timestamp(timestamp: np.array, timestamp_check: np.array):
    df = pd.DataFrame(index=timestamp)
    df["checked"] = 0
    for dt in timestamp_check:
        dt = str(dt)
        if dt in df.index:
            df.loc[dt, "checked"] = 1
    return df.checked.values


def is_hour_away(timestamp: np.array, search_timestamp: np.array):
    idxs = search_timestamp.searchsorted(
        timestamp,
        side="left"
    )

    idxs[idxs == len(idxs)] = len(idxs) - 1
    timediff = np.vectorize(timedelta.total_seconds)(search_timestamp[idxs] - timestamp)
    return (0. <= timediff) & (timediff <= 3600.) * 1


def supported_countries():
    return registry.region_registry.keys()


def get_market_calendar(market_name: str):
    return mcal.get_calendar(market_name)


def get_market_calendar_schedule(market_name: str, timestamps: np.array):
    start_date, end_date = timestamps[0].date(), timestamps[-1].date()
    calendar = get_market_calendar(market_name)
    return calendar.schedule(
        start_date=start_date,
        end_date=end_date + timedelta(days=2)
    )


def get_market_open_close(market_name: str, timestamps):
    calendar_schedule = get_market_calendar_schedule(market_name, timestamps)

    market_open = pd.DatetimeIndex(calendar_schedule.market_open.values).tz_localize(tz='UTC').to_pydatetime()
    market_close = pd.DatetimeIndex(calendar_schedule.market_close.values).tz_localize(tz='UTC').to_pydatetime()

    return market_open, market_close


def get_holiday_calendar(country_name: str):
    return registry.get_calendar_class(country_name)()


def get_holidays(country_name: str, timestamps: np.array):
    calendar = get_holiday_calendar(country_name)
    holidays = np.array([])

    years = np.unique(
        np.vectorize(lambda x: x.year)(timestamps)
    )

    for year in years:
        holidays = np.concatenate(
            (holidays, np.array([x[0] for x in calendar.holidays(year)])),
            axis=None
        )
    return holidays
