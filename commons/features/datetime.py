import numpy as np


def _get_obj_attr(objs: np.array, attr: str, func: bool = False) -> np.array:
    if func:
        return np.array([getattr(dt, attr)() for dt in objs.tolist()])

    return np.array([getattr(dt, attr) for dt in objs.tolist()])


def sec_of_min(datetimes: np.array) -> np.array:
    """
    Extract seconds of minute
    :param datetimes: np.array of native python datetime
    :return:
    """
    return _get_obj_attr(datetimes, "second")


def min_of_hour(datetimes: np.array) -> np.array:
    """
    Extract minutes
    :param datetimes:
    :return:
    """
    return _get_obj_attr(datetimes, "minute")


def hour_of_day(datetimes: np.array) -> np.array:
    """
    Extract hour
    :param datetimes:
    :return:
    """
    return _get_obj_attr(datetimes, "hour")


def day_of_month(datetimes: np.array) -> np.array:
    """
    Extract day of month
    :param datetimes:
    :return:
    """
    return _get_obj_attr(datetimes, "day")


def day_of_week(datetimes: np.array) -> np.array:
    """
    Extract day of week
    :param datetimes:
    :return:
    """
    return _get_obj_attr(datetimes, "weekday", func=True)


def day_of_year(datetimes: np.array) -> np.array:
    """
    Extract day of year
    :param datetimes:
    :return:
    """
    return np.array([dt.timetuple().tm_yday for dt in datetimes.tolist()])


_week_of_month = np.vectorize(
    lambda dt: int(
        np.ceil(
            (dt.replace(day=1).weekday() + dt.day) / 7.
        )
    )
)


def week_of_month(datetimes: np.array) -> np.array:
    """
    Extract week of month
    :param datetimes:
    :return:
    """
    return _week_of_month(datetimes)


def week_of_year(datetimes: np.array) -> np.array:
    """
    Extract week of year
    :param datetimes:
    :return:
    """
    return np.array([dt.isocalendar()[1] for dt in datetimes.tolist()])


def month_of_year(datetimes: np.array) -> np.array:
    """
    Extract week of year
    :param datetimes:
    :return:
    """
    return _get_obj_attr(datetimes, "month")
