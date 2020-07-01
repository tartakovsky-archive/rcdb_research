import numpy as np


def convert_dt_type(datetimes: np.ndarray) -> np.ndarray:
    if datetimes.dtype == 'datetime64[ns]':
        return datetimes.astype('datetime64[s]')
    return datetimes


def _get_obj_attr(objs: np.ndarray, attr: str, func: bool = False) -> np.ndarray:
    objs = convert_dt_type(objs)

    if func:
        return np.array([getattr(dt, attr)() for dt in objs.tolist()])

    return np.array([getattr(dt, attr) for dt in objs.tolist()])


def sec_of_min(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract seconds of minute

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted seconds

    Examples
    --------
    >>> sec_of_min(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([11, 21,  1])
    """
    return _get_obj_attr(datetimes, "second")


def min_of_hour(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract minutes

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted minutes

    Examples
    --------
    >>> min_of_hour(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([44, 34,  2])
    """
    return _get_obj_attr(datetimes, "minute")


def hour_of_day(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract hour

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted hours

    Examples
    --------
    >>> hour_of_day(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([11, 12,  1])
    """
    return _get_obj_attr(datetimes, "hour")


def day_of_month(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract day of month

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted days

    Examples
    --------
    >>> day_of_month(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([27, 28,  1])
    """
    return _get_obj_attr(datetimes, "day")


def day_of_week(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract day of week

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted days

    Examples
    --------
    >>> day_of_week(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([1, 2, 6])
    """
    return _get_obj_attr(datetimes, "weekday", func=True)


def day_of_year(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract day of year

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted days

    Examples
    --------
    >>> day_of_year(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([239, 240, 335])
    """
    return np.array([dt.timetuple().tm_yday for dt in convert_dt_type(datetimes).tolist()])


_week_of_month = np.vectorize(
    lambda dt: int(
        np.ceil(
            (dt.replace(day=1).weekday() + dt.day) / 7.
        )
    )
)


def week_of_month(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract week of month

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted week numbers

    Examples
    --------
    >>> week_of_month(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([5, 5, 1])
    """
    return _week_of_month(convert_dt_type(datetimes).tolist())


def week_of_year(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract week of year

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted week numbers

    Examples
    --------
    >>> week_of_year(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([35, 35, 48])
    """
    return np.array([dt.isocalendar()[1] for dt in convert_dt_type(datetimes).tolist()])


def month_of_year(datetimes: np.ndarray) -> np.ndarray:
    """
    Extract month of year

    Parameters
    ----------
    datetimes : np.ndarray
        np.array of native python datetime

    Returns
    -------
    np.array
        array with extracted seconds

    Examples
    --------
    >>> month_of_year(
    ...     np.array(['2019-08-27T11:44:11', '2019-08-28T12:34:21', '2019-12-01T01:02:01'], dtype='datetime64[s]'))
    array([ 8,  8, 12])
    """
    return _get_obj_attr(convert_dt_type(datetimes), "month")
