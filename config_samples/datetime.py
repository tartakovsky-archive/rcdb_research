from rcdb_research.features import datetime
from rcdb_research.features.parallel_calc_all import km


datetime_config = dict(
    datetime=[
        dict(
            fn=datetime.sec_of_min,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.min_of_hour,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.hour_of_day,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.day_of_month,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.day_of_week,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.day_of_year,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.week_of_month,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.week_of_year,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetime.month_of_year,
            dm=km(datetimes=['timestamp']),
        ),
    ]
)
