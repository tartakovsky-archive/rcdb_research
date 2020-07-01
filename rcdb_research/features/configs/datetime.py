from ..features import datetimes
from ..job_manager import km


datetime_config = dict(
    datetime=[
        dict(
            fn=datetimes.sec_of_min,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.min_of_hour,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.hour_of_day,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.day_of_month,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.day_of_week,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.day_of_year,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.week_of_month,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.week_of_year,
            dm=km(datetimes=['timestamp']),
        ),
        dict(
            fn=datetimes.month_of_year,
            dm=km(datetimes=['timestamp']),
        ),
    ]
)
