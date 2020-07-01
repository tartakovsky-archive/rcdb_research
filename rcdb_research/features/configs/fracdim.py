from ..features import fracdim
from ..job_manager import km, t


window = [50]
fracdim_config = dict(
    fracdim=[
        dict(
            fn=fracdim.katz_fd,
            pg=km(window=window),
            dm=km(series=['close']),
            tr=[t.symlog()]
        ),
        dict(
            fn=fracdim.petrosian_fd,
            pg=km(window=window),
            dm=km(series=['close'])
        )
    ]
)
