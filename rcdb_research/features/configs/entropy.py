import numpy_ext as npext

from ..features import entropy
from ..job_manager import km


window = npext.expstep_range(1, 100, 1, step_mult=0.03)

sf = [100]
spectral_method = ['fft', 'welch']
bins = [4]

entropy_conf = dict(
    entropy=[
        dict(
            fn=entropy.app_entropy,
            pg=km(window=window),
            dm=km(series=['close', 'change', 'timediff', 'volume', 'ticks']),
        ),
        dict(
            fn=entropy.sample_entropy,
            pg=km(window=window),
            dm=km(series=['close', 'change', 'timediff', 'volume', 'ticks']),
        ),
        dict(
            fn=entropy.spectral_entropy,
            pg=km(window=window, sf=sf, method=spectral_method),
            dm=km(series=['close', 'change', 'timediff', 'volume', 'ticks']),
        ),
        dict(
            fn=entropy.svd_entropy,
            pg=km(window=window),
            dm=km(series=['close', 'change', 'timediff', 'volume', 'ticks']),
        ),
        dict(
            fn=entropy.perm_entropy,
            pg=km(window=window),
            dm=km(series=['close', 'change', 'timediff', 'volume', 'ticks']),
        ),
        dict(
            fn=entropy.binned_entropy,
            pg=km(window=window, max_bins=bins),
            dm=km(series=['close', 'change', 'timediff', 'volume', 'ticks']),
        ),
    ]
)
