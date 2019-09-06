from fnlib import entropy

from . import utils
from commons.features.parallel_calc_all import km


window = utils.pct_range(1, 100, 1, mult_step=0.03)

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
