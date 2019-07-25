# from collections import Counter
import inspect

import numpy as np
# import pandas as pd


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def mne_doc_helper(mne_func):
    def inner(func):
        func.__doc__ = mne_func.__doc__\
            .replace("The signals.\n", "")\
            .replace(
                "data : ndarray, shape (n_channels, n_times)\n",
                "series : np.array\n\tInput data.\n\n    window : int\n\tWindow size\n\n"
            )
        return func
    return inner


def feature_filter(o):
    try:
        if inspect.isfunction(o) and o.__name__[0] == "f":
            int(o.__name__[1:])
            return True

    except ValueError:
        pass

    return False
