import inspect

import numpy as np
import pandas as pd


def get_inputs(features_list):
    inputs = set()
    for f in features_list:
        for name, param in inspect.signature(f).parameters.items():
            if param.annotation in (np.array, pd.core.series.Series):
                inputs.add(name)
    return tuple(inputs)
