import inspect
import os

import numpy as np
import pandas as pd


def get_inputs(features_list):
    inputs = set()
    for f in features_list:
        for name, param in inspect.signature(f).parameters.items():
            if param.annotation in (np.array, pd.core.series.Series):
                inputs.add(name)
    return tuple(sorted(inputs))


def get_namespaces_around(file):
    path = os.path.dirname(os.path.abspath(file))
    namespaces = [
        file.replace('.py', '') for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file[0] != '_'
    ]
    return namespaces
