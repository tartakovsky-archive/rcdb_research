import inspect
import os

import numpy as np


def get_inputs(features_list, exclude=[]):
    inputs = set()
    for f in features_list:
        for name, param in inspect.signature(f).parameters.items():
            if param.annotation == np.array and name not in exclude:
                inputs.add(name)
    return tuple(sorted(inputs))


def get_namespaces_around(file):
    """Returens subneamespaces list around file.

    Looking for python modules that locates around file. If the module name
    doesn't start with "_" then the returned list contains that name.
    You should to name your util modules beginning with "_" if you going to use
    this function.

    :param file: the file around which subnamespaces are located
    :return: list of namespaces
    """
    path = os.path.dirname(os.path.abspath(file))
    namespaces = [
        file.replace('.py', '') for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file[0] != '_'
    ]
    return namespaces
