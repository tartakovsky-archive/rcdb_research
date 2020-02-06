import numba  # noqa

import numpy as np


@numba.jit
def symlog(x, C=0):
    return np.sign(x) * (np.log2(1 + np.abs(x) / (10 ** C)))


def symscale(x: np.array, center: float = 0, to_range: tuple = (-1, 0, 1)) -> np.array:
    x_above = x[x >= center]
    x_below = x[x <= center]

    scaled_above = np.interp(x_above, (x_above.min(), x_above.max()), (to_range[1], to_range[2]))
    scaled_below = np.interp(x_below, (x_below.min(), x_below.max()), (to_range[0], to_range[1]))

    x_scaled = np.zeros(x.size)
    x_scaled[x >= center] = scaled_above
    x_scaled[x <= center] = scaled_below
    x_scaled[x == center] = to_range[1]

    return x_scaled
