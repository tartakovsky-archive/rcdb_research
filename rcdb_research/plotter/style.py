from typing import Optional
import matplotlib.ticker as ticker
from matplotlib.ticker import Formatter, Locator


def fig_kwargs(figsize: tuple = (16, 7), dpi: int = 150, facecolor: str = "w", **kwargs) -> dict:
    # return args as dict
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}


def ax_kwargs(titlesize: int = 16, labelsize: int = 14, ticksize: int = 12,
              showx: bool = True, showy: bool = True, fontfamily: str = 'sans-serif',
              labelpad: int = 15, titlepad: int = 15, tickrotation: int = 0,
              xformatter: Formatter = ticker.FormatStrFormatter('%.2f'),
              yformatter: Formatter = ticker.FormatStrFormatter('%.2f'),
              xlocator: Locator = None,
              ylocator: Locator = None) -> dict:
    return locals()


def line_kwargs(linewidth: int = 2, marker: Optional[str] = None) -> dict:
    return locals()


def hist_kwargs(rwidth: float = 0.7) -> dict:
    return locals()


def suptitle_kwargs(fontsize=18, fontweight='500', **kwargs) -> dict:
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}
