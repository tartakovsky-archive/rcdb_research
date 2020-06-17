from typing import Optional
import matplotlib.ticker as ticker
from matplotlib.ticker import Formatter, Locator


def fig_kwargs(figsize: tuple = (16, 7),
               dpi: int = 150,
               facecolor: str = 'w',
               constrained_layout=True,
               **kwargs) -> dict:
    # return args as dict
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}


def ax_kwargs(titlesize: int = 16,
              labelsize: int = 14,
              ticksize: int = 12,
              facecolor: str = 'w',
              fontfamily: str = 'sans-serif',
              labelpad: int = 15,
              titlepad: int = 15,
              xtickrotation: int = 0,
              ytickrotation: int = 0,
              tick_params: Optional[dict] = None,
              xformatter: Optional[Formatter] = ticker.FormatStrFormatter('%.2f'),
              yformatter: Optional[Formatter] = ticker.FormatStrFormatter('%.2f'),
              xlocator: Optional[Locator] = None,
              ylocator: Optional[Locator] = None,
              **kwargs) -> dict:
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}


def line_kwargs(linewidth: int = 2, marker: Optional[str] = None, **kwargs) -> dict:
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}


def hist_kwargs(**kwargs) -> dict:
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}


def suptitle_kwargs(fontsize=18, fontweight='500', **kwargs) -> dict:
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}
