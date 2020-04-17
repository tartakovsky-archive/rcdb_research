from typing import Optional


def fig_kwargs(figsize: tuple = (16, 7), dpi: int = 100, facecolor: str = "w", **kwargs) -> dict:
    # return args as dict
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}


def ax_kwargs(titlesize: int = 15, labelsize: int = 14, ticksize: int = 14,
              showx: bool = True, showy: bool = True, fontfamily: str = 'monospace',
              labelpad: int = 15, titlepad: int = 25, tickrotation: int = 0) -> dict:
    return locals()


def line_kwargs(linewidth: int = 2, marker: Optional[str] = None) -> dict: return locals()


def suptitle_kwargs(fontsize=16, **kwargs) -> dict:
    params = locals()
    del params['kwargs']
    return {**params, **kwargs}
