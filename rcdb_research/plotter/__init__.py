from .primitives import area, area_colors  # noqa
from .primitives import bars_pn, bars_colors, bars_legend  # noqa
from .primitives import line_pn, curve_colors, curve_legend  # noqa
from .primitives import hist, hist_colors  # noqa
from .primitives import line, line_colors  # noqa

from .reports import preds_report, preds_colors  # noqa
from .reports import proba_report, proba_colors  # noqa
from .reports import threshold_report, threshold_colors  # noqa
from .reports import trading_report, trading_colors  # noqa
from .reports import splits, splits_colors  # noqa


from .utils import second_index, configure_axis  # noqa

from .style import fig_kwargs, ax_kwargs, line_kwargs, hist_kwargs, suptitle_kwargs  # noqa

from .components import calibration, monte_carlo, distcomp, histcomp # noqa
