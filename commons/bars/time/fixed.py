import pandas as pd

from commons import bars

# def fixed(ohlc, period):
#     bars.base.idx_to_column(ohlc)
#     bars.base.validate_columns(ohlc)
#
#     if not 0 < period:
#         raise AttributeError(f"Period must be greater than zero.")
#
#     ohlc = ohlc[bars.COLUMNS]
#     consolidated_bars = []
#     current_bar = None
#
#     for bar in ohlc.to_numpy():
#         if current_bar is None:
#             current_bar = list(bar)
#             count = 1
#         else:
#             bars.base.update(current_bar, bar)
#             count += 1
#
#         if (count == period):
#             consolidated_bars.append(current_bar)
#             current_bar = None
#
#     return bars.base.output_format(consolidated_bars)
