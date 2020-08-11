from functools import partial

from . import facade


# Fixed threshold
time = facade.time
percent_o2c = facade.percent_o2c
percent = facade.percent
all_possible_percent_bars = facade.all_possible_percent_bars
fixed_volume = partial(facade.fixed, column='volume')
fixed_quote_volume = partial(facade.fixed, column='volume_quote')
fixed_ticks = partial(facade.fixed, column='ticks')

# Adaptive threshold
adaptive_volume = partial(facade.adaptive, column='volume')
adaptive_quote_volume = partial(facade.adaptive, column='volume_quote')
adaptive_ticks = partial(facade.adaptive, column='ticks')

# Hybrid
fixed_percent_fixed_ticks = partial(facade.fixed_percent_fixed_series, series_column='ticks')

adaptive_percent = facade.adaptive_percent

# Fixed time fixed percent
fixed_percent_fixed_time = facade.fixed_percent_fixed_time


# Imbalance
imbalance = partial(facade.imbalance, column='volume')
imbalance_quote = partial(facade.imbalance, column='volume_quote')

imbalance_fixed = partial(facade.imbalance_fixed, column='volume')
imbalance_fixed_quote = partial(facade.imbalance_fixed, column='volume_quote')
