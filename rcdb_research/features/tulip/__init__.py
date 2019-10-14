# flake8: noqa
from rcdb_research.features.tulip import bbands, rsi, macd, stoch, cci, psar, adx, roc, willr, obv, bop
from rcdb_research.features.tulip._utils import get_sub_inputs, calc_all, NAMESPACES

namespaces = NAMESPACES
inputs = get_sub_inputs(namespaces)
