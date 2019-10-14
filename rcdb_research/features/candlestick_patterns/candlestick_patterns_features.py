from typing import List, Dict

import pandas as pd
import talib
from talib import get_function_groups


def calc_all(
    data: pd.DataFrame,
    param_set: List[Dict] = None,
    column_names: dict = dict(
        open = "open",
        high = "high",
        low = "low",
        close = "close",
    )
) -> pd.DataFrame:
    open = data[column_names["open"]]
    high = data[column_names["high"]]
    low = data[column_names["low"]]
    close = data[column_names["close"]]
    # CDL functions docs https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html
    CDL_functions = {func_name: getattr(talib, func_name) for func_name in get_function_groups()["Pattern Recognition"]}

    results = {cdl_name: cdl_func(open, high, low, close)
               for cdl_name, cdl_func in CDL_functions.items()}
    results_df = pd.DataFrame.from_dict(results, orient="index")

    return results_df.T
