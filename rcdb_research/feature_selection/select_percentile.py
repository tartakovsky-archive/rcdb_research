import pandas as pd
import numpy as np
import logging

from typing import List


def select_percentile(X: pd.DataFrame,
                      scores: pd.DataFrame,
                      clusters: List[dict],
                      percentile: float,
                      by: str = 'rank',
                      ascending: bool = True):
    pass
