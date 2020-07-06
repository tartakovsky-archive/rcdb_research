import pandas as pd
import numpy as np
import logging

from typing import List


def select_k_best(X: pd.DataFrame,
                  scores: pd.DataFrame,
                  clusters: List[dict],
                  k: int,
                  by: str = 'rank',
                  ascending: bool = True):
    pass
