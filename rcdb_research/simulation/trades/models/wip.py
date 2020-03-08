from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy

from typing import NamedTuple, Optional, List, FrozenSet, Dict
from enum import Enum
from collections import OrderedDict

import numpy as np
import pandas as pd

from .execution.actions import PositionAction, OpenAction, CloseAction, AddAction, ReduceAction












class OrderType(Enum):
    market = 'market'
    limit = 'limit'


class Order(NamedTuple):
    type: OrderType
    price: float
    size: float

    @property
    def direction(self) -> float:
        return 1 if self.size > 0 else -1 if self.size < 0 else 0

    def __str__(self):
        params = str(self.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
        return f"{self.__class__.__name__}({params})"

    def to_dict(self, prefix=""):
        return {
            f"{prefix}type": self.type.value,
            f"{prefix}price": self.price,
            f"{prefix}size": self.size
        }




class BarContext(NamedTuple):
    account_pre_trade: Account
    position_desired: Optional[DesiredPosition]
    order: Optional[Order]
    execution_result: Optional[ExecutionResult]
    account_post_trade: Account
    #
    # def __str__(self):
    #     order_params = str(self.order.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
    #     order_str = f"{self.order.__class__.__name__}({order_params})"
    #     d = self.to_dict()
    #     d['order'] = order_str
    #     params = str(d).strip("{}").replace(': ', '=').replace("'", '')
    #     return f"{self.__class__.__name__}({params})"
    #
    # def to_dict(self, prefix=""):
    #     d = dict(self._asdict())
    #
    #     d = {k: v.to_dict() for (k, v) in d.items() if v is not None}
    #
    #     return d

#     def to_df(self):
#         d = dict(self._asdict())

#         nd = OrderedDict()
#         nd.update(d['pre_trade_state'].to_dict(prefix="pre_"))

#         trade_signal = d.get('trade_signal', None)
#         if trade_signal is not None:
#             nd.update(trade_signal.to_dict())

#         order = d.get('order', None)
#         if order is not None:
#             nd.update({(f"order_{k}" if k == 'type' else k): v for (k, v) in order.to_dict().items()})

#         execution_result = d.get('execution_result', None)
#         if execution_result is not None:
#             nd.update({k: v for (k, v) in execution_result.to_dict().items() if k != 'order'})

#         nd.update(d['post_trade_state'].to_dict(prefix="post_"))

#         return pd.DataFrame([nd])
