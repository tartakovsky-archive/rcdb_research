from __future__ import annotations

from dataclasses import dataclass
import weakref
from typing import Optional, List, Tuple, NamedTuple
import numpy as np


@dataclass
class Position:
    size: float
    avg_price: float

    def __post_init__(self):
        self.metrics = PositionMetrics(self)

    @property
    def direction(self) -> float:
        return 1 if self.size > 0 else -1 if self.size < 0 else 0


class PositionMetrics:
    def __init__(self, position: Position):
        self.position = weakref.proxy(position)


class PositionManager:
    @staticmethod
    def merge_change(position: Optional[Position], change: PositionChange) -> Tuple[Optional[Position], float]:
        """
        Get new position and realized_pnl after incorporating a change
        Method 2 from here: https://www.deltastock.com/english/education/average-price.asp

        Parameters
        ----------
        position : Position
        change : PositionChange

        Returns
        -------
        Tuple[Optional[Position], float]
            A tuple of new Position instance (or None) and float for realized PnL

        """
        action = change.action

        if position is None and not isinstance(action, OpenPosition):
            raise ValueError('Current position can only be None if required action is OpenPosition')

        if isinstance(action, OpenPosition):
            new_position = Position(size=change.size, avg_price=change.avg_price)
            realized_pnl = change.fee
            return new_position, realized_pnl

        elif isinstance(action, AddToPosition):
            size = position.size + change.size
            avg_price = np.average([position.avg_price, change.avg_price], weights=[position.size, change.size])

            new_position = Position(size=size, avg_price=avg_price)
            realized_pnl = change.fee
            return new_position, realized_pnl

        elif isinstance(action, ReducePosition) or isinstance(action, ClosePosition):
            size = position.size + change.size
            avg_price = position.avg_price

            price_change_pct = change.avg_price / position.avg_price - 1
            pnl_pct = position.direction * price_change_pct
            pnl_abs = abs(change.size) * pnl_pct

            new_position = Position(size=size, avg_price=avg_price)
            realized_pnl = pnl_abs + change.fee
            return new_position, realized_pnl

        else:
            return position, 0

    @staticmethod
    def diff(current_position: Optional[Position], desired_position: Optional[Position]) -> List[PositionAction]:

        if desired_position is None and current_position is None:
            # No action needs to be taken
            return []

        if desired_position is None and current_position is not None:
            # Close position, don't open new one
            return [ClosePosition(size=-current_position.size)]

        if desired_position is not None and current_position is None:
            # There is no position, open new one
            return [OpenPosition(size=desired_position.size)]

        if desired_position.direction == current_position.direction:
            # Stay in the same direction
            size_diff = desired_position.size - current_position.size

            if abs(desired_position.size) > abs(current_position.size):
                # Add to position
                return [AddToPosition(size=size_diff)]

            elif abs(desired_position.size) < abs(current_position.size):
                # Reduce position
                return [ReducePosition(size=size_diff)]

        if desired_position.direction != current_position.direction:
            # Reverse direction. Close existing position first, open new one
            return [
                ClosePosition(size=-current_position.size),
                OpenPosition(size=desired_position.size),
            ]

        # TODO: that return should never happen. Raise exception if it does
        return []


class PositionChange(NamedTuple):
    action: PositionAction
    size: float
    avg_price: float
    fee: float
    slippage: float


@dataclass
class PositionAction:
    size: float

    @property
    def direction(self) -> float:
        return 1 if self.size > 0 else -1 if self.size < 0 else 0


class OpenPosition(PositionAction):
    pass


class AddToPosition(PositionAction):
    pass


class ReducePosition(PositionAction):
    pass


class ClosePosition(PositionAction):
    pass
