from __future__ import annotations
from dataclasses import dataclass
import weakref
from typing import Optional, Dict, List

from .exchange import Pair


@dataclass
class Position:
    pair: Pair
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
    def diff(pair: Pair, current_position: Optional[Position], desired_position: Optional[Position]) -> Dict[
        Pair,
        List[PositionAction]
    ]:

        if desired_position is None and current_position is None:
            # No action needs to be taken
            return {pair: []}

        if desired_position is None and current_position is not None:
            # Close position, don't open new one
            return {pair: [ClosePosition(size=-current_position.size)]}

        if desired_position is not None and current_position is None:
            # There is no position, open new one
            return {pair: [OpenPosition(size=desired_position.size)]}

        if desired_position.direction == current_position.direction:
            # Stay in the same direction
            size_diff = desired_position.size - current_position.size

            if abs(desired_position.size) > abs(current_position.size):
                # Add to position
                return {pair: [AddToPosition(size=size_diff)]}

            elif abs(desired_position.size) < abs(current_position.size):
                # Reduce position
                return {pair: [ReducePosition(size=size_diff)]}

        if desired_position.direction != current_position.direction:
            # Reverse direction. Close existing position first, open new one
            return {
                pair: [
                    ClosePosition(size=-current_position.size),
                    OpenPosition(size=desired_position.size),
                ]
            }

        # TODO: that return should never happen. Raise exception if it does
        return {}


@dataclass
class PositionAction:
    size: float

    @property
    def direction(self) -> float:
        return 1 if self.size > 0 else -1 if self.size < 0 else 0


@dataclass
class OpenPosition(PositionAction):
    pass


@dataclass
class AddToPosition(PositionAction):
    pass


@dataclass
class ReducePosition(PositionAction):
    pass


@dataclass
class ClosePosition(PositionAction):
    pass
