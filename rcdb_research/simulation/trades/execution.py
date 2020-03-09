from __future__ import annotations

from typing import NamedTuple, List

from .exchange import BidAsk, Costs
from .position import PositionAction


class ExecutionResult(NamedTuple):
    action: PositionAction
    filled_size: float
    avg_price: float
    fee: float
    slippage: float


class ExecutionManager(NamedTuple):
    @staticmethod
    def execute(actions: List[PositionAction], price: BidAsk,
                costs: Costs, algo: ExecutionAlgo) -> List[ExecutionResult]:
        changes = [algo.execute(action, price, costs) for action in actions]

        # TODO:
        # introduce PositionChange object? return position_changes: [PositionChange] to track realized_pnl and so on?

        return changes


class ExecutionAlgo:
    def execute(self, action: PositionAction, price: BidAsk, costs: Costs) -> ExecutionResult:
        pass


class MMMEA(ExecutionAlgo):
    """
    [M]arket entry, [M]arket take profit, [M]arket stop loss [E]xecution [A]lgorithm

    Entry order: market
    Take profit order: market
    Stop loss order: market
    """

    def execute(self, action: PositionAction, bidask: BidAsk, costs: Costs) -> ExecutionResult:
        price = bidask.ask if action.direction == 1 else bidask.bid
        slipped_price = price * (1 + action.direction * abs(costs.slippage))

        execution_res = ExecutionResult(
            action=action,
            filled_size=action.size,
            avg_price=slipped_price,
            fee=abs(action.size) * costs.taker_fee,
            slippage=slipped_price - price,
        )

        return execution_res


class LMMEA(ExecutionAlgo):
    """
    [L]imit entry, [M]arket take profit, [M]arket stop loss [E]xecution [A]lgorithm

    Entry order: limit
    Take profit order: market
    Stop loss order: market
    """
    pass


class LLMEA(ExecutionAlgo):
    """
    [L]imit entry, [L]imit take profit, [M]arket stop loss [E]xecution [A]lgorithm

    Entry order: limit
    Take profit order: limit
    Stop loss order: market
    """
    pass
