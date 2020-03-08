from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, List, Dict

from .exchange import BidAsk, Costs, SupportedExchange, Pair
from .position import PositionAction


class ExecutionAlgo:
    @staticmethod
    def execute(action: PositionAction, price: BidAsk, costs: Costs):
        # if isinstance(action, OpenAction):
        #     pass
        # elif isinstance(action, AddAction):
        #     pass
        # elif isinstance(action, ReduceAction):
        #     pass
        # elif isinstance(action, CloseAction):
        #     pass

        raise NotImplmentedError


class ExecutionResult(NamedTuple):
    action: PositionAction
    filled_size: float
    avg_price: float
    fee: float
    slippage: float


class ExecutionManager(NamedTuple):
    exchange: SupportedExchange  # TODO: make Exchange a base class, use it as type
    pair: Pair

    @staticmethod
    def execute(changes: Dict[Pair, List[PositionAction]], price: BidAsk, costs: Costs, algo: ExecutionAlgo):
        # for action in actions:
        #
        #     if isinstance(action, OpenAction):
        #         realized_pnl = 0
        #         pass
        #     elif isinstance(action, AddAction):
        #         realized_pnl = 0
        #         pass
        #     elif isinstance(action, ReduceAction):
        #         realized_pnl = 0  # Not zero, calculate
        #         pass
        #     elif isinstance(action, CloseAction):
        #         realized_pnl = 0  # Not zero, calculate
        #         pass

        # TODO:
        # return: (new_position, taken_actions: [PositionAction], exec_results: [ExecutionResult])
        # introduce PositionChange object? return position_changes: [PositionChange] to track realized_pnl and so on?
        # apply each executed action to current position to get series of states
        # sum realized pnl on each iteration for reduce and close actions

        pass


class MMMEA(ExecutionAlgo):
    """
    [M]arket entry, [M]arket take profit, [M]arket stop loss [E]xecution [A]lgorithm

    Entry order: market
    Take profit order: market
    Stop loss order: market
    """

    @staticmethod
    def execute(action: PositionAction, bidask: BidAsk, costs: Costs) -> ExecutionResult:
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
