"""Bounded execution (must-have #5): hard caps on model turns, tool calls,
and wall-clock time. Every exit path -- normal completion, retries
exhausted, or a cap hit -- must still produce a clear user-facing message;
agent_loop.py's finalize() is what guarantees that, this module just tracks
and raises.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


class LimitExceeded(Exception):
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"limit exceeded: {kind} {detail}".strip())


@dataclass
class Budget:
    max_model_turns: int = 20
    max_tool_calls: int = 20
    max_wall_clock_seconds: float = 30.0


class BudgetTracker:
    def __init__(self, budget: Budget | None = None):
        self.budget = budget or Budget()
        self.model_turns = 0
        self.tool_calls = 0
        self._started_at = time.monotonic()

    def _check_wall_clock(self) -> None:
        elapsed = time.monotonic() - self._started_at
        if elapsed > self.budget.max_wall_clock_seconds:
            raise LimitExceeded("wall_clock", f"{elapsed:.1f}s > {self.budget.max_wall_clock_seconds}s")

    def note_model_turn(self) -> None:
        self.model_turns += 1
        if self.model_turns > self.budget.max_model_turns:
            raise LimitExceeded("model_turns", f"{self.model_turns} > {self.budget.max_model_turns}")
        self._check_wall_clock()

    def note_tool_call(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.budget.max_tool_calls:
            raise LimitExceeded("tool_calls", f"{self.tool_calls} > {self.budget.max_tool_calls}")
        self._check_wall_clock()
