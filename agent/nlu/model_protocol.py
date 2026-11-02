"""The pluggable understanding interface.

Deliberately narrow: a Model's only job is to turn raw user text (plus the
current state, for context-dependent disambiguation like "cancel" meaning
"reject this proposal" vs. "cancel my booking") into a structured
Understanding. It never decides which tool to call, never sets approval,
and never touches the provider API directly -- all of that lives in
orchestrator/agent_loop.py as deterministic code, regardless of which Model
is plugged in. That split is what lets "Approval enforced by code" hold
even if a different model (e.g. a live LLM) occasionally misreads the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class Understanding:
    intent: Optional[str] = None          # agent.core.state.Intent value, or None if no new intent this turn
    slot_updates: dict = field(default_factory=dict)
    confirmation: Optional[bool] = None   # True/False if this turn is an unambiguous confirm/deny, else None
    selection_index: Optional[int] = None  # 0-based index into currently presented candidates
    memory_forget_key: Optional[str] = None
    memory_confirm_key: Optional[str] = None
    memory_show_requested: bool = False
    raw_text: str = ""


class Model(Protocol):
    name: str

    def understand(self, text: str, state) -> Understanding:
        ...
