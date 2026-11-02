"""Explicit workflow state (must-have #1) plus the session persistence that
makes "restart mid-booking" recoverable (must-have #3 / eval #10).

Only known slots are ever asked for; a correction updates exactly the slot
it mentions (Slots.update ignores None values, so unmentioned slots are
never clobbered). Everything the orchestrator needs to resume a crashed
session -- stage, candidates, the active hold, the pending approval, the
idempotency key in flight -- is part of this dataclass and is written to
disk after every turn.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "sessions"


class Stage(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    SEARCHING = "searching"
    PRESENTING = "presenting"
    AWAITING_APPROVAL = "awaiting_approval"
    HOLDING = "holding"
    EXECUTING = "executing"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Intent(str, Enum):
    NEW_BOOKING = "new_booking"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    VIEW_BOOKINGS = "view_bookings"
    UNKNOWN = "unknown"


REQUIRED_SLOTS_FOR_BOOKING = ["service", "area", "time_window"]


@dataclass
class Slots:
    service: Optional[str] = None
    area: Optional[str] = None
    time_window: Optional[str] = None
    budget: Optional[int] = None
    provider_preference: Optional[str] = None
    earliest_time: Optional[str] = None

    def update(self, **kwargs) -> list:
        """Sets only the provided non-None fields; returns the list of field
        names actually changed (so the caller can log/trace it)."""
        changed = []
        for key, value in kwargs.items():
            if value is not None and getattr(self, key, None) != value:
                setattr(self, key, value)
                changed.append(key)
        return changed

    def missing(self, required: list) -> list:
        return [f for f in required if getattr(self, f, None) is None]

    def clear(self, field: str) -> None:
        setattr(self, field, None)


@dataclass
class ConversationState:
    user_id: str
    session_id: str
    intent: str = Intent.UNKNOWN.value
    stage: str = Stage.IDLE.value
    slots: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    candidates_kind: str = "providers"  # "providers" | "bookings" -- what `candidates` holds
    selected_provider: Optional[dict] = None
    hold: Optional[dict] = None
    pending_approval: Optional[dict] = None
    current_idempotency_key: Optional[str] = None
    target_booking_id: Optional[str] = None
    last_booking_id: Optional[str] = None
    turn_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def get_slots(self) -> Slots:
        return Slots(**self.slots)

    def set_slots(self, slots: Slots) -> None:
        self.slots = asdict(slots)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationState":
        return cls(**d)

    def save(self, data_dir: Optional[Path] = None) -> None:
        self.updated_at = time.time()
        d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.user_id}.json").write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, user_id: str, data_dir: Optional[Path] = None) -> Optional["ConversationState"]:
        d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        path = d / f"{user_id}.json"
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def new(cls, user_id: str) -> "ConversationState":
        return cls(user_id=user_id, session_id=uuid.uuid4().hex[:10])

    def start_new_intent(self, intent: Intent) -> None:
        """Resets everything about the in-flight action. Stable preferences
        (area, budget, time_window, earliest_time) survive across intents in
        the same session, but `service` and `provider_preference` are
        request-specific and are cleared here -- otherwise a stray leftover
        like service="plumber" from an earlier search could get silently
        reused to resolve an unrelated "cancel my booking" to the wrong
        booking (see BUGS.md). The caller re-applies this turn's own slot
        extraction afterwards, so an explicit mention (e.g. "cancel my
        electrician booking") still sets it correctly."""
        self.intent = intent.value
        self.stage = Stage.COLLECTING.value
        self.candidates = []
        self.candidates_kind = "providers"
        self.selected_provider = None
        self.hold = None
        self.pending_approval = None
        self.current_idempotency_key = None
        self.target_booking_id = None
        slots = self.get_slots()
        slots.service = None
        slots.provider_preference = None
        self.set_slots(slots)
