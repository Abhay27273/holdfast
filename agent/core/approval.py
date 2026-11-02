"""Code-enforced approval records.

The critical invariant (must-have #2): approval is a *record the code
checks*, never something inferred from model output or raw tool args. A
PendingApproval is created only by the orchestrator when it has shown the
user a concrete proposal (exact action, provider, slot, price). It becomes
`confirmed` only when the orchestrator itself matches the user's very next
message against that specific proposal via `parse_confirmation`. Before
executing book/cancel/reschedule, the orchestrator recomputes a fingerprint
of what it is about to *actually* do and compares it to the approved
fingerprint -- any drift (e.g. hold_slot quoting a different price) voids
the approval automatically.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class PendingApproval:
    action: str  # "book" | "cancel" | "reschedule"
    provider_id: Optional[str] = None
    provider_name: Optional[str] = None
    slot: Optional[str] = None
    price: Optional[int] = None
    booking_id: Optional[str] = None
    proposal_text: str = ""
    created_at: float = 0.0
    confirmed: bool = False
    confirmed_at: Optional[float] = None

    def _key_payload(self, action=None, provider_id=None, slot=None, price=None, booking_id=None) -> dict:
        return {
            "action": action if action is not None else self.action,
            "provider_id": provider_id if provider_id is not None else self.provider_id,
            "slot": slot if slot is not None else self.slot,
            "price": price if price is not None else self.price,
            "booking_id": booking_id if booking_id is not None else self.booking_id,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._key_payload())

    def matches(self, action: str, provider_id=None, slot=None, price=None, booking_id=None) -> bool:
        candidate = _fingerprint({
            "action": action, "provider_id": provider_id, "slot": slot,
            "price": price, "booking_id": booking_id,
        })
        return candidate == self.fingerprint

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["PendingApproval"]:
        if d is None:
            return None
        return cls(**d)


def create_pending_approval(action: str, proposal_text: str, provider_id=None, provider_name=None,
                             slot=None, price=None, booking_id=None) -> PendingApproval:
    return PendingApproval(
        action=action, provider_id=provider_id, provider_name=provider_name,
        slot=slot, price=price, booking_id=booking_id,
        proposal_text=proposal_text, created_at=time.time(),
    )


CONFIRM_PHRASES = (
    "yes", "y", "yep", "yeah", "confirm", "confirmed", "approve", "approved",
    "do it", "go ahead", "book it", "sounds good", "ok", "okay", "sure",
    "correct", "that's right", "proceed",
)
DENY_PHRASES = (
    "no", "n", "nope", "cancel", "cancel that", "don't", "do not", "stop", "wait",
    "nevermind", "never mind", "hold on", "not yet",
)


def parse_confirmation(user_text: str) -> Optional[bool]:
    """Returns True/False if the raw user text is an unambiguous confirm/deny
    of *something currently pending*, else None (caller must not assume
    consent from None). This is deliberately conservative -- free text that
    isn't a clear yes/no does not grant approval.
    """
    t = user_text.strip().lower().rstrip(".!")
    if not t:
        return None
    if t in CONFIRM_PHRASES or any(t.startswith(p) for p in CONFIRM_PHRASES):
        return True
    if t in DENY_PHRASES or any(t.startswith(p) for p in DENY_PHRASES):
        return False
    return None


def check_approval(pending: Optional[PendingApproval], action: str, provider_id=None,
                    slot=None, price=None, booking_id=None) -> bool:
    """The single gate every book/cancel/reschedule call must pass. Returns
    False (never raises) so callers always have a clean "ask again" path."""
    if pending is None or not pending.confirmed:
        return False
    return pending.matches(action, provider_id=provider_id, slot=slot, price=price, booking_id=booking_id)
