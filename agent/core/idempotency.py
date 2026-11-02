"""Idempotency-key issuance and the recovery logic around book()/cancel().

Booking correctness (must-have #3) depends on two rules, both implemented
here so they can be unit-tested directly instead of only being described:

1. On a `timeout` from book(), always consult get_booking_status() -- the
   contract's authoritative source of truth -- before retrying or telling
   the user anything. We never call book() a second time for the same
   logical attempt; that's exactly what reusing the idempotency_key across
   retries is for.
2. cancel()'s scripted transient 503 is retried automatically (bounded)
   using the same idempotency_key, so the retry is a true replay rather than
   a new attempt.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from agent.mocks.provider_api import TransientError503


def new_idempotency_key(prefix: str = "op") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def resolve_book_outcome(api, hold_id: str, idempotency_key: str, trace=None, budget=None) -> dict:
    """Call book() exactly once for this idempotency_key. If the result is a
    timeout, resolve it via get_booking_status before returning. Returns a
    normalized dict: {"status": "booked", "booking_id": ...} |
    {"status": "slot_taken"} | {"status": "hold_expired"} | {"status": "failed"}.

    `trace`/`budget` are optional so this stays unit-testable in isolation;
    the orchestrator passes both so every underlying call is still visible
    (and bounded) in the real run.
    """
    if budget is not None:
        budget.note_tool_call()
    t0 = time.monotonic()
    result = api.book(hold_id, idempotency_key)
    if trace is not None:
        trace.tool_call("book", {"hold_id": hold_id, "idempotency_key": idempotency_key},
                         result=result, latency_ms=round((time.monotonic() - t0) * 1000, 2))

    if result["status"] == "timeout":
        if budget is not None:
            budget.note_tool_call()
        t1 = time.monotonic()
        status = api.get_booking_status(idempotency_key)
        if trace is not None:
            trace.tool_call("get_booking_status", {"idempotency_key": idempotency_key},
                             result=status, latency_ms=round((time.monotonic() - t1) * 1000, 2))
        if status["status"] == "booked":
            return {"status": "booked", "booking_id": status["booking_id"]}
        return {"status": "failed"}
    return result


def retry_cancel(api, booking_id: str, idempotency_key: str, max_attempts: int = 3,
                  trace=None, budget=None) -> dict:
    """Retries cancel() on TransientError503 using the SAME idempotency_key,
    up to max_attempts. Re-raises the last error if every attempt fails."""
    last_error: Optional[Exception] = None
    for _ in range(max_attempts):
        if budget is not None:
            budget.note_tool_call()
        t0 = time.monotonic()
        try:
            result = api.cancel(booking_id, idempotency_key)
            if trace is not None:
                trace.tool_call("cancel", {"booking_id": booking_id, "idempotency_key": idempotency_key},
                                 result=result, latency_ms=round((time.monotonic() - t0) * 1000, 2))
            return result
        except TransientError503 as e:
            if trace is not None:
                trace.tool_call("cancel", {"booking_id": booking_id, "idempotency_key": idempotency_key},
                                 error=str(e), latency_ms=round((time.monotonic() - t0) * 1000, 2))
            last_error = e
            continue
    assert last_error is not None
    raise last_error
