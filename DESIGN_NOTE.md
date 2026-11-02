# Design Note

## Overview

A chat agent for booking/rescheduling/cancelling home-service errands
against a mocked provider API. The core design decision: the "brain" that
turns chat text into structured intent (`agent/nlu/rule_model.py`, a
deterministic regex/keyword parser) is completely separate from the
orchestrator that decides which tools to call and when
(`agent/orchestrator/agent_loop.py`). The model is swappable (an optional
`AnthropicModel` implements the same `Model` protocol) but **every
safety-critical behaviour -- approval, idempotency, injection handling,
limits -- lives in the orchestrator and is identical regardless of which
model is plugged in.** This is what lets "approval enforced by code" and
"offline, reproducible evals" both hold without tension: the deterministic
rule-based model *is* the scripted model the evals run against, not a stand-in
for one.

## State model

`ConversationState` (`agent/core/state.py`) is explicit and persisted to
disk after every turn: `intent`, `stage` (an enum: `COLLECTING ->
SEARCHING -> PRESENTING -> AWAITING_APPROVAL -> EXECUTING ->
CONFIRMED/FAILED/CANCELLED`), the slots collected so far, the active hold,
the pending approval record, and the idempotency key in flight. `Slots` is a
dataclass whose `.update(**kwargs)` only overwrites fields actually present
in the call, so a correction ("actually, mornings only") can never clobber
unrelated slots, and the orchestrator asks only for fields that are still
`None`. Stable preferences (`area`, `budget`, `time_window`, `earliest_time`)
survive a *switch* of intent within a session; request-specific fields
(`service`, `provider_preference`) are deliberately cleared on
`start_new_intent` so a stale `service="plumber"` from an earlier search can
never leak into an unrelated `cancel`/`reschedule` resolution (see BUGS.md
#2 -- this was a real bug, not a hypothetical).

## Approval enforcement

`PendingApproval` (`agent/core/approval.py`) is a record with a
`fingerprint` over `(action, provider_id, slot, price, booking_id)`.
`confirmed` is set **only** by the orchestrator, in response to the user's
literal next message being matched as an unambiguous yes/no against the
*currently pending* proposal -- never inferred from model output. Critically,
the proposal is built from the **search-time price**, and only *after* the
user approves does the orchestrator call `hold_slot` to actually lock it.
If the real quoted price (or availability) differs from what was approved,
`check_approval()` recomputes a fresh fingerprint from the hold's actual
values and the mismatch **voids** the approval automatically -- the user is
shown the new price and must re-confirm before `book()` is ever called. The
same gate runs before `cancel`/`reschedule`, recomputing from
`state.target_booking_id`, not from the approval record itself, so a bug
that mutated `target_booking_id` after approval would still be caught.

## Idempotency and recovery

Every `book`/`cancel`/`reschedule` attempt gets one idempotency key, created
once per hold (or per cancel/reschedule target) and persisted as part of
`ConversationState`. `agent/core/idempotency.py::resolve_book_outcome` calls
`book()` exactly once per key; on a `timeout` it consults
`get_booking_status()` -- the contract's authoritative source -- before
returning anything, and never calls `book()` again for that key. A repeated
"yes" after a booking is already `CONFIRMED` is a state-machine no-op, not a
second tool call. `retry_cancel` retries the scripted transient 503
automatically with the *same* key. Because both the idempotency key and the
hold are part of persisted state, a process restart recovers by reloading
state and re-checking `get_booking_status` with the same key -- the mock
backend (`ProviderAPI`) also persists its own snapshot independently, which
is what makes "restart mid-booking" a meaningful, provable guarantee rather
than something only true inside the eval harness (this independence is also
what bug #3 in BUGS.md broke, and the fix is what makes it real now).

**Annotated excerpt** (full file: `traces/sample_timeout_after_success.jsonl`):
```
tool_call  hold_slot  -> {"status":"ok","hold_id":"hold_e534459406","quoted_price":971}
tool_call  book       -> {"status":"timeout"}                       # no booking_id yet!
tool_call  get_booking_status -> {"status":"booked","booking_id":"bkg_0182d603c1"}
tool_call  list_bookings      -> [...,{"id":"bkg_0182d603c1","status":"booked"}]
assistant_message: "Booked! FreshSpace Cleaners on 2030-03-20T04:46:00 for ₹971."
```
The agent never claims success between the `timeout` and the
`get_booking_status` call -- there is no `assistant_message` in between.

## Memory

`MemoryStore` (`agent/core/memory.py`) persists `Fact{key, value, source,
confirmed}` per user. `source="stated"` facts (the user said it) are usable
immediately; `source="inferred"` facts (e.g. noticing a repeat provider)
start `confirmed=False` and `get_fact(..., only_confirmed=True)` -- the
default used everywhere a fact is auto-applied -- won't return them.
`forget()` writes a tombstone that blocks the same fact from being silently
re-inferred later, not just re-read; only an explicit new user statement can
re-establish it. Only the three fields the assignment names as examples of
stable cross-session preference (area, earliest time, preferred technician)
are ever persisted -- not the full slot set -- which is a deliberate
narrowing, not an oversight.

## Observability

One JSONL file per run (`agent/trace.py`), appended to across the whole
session: `user_message`, `model_call` (model name + structured output),
`state_transition`, `tool_call` (name/args/result/latency), `assistant_message`,
`limit_exceeded`, `error`. Provider review text is sanitised
(`core/sanitize.py`) before it ever reaches the trace -- injected text is
replaced with `[redacted: ...]` and an `injection_flagged: true` marker is
added, so a reviewer can see an injection attempt occurred without the raw
payload being replayed. In practice no review text reaches *any* model at
all in this architecture (reviews are read only for structured fields), which
is a stronger guarantee than sanitization alone.

## What breaks first at 100x, and what I'd change

1. **Flat JSON files as the state/memory/backend store.** Fine for one
   process and a handful of users; at 100x concurrent sessions this is a
   correctness hazard (concurrent writes to the same user's file) long
   before it's a performance one. First change: SQLite with per-user row
   locking, or a real key-value store, behind the exact same
   `ConversationState.save/load` interface.
2. **The synchronous, single-process agent loop.** Each turn fully resolves
   before returning; at scale this serializes all of one user's tool calls
   on one worker. I'd move to a per-session worker/queue model so a slow
   mock (or real) provider call doesn't block that user's next message, and
   so the wall-clock budget in `core/limits.py` reflects real concurrency
   contention, not just one call chain.
3. **The regex-based NLU's precision.** It works because the eval scenarios
   use phrasing I control; at 100x real user variety, recall would degrade
   noticeably on phrasing it doesn't recognize. The architecture already
   isolates this behind the `Model` protocol specifically so swapping in
   `AnthropicModel` (or a hybrid: rules first, LLM fallback on low
   confidence) is a one-line change with zero impact on the approval/
   idempotency/limits code that actually keeps bookings safe.
