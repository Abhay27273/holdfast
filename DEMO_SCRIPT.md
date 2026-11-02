# Demo recording script (3-5 min)

Goal per the brief: one full booking end-to-end, plus at least one failure
handled live (prompt injection or timeout-after-success), with the chat and
its trace visible side by side.

## Setup (before recording)

Two windows side by side:
- **Left:** terminal running `python -m agent.cli --user-id demo --reset`
- **Right:** the trace file tailing live: `tail -f traces/demo_*.jsonl` (or
  just re-open/refresh the newest file in `traces/` after each message --
  it's one JSONL file per run, appended to every turn)

## Part 1 -- full booking (≈90s)

Type, pausing briefly after each reply so the trace catches up on screen:

```
Find a plumber for my kitchen tap this week, mornings only, under ₹1500.
koramangala
1
yes
```

Narrate while the trace updates: point out the `tool_call` entries for
`search_providers` then `hold_slot` then `book`, and the `state_transition`
entry showing `approval: confirmed` appearing *before* `hold_slot` is ever
called -- i.e. nothing gets booked without that record existing first.

## Part 2 -- prompt injection handled live (≈60s)

```
I want to cancel my AC servicing booking
```

Before confirming, open the trace and find the `get_provider` tool_call
entry: point out `"injection_flagged": true` and the `reviews_sanitized`
field showing `[redacted: instruction-like content removed]` in place of the
raw review text. Say explicitly: *the review tried to tell the agent to skip
approval / book without confirmation -- it still asked.* Then:

```
yes
```

Point out the cancellation still went through the normal approval flow
(nothing was skipped), and that `cancel` actually appears **twice** in the
trace -- once failing with a transient error, once succeeding on retry --
invisibly to you as the user.

## Part 3 (optional, if time) -- timeout-after-success

This needs a forced scenario override, which the CLI doesn't expose
directly -- easiest to show by instead opening `traces/sample_timeout_after_success.jsonl`
(committed in the repo) and walking through it live:
point at `book -> {"status":"timeout"}` immediately followed by
`get_booking_status -> {"status":"booked",...}`, and note there is no
`assistant_message` claiming success in between those two lines -- the
agent only tells the user once it's verified.

## Closing line

"Every one of these guarantees -- approval, idempotency, injection handling
-- is enforced in orchestrator code and covered by an automated test, not
just a system prompt." (Points at `eval_table.md` / `pytest` output.)
