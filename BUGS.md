# Three real bugs hit while building this

## 1. Bare "cancel" during an approval prompt wasn't recognized as a denial

**What broke:** `tests/test_rule_model.py::test_bare_cancel_while_awaiting_approval_is_a_denial_not_a_new_cancel_intent`
failed: sending the single word "cancel" while a booking proposal was
awaiting approval produced `confirmation=None, intent=None` instead of
`confirmation=False` -- the agent would have just re-shown the same
proposal forever instead of treating "cancel" as "no, don't do that."

**Why:** `core/approval.py`'s `DENY_PHRASES` tuple had `"cancel that"` but
not the bare word `"cancel"`, so `parse_confirmation("cancel")` returned
`None` (ambiguous) instead of `False`.

**Fix:** Added `"cancel"` to `DENY_PHRASES` in
[agent/core/approval.py](agent/core/approval.py).

**Regression test:**
[tests/test_rule_model.py::test_bare_cancel_while_awaiting_approval_is_a_denial_not_a_new_cancel_intent](tests/test_rule_model.py)
asserts a bare "cancel" while `stage == AWAITING_APPROVAL` yields
`confirmation=False` and does *not* also set a new `cancel` intent.

---

## 2. "Cancel my electrician booking" silently cancelled an unrelated AC-servicing booking

**What broke:** Manual smoke testing (not a unit test -- this one only
showed up when actually talking to the agent) revealed that asking to
cancel an electrician booking when the user's only *upcoming* booking was
for AC servicing returned a proposal to cancel the **AC servicing** booking,
not an "I don't see a matching booking" message.

**Why:** `_resolve_target_booking` in
[agent/orchestrator/agent_loop.py](agent/orchestrator/agent_loop.py)
filtered upcoming bookings by the stated service, but on an **empty**
result it silently fell back to the *unfiltered* list:

```python
narrowed = [b for b in upcoming if b["service"] == slots.service]
if narrowed:
    upcoming = narrowed
# else: keep the unfiltered `upcoming` -- including bookings for a
# completely different service!
```

With only one upcoming booking total (the seeded AC-servicing one), the
unfiltered list had exactly one entry, so the "exactly one candidate -> auto
select it" path fired and proposed cancelling the wrong booking. This is
exactly the class of bug the assignment's "approval enforced by code" and
"booking correctness" requirements exist to catch -- except here the danger
was *upstream* of approval: the user would have been shown a proposal
naming the right slot/price for the *wrong* booking and could easily have
approved it without noticing.

**Fix:** Removed the fallback. If a stated service filters the upcoming
bookings down to zero, that is reported as "no matching booking" --
never silently widened back out.

**Regression test:**
[tests/test_agent_loop.py::test_cancel_with_wrong_service_reports_no_match_instead_of_wrong_booking](tests/test_agent_loop.py)
and
[evals/scenarios.py::cancel_wrong_service_no_match](evals/scenarios.py)
(scenario 20 in `eval_table.md`).

---

## 3. The mock provider "backend" snapshot silently saved to the wrong directory

**What broke:** `tests/test_agent_loop.py::test_restart_mid_booking_recovers_without_duplicate`
failed: after "restarting" (a fresh `Agent` object reloading persisted
state against the same mock backend), the reloaded backend had *no record*
of the booking made in the prior session -- `list_bookings` came back
empty for a booking that had definitely succeeded.

**Why:** `Agent.__init__` stores `self._state_dir` for later use by
`ConversationState.save()`, but the equivalent `provider_dir` parameter was
only used once, to *construct* the initial `ProviderAPI` -- it was never
stored on `self`. `Agent._persist()` called `self.api.save()` with no
arguments, which falls back to `ProviderAPI`'s hard-coded default directory
(`data/provider_backend/`) instead of whatever custom directory the agent
was actually configured with. In production (default paths everywhere) this
is invisible; in tests and in any multi-directory/multi-tenant setup, every
backend snapshot after the first silently went to the wrong file, so a
"restart" that pointed `ProviderAPI.load_or_create` at the *real* configured
directory found nothing and quietly re-seeded a fresh backend --
masking the very recovery guarantee the restart test exists to prove.

**Fix:** Added `self._provider_dir = provider_dir` in `Agent.__init__` and
changed `_persist()` to call `self.api.save(data_dir=self._provider_dir)`.

**Regression test:**
[tests/test_agent_loop.py::test_restart_mid_booking_recovers_without_duplicate](tests/test_agent_loop.py)
and [evals/scenarios.py::restart_mid_booking](evals/scenarios.py) (scenario
15) both construct the agent with a non-default `provider_dir` specifically
so this class of bug can't silently pass by accident of using the default
path everywhere.
