"""The orchestrator: the only code allowed to call provider tools or set
approval. The Model (rule-based or optional LLM) only ever returns an
Understanding -- structured data -- which this module interprets through an
explicit, deterministic state machine. That split is what makes every
must-have enforceable in code rather than resting on model behaviour:

- Approval is created here from a concrete proposal, confirmed here by
  matching the user's literal next message, and re-validated here
  (core/approval.check_approval) against the *actual* values about to be
  executed before every book/cancel/reschedule call.
- Provider review text is only ever read for display fields (price, rating,
  free_slots); get_provider's free text is sanitised before it even reaches
  the trace, and is never parsed for instructions.
- Every exit path funnels through handle_message's try/except so a clear
  message is always returned, even on a limit hit or an unexpected error.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Optional

from agent.core.approval import PendingApproval, check_approval, create_pending_approval
from agent.core.idempotency import new_idempotency_key, resolve_book_outcome, retry_cancel
from agent.core.limits import Budget, BudgetTracker, LimitExceeded
from agent.core.memory import MemoryStore
from agent.core.sanitize import sanitize_reviews
from agent.core.state import REQUIRED_SLOTS_FOR_BOOKING, ConversationState, Intent, Stage
from agent.mocks.provider_api import ProviderAPI, ProviderAPIError, TransientError503
from agent.nlu.rule_model import RuleBasedModel
from agent.trace import TraceWriter

from . import responses
from .matching import best_matching_slot

PERSISTABLE_SLOT_KEYS = {"area", "earliest_time", "budget"}


class Agent:
    def __init__(self, user_id: str, model=None, api: Optional[ProviderAPI] = None,
                 memory: Optional[MemoryStore] = None, budget: Optional[Budget] = None,
                 trace: Optional[TraceWriter] = None, state_dir: Optional[Path] = None,
                 provider_dir: Optional[Path] = None, memory_dir: Optional[Path] = None,
                 trace_dir: Optional[Path] = None, seed: int = 42):
        self.user_id = user_id
        self.model = model or RuleBasedModel()
        self.api = api or ProviderAPI.load_or_create(user_id, seed=seed, data_dir=provider_dir)
        self.memory = memory or MemoryStore(user_id, data_dir=memory_dir)
        self.budget_config = budget or Budget()
        self._state_dir = state_dir
        self._provider_dir = provider_dir

        self.state = ConversationState.load(user_id, data_dir=state_dir) or ConversationState.new(user_id)
        if self.state.stage in (Stage.CONFIRMED.value, Stage.FAILED.value, Stage.CANCELLED.value):
            self.state.stage = Stage.IDLE.value
            self.state.intent = Intent.UNKNOWN.value

        self.trace = trace or TraceWriter(run_id=f"{user_id}_{self.state.session_id}", trace_dir=trace_dir)

    # ------------------------------------------------------------------
    # top-level entry point
    # ------------------------------------------------------------------
    def handle_message(self, text: str) -> str:
        self.trace.log("user_message", text=text)
        try:
            reply = self._handle_message_inner(text)
        except LimitExceeded as e:
            reply = responses.limit_exceeded(e.kind)
            self.trace.log("limit_exceeded", kind=e.kind, detail=e.detail)
        except Exception as e:  # last-resort bounded exit -- must-have #5
            reply = responses.unexpected_error()
            self.trace.log("error", error_type=type(e).__name__, error=str(e))
        self._persist()
        self.trace.log("assistant_message", text=reply)
        return reply

    def _persist(self) -> None:
        self.state.save(data_dir=self._state_dir)
        self.api.save(data_dir=self._provider_dir)

    # ------------------------------------------------------------------
    # per-turn processing
    # ------------------------------------------------------------------
    def _handle_message_inner(self, text: str) -> str:
        budget = BudgetTracker(self.budget_config)
        budget.note_model_turn()
        u = self.model.understand(text, self.state)
        self.trace.model_call(self.model.name, text, dataclasses.asdict(u))

        if u.memory_show_requested:
            return responses.memory_summary(self.memory.list_facts())
        if u.memory_confirm_key:
            self._apply_to_memory_keys(u.memory_confirm_key, self.memory.confirm)
            return "Got it, I'll remember that."
        if u.memory_forget_key:
            self._apply_to_memory_keys(u.memory_forget_key, self.memory.forget)
            return responses.memory_forgotten(u.memory_forget_key)

        # Intent switches are resolved BEFORE this turn's slot updates are
        # applied: start_new_intent() clears request-specific slots
        # (service, provider_preference) so a stale value from an earlier,
        # unrelated request can never leak into a new one. This turn's own
        # slot extraction (applied right after) still sets them correctly
        # when the new message actually mentions them.
        awaiting_approval = self.state.stage == Stage.AWAITING_APPROVAL.value
        consumed_confirmation = awaiting_approval and u.confirmation is not None
        started_new_booking = False

        if u.intent and not consumed_confirmation:
            self.state.start_new_intent(Intent(u.intent))
            self.trace.log("state_transition", new_intent=u.intent)
            started_new_booking = Intent(u.intent) == Intent.NEW_BOOKING

            if Intent(u.intent) == Intent.VIEW_BOOKINGS:
                bookings = self._fetch_and_trace_bookings(budget)
                self.state.stage = Stage.IDLE.value
                self.state.intent = Intent.UNKNOWN.value
                return responses.bookings_list(bookings)

        slots = self.state.get_slots()
        changed = slots.update(**u.slot_updates)
        if started_new_booking:
            self._apply_memory_defaults(slots)
        self.state.set_slots(slots)
        if changed:
            self.trace.log("state_transition", changed_slots={k: getattr(slots, k) for k in changed})
            for key in changed:
                if key in PERSISTABLE_SLOT_KEYS:
                    self.memory.set_fact(key, getattr(slots, key), source="stated")

        # A correction while a proposal is pending invalidates it -- better
        # to rebuild the proposal than silently book/cancel/reschedule
        # something the user just changed their mind about.
        if self.state.stage == Stage.AWAITING_APPROVAL.value and changed and u.confirmation is None:
            self.state.pending_approval = None
            self.state.stage = Stage.COLLECTING.value
            self.trace.log("state_transition", approval="voided", reason="user_changed_slots", changed=changed)

        if consumed_confirmation:
            if u.confirmation:
                pending = PendingApproval.from_dict(self.state.pending_approval)
                pending.confirmed = True
                pending.confirmed_at = time.time()
                self.state.pending_approval = pending.to_dict()
                self.state.stage = Stage.EXECUTING.value
                self.trace.log("state_transition", approval="confirmed", action=pending.action)
            else:
                self.state.pending_approval = None
                self.state.stage = Stage.COLLECTING.value
                self.trace.log("state_transition", approval="denied")
                return responses.denied()

        if u.selection_index is not None and self.state.stage == Stage.PRESENTING.value:
            self._select_candidate(u.selection_index)

        while True:
            reply = self._advance(budget)
            if reply is not None:
                return reply

    def _apply_to_memory_keys(self, key: str, action_fn) -> None:
        if key == "__all__":
            for k in list(PERSISTABLE_SLOT_KEYS) + ["preferred_technician"]:
                action_fn(k)
        else:
            action_fn(key)

    def _apply_memory_defaults(self, slots) -> None:
        for key in PERSISTABLE_SLOT_KEYS:
            if getattr(slots, key, None) is None:
                value = self.memory.get_fact(key, only_confirmed=True)
                if value is not None:
                    setattr(slots, key, value)
                    self.trace.log("state_transition", applied_from_memory={key: value})

    # ------------------------------------------------------------------
    # the deterministic stage machine
    # ------------------------------------------------------------------
    def _advance(self, budget: BudgetTracker) -> Optional[str]:
        stage = self.state.stage

        if stage == Stage.IDLE.value:
            return responses.greeting()

        if stage == Stage.COLLECTING.value:
            return self._advance_collecting(budget)

        if stage == Stage.SEARCHING.value:
            return self._advance_searching(budget)

        if stage == Stage.PRESENTING.value:
            if self.state.candidates_kind == "providers":
                return responses.present_candidates(self.state.candidates)
            return responses.present_bookings_for_selection(self.state.candidates)

        if stage == Stage.HOLDING.value:
            return self._advance_proposing_booking()

        if stage == Stage.AWAITING_APPROVAL.value:
            pending = PendingApproval.from_dict(self.state.pending_approval)
            return pending.proposal_text if pending else responses.unexpected_error()

        if stage == Stage.EXECUTING.value:
            return self._advance_executing(budget)

        self.state.stage = Stage.IDLE.value
        self.state.intent = Intent.UNKNOWN.value
        return responses.greeting()

    def _advance_collecting(self, budget: BudgetTracker) -> Optional[str]:
        intent = self.state.intent
        slots = self.state.get_slots()

        if intent == Intent.NEW_BOOKING.value:
            if slots.provider_preference == "SAME_AS_LAST":
                # "Book the same electrician as last time, not before 10am":
                # resolve via booking history instead of a fresh search --
                # area/time_window aren't needed when the provider is implied.
                return self._advance_new_booking_same_as_last(budget, slots)
            missing = slots.missing(REQUIRED_SLOTS_FOR_BOOKING)
            if missing:
                return responses.ask_for_slot(missing[0])
            self.state.stage = Stage.SEARCHING.value
            return None

        if intent in (Intent.CANCEL.value, Intent.RESCHEDULE.value):
            if self.state.target_booking_id is None:
                msg = self._resolve_target_booking(budget)
                if msg is not None:
                    return msg

            target = self._find_target_booking_dict(budget)
            provider = self._fetch_and_trace_provider(budget, target["provider_id"])

            if intent == Intent.CANCEL.value:
                proposal_text = responses.propose_cancel_approval(provider["name"], target["slot"], target["id"])
                pending = create_pending_approval(
                    "cancel", proposal_text, provider_id=target["provider_id"], provider_name=provider["name"],
                    slot=target["slot"], booking_id=target["id"],
                )
            else:
                if slots.time_window is None:
                    return responses.ask_for_slot("time_window")
                new_slot = best_matching_slot(provider["free_slots"], slots.time_window, slots.earliest_time)
                if new_slot is None:
                    return responses.reschedule_no_matching_slot()
                proposal_text = responses.propose_reschedule_approval(provider["name"], target["slot"], new_slot)
                pending = create_pending_approval(
                    "reschedule", proposal_text, provider_id=target["provider_id"], provider_name=provider["name"],
                    slot=new_slot, booking_id=target["id"],
                )

            self.state.pending_approval = pending.to_dict()
            self.state.stage = Stage.AWAITING_APPROVAL.value
            return proposal_text

        return responses.greeting()

    def _advance_new_booking_same_as_last(self, budget: BudgetTracker, slots) -> Optional[str]:
        bookings = self._fetch_and_trace_bookings(budget)
        pool = [b for b in bookings if not slots.service or b["service"] == slots.service]
        if not pool:
            return responses.no_matching_booking()
        reference = pool[-1]
        provider = self._fetch_and_trace_provider(budget, reference["provider_id"])
        chosen_slot = best_matching_slot(provider["free_slots"], slots.time_window, slots.earliest_time)
        if chosen_slot is None:
            return responses.no_candidates_found()
        self.state.selected_provider = provider
        self.state.stage = Stage.HOLDING.value
        return None

    def _advance_searching(self, budget: BudgetTracker) -> str:
        slots = self.state.get_slots()
        date_range = (slots.time_window or "", slots.time_window or "")
        budget.note_tool_call()
        t0 = time.monotonic()
        results = self.api.search_providers(slots.service, slots.area, date_range)
        self.trace.tool_call(
            "search_providers", {"service": slots.service, "area": slots.area, "date_range": date_range},
            result=results, latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
        filtered = [c for c in results if self._candidate_matches(c, slots)]
        if not filtered:
            self.state.stage = Stage.COLLECTING.value
            return responses.no_candidates_found()
        self.state.candidates = filtered
        self.state.candidates_kind = "providers"
        self.state.stage = Stage.PRESENTING.value
        return responses.present_candidates(filtered)

    def _candidate_matches(self, candidate: dict, slots) -> bool:
        if slots.budget is not None and candidate["price"] > slots.budget:
            return False
        if (slots.time_window or slots.earliest_time) and best_matching_slot(
            candidate["free_slots"], slots.time_window, slots.earliest_time
        ) is None:
            return False
        return True

    def _select_candidate(self, index: int) -> None:
        items = self.state.candidates
        if not (0 <= index < len(items)):
            return
        selected = items[index]
        if self.state.candidates_kind == "providers":
            self.state.selected_provider = selected
            self.state.stage = Stage.HOLDING.value
        else:
            self.state.target_booking_id = selected["id"]
            self.state.stage = Stage.COLLECTING.value

    def _advance_proposing_booking(self) -> str:
        slots = self.state.get_slots()
        provider = self.state.selected_provider
        chosen_slot = best_matching_slot(provider["free_slots"], slots.time_window, slots.earliest_time)
        if chosen_slot is None:
            chosen_slot = provider["free_slots"][0]
        proposal_text = responses.propose_booking_approval(provider["name"], chosen_slot, provider["price"])
        pending = create_pending_approval(
            "book", proposal_text, provider_id=provider["id"], provider_name=provider["name"],
            slot=chosen_slot, price=provider["price"],
        )
        self.state.pending_approval = pending.to_dict()
        self.state.stage = Stage.AWAITING_APPROVAL.value
        return proposal_text

    def _resolve_target_booking(self, budget: BudgetTracker) -> Optional[str]:
        bookings = self._fetch_and_trace_bookings(budget)
        slots = self.state.get_slots()

        if slots.provider_preference == "SAME_AS_LAST":
            pool = [b for b in bookings if not slots.service or b["service"] == slots.service]
            if pool:
                self.state.target_booking_id = pool[-1]["id"]
                return None
            return responses.no_matching_booking()

        upcoming = [b for b in bookings if b["status"] == "booked"]
        if slots.service:
            # Strict filter: if the user named a service, a booking for a
            # DIFFERENT service is never an acceptable match -- silently
            # falling back to "whatever's upcoming" here previously caused
            # "cancel my electrician booking" to match an AC-servicing
            # booking when no electrician booking existed (see BUGS.md).
            upcoming = [b for b in upcoming if b["service"] == slots.service]

        if len(upcoming) == 0:
            return responses.no_matching_booking()
        if len(upcoming) == 1:
            self.state.target_booking_id = upcoming[0]["id"]
            return None

        self.state.candidates = upcoming
        self.state.candidates_kind = "bookings"
        self.state.stage = Stage.PRESENTING.value
        return responses.present_bookings_for_selection(upcoming)

    def _advance_executing(self, budget: BudgetTracker) -> str:
        pending = PendingApproval.from_dict(self.state.pending_approval)
        if pending is None or not pending.confirmed:
            self.state.stage = Stage.COLLECTING.value
            return responses.greeting()

        if pending.action == "book":
            return self._execute_book(budget, pending)
        if pending.action == "cancel":
            return self._execute_cancel(budget, pending)
        if pending.action == "reschedule":
            return self._execute_reschedule(budget, pending)

        self.state.stage = Stage.FAILED.value
        return responses.unexpected_error()

    def _execute_book(self, budget: BudgetTracker, pending: PendingApproval) -> str:
        provider = self.state.selected_provider

        if self.state.hold is None:
            budget.note_tool_call()
            t0 = time.monotonic()
            result = self.api.hold_slot(provider["id"], pending.slot)
            self.trace.tool_call("hold_slot", {"provider_id": provider["id"], "slot": pending.slot},
                                  result=result, latency_ms=round((time.monotonic() - t0) * 1000, 2))
            if result["status"] == "slot_unavailable":
                self.state.pending_approval = None
                self.state.stage = Stage.PRESENTING.value
                self.trace.log("state_transition", approval="voided", reason="slot_unavailable")
                return responses.approval_voided_slot_unavailable()
            self.state.hold = {
                "hold_id": result["hold_id"], "provider_id": provider["id"],
                "slot": pending.slot, "quoted_price": result["quoted_price"],
                "expires_at": result["expires_at"],
            }
            self.state.current_idempotency_key = new_idempotency_key("book")

        hold = self.state.hold
        if not check_approval(pending, "book", provider_id=hold["provider_id"], slot=hold["slot"],
                               price=hold["quoted_price"]):
            new_pending = create_pending_approval(
                "book", responses.propose_booking_approval(provider["name"], hold["slot"], hold["quoted_price"]),
                provider_id=hold["provider_id"], provider_name=provider["name"],
                slot=hold["slot"], price=hold["quoted_price"],
            )
            self.state.pending_approval = new_pending.to_dict()
            self.state.stage = Stage.AWAITING_APPROVAL.value
            self.trace.log("state_transition", approval="voided", reason="price_drift",
                            old_price=pending.price, new_price=hold["quoted_price"])
            return responses.approval_voided_price_changed(pending.price, hold["quoted_price"])

        outcome = resolve_book_outcome(
            self.api, hold["hold_id"], self.state.current_idempotency_key, trace=self.trace, budget=budget,
        )

        if outcome["status"] == "booked":
            booking_id = outcome["booking_id"]
            self.state.last_booking_id = booking_id
            self.state.stage = Stage.CONFIRMED.value
            self._maybe_infer_preferred_technician(provider["id"], booking_id, budget)
            return responses.booking_confirmed(provider["name"], hold["slot"], hold["quoted_price"], booking_id)

        self.state.hold = None
        self.state.pending_approval = None
        if outcome["status"] == "slot_taken":
            self.state.stage = Stage.PRESENTING.value
            return responses.slot_taken()
        if outcome["status"] == "hold_expired":
            self.state.stage = Stage.PRESENTING.value
            return responses.hold_expired()
        self.state.stage = Stage.FAILED.value
        return responses.booking_failed()

    def _execute_cancel(self, budget: BudgetTracker, pending: PendingApproval) -> str:
        if self.state.current_idempotency_key is None:
            self.state.current_idempotency_key = new_idempotency_key("cancel")
        if not check_approval(pending, "cancel", booking_id=self.state.target_booking_id,
                               slot=pending.slot, provider_id=pending.provider_id):
            self.state.pending_approval = None
            self.state.stage = Stage.COLLECTING.value
            return responses.denied()
        try:
            retry_cancel(self.api, self.state.target_booking_id, self.state.current_idempotency_key,
                         trace=self.trace, budget=budget)
        except TransientError503:
            self.state.stage = Stage.FAILED.value
            return responses.unexpected_error()
        self.state.stage = Stage.CANCELLED.value
        return responses.cancelled(self.state.target_booking_id)

    def _execute_reschedule(self, budget: BudgetTracker, pending: PendingApproval) -> str:
        if self.state.current_idempotency_key is None:
            self.state.current_idempotency_key = new_idempotency_key("reschedule")
        if not check_approval(pending, "reschedule", booking_id=self.state.target_booking_id,
                               slot=pending.slot, provider_id=pending.provider_id):
            self.state.pending_approval = None
            self.state.stage = Stage.COLLECTING.value
            return responses.denied()

        budget.note_tool_call()
        t0 = time.monotonic()
        args = {"booking_id": self.state.target_booking_id, "new_slot": pending.slot,
                "idempotency_key": self.state.current_idempotency_key}
        result = self.api.reschedule(self.state.target_booking_id, pending.slot, self.state.current_idempotency_key)
        self.trace.tool_call("reschedule", args, result=result, latency_ms=round((time.monotonic() - t0) * 1000, 2))

        if result["status"] == "slot_conflict":
            self.state.pending_approval = None
            self.state.current_idempotency_key = None
            slots = self.state.get_slots()
            slots.clear("time_window")
            self.state.set_slots(slots)
            self.state.stage = Stage.COLLECTING.value
            return responses.reschedule_slot_conflict()

        self.state.stage = Stage.CONFIRMED.value
        return responses.rescheduled(pending.provider_name, pending.slot)

    def _maybe_infer_preferred_technician(self, provider_id: str, booking_id: str, budget: BudgetTracker) -> None:
        bookings = self._fetch_and_trace_bookings(budget)
        prior = [b for b in bookings if b["provider_id"] == provider_id and b["id"] != booking_id]
        if prior:
            self.memory.set_fact("preferred_technician", provider_id, source="inferred")

    # ------------------------------------------------------------------
    # traced tool wrappers
    # ------------------------------------------------------------------
    def _fetch_and_trace_bookings(self, budget: BudgetTracker) -> list:
        budget.note_tool_call()
        t0 = time.monotonic()
        bookings = self.api.list_bookings(self.user_id)
        self.trace.tool_call("list_bookings", {"user_id": self.user_id}, result=bookings,
                              latency_ms=round((time.monotonic() - t0) * 1000, 2))
        return bookings

    def _fetch_and_trace_provider(self, budget: BudgetTracker, provider_id: str) -> dict:
        budget.note_tool_call()
        t0 = time.monotonic()
        provider = self.api.get_provider(provider_id)
        sanitized = sanitize_reviews(provider.get("reviews", []))
        trace_result = {k: v for k, v in provider.items() if k != "reviews"}
        trace_result["reviews_sanitized"] = [r.cleaned for r in sanitized]
        if any(r.flagged for r in sanitized):
            trace_result["injection_flagged"] = True
        self.trace.tool_call("get_provider", {"provider_id": provider_id}, result=trace_result,
                              latency_ms=round((time.monotonic() - t0) * 1000, 2))
        return provider

    def _find_target_booking_dict(self, budget: BudgetTracker) -> dict:
        bookings = self._fetch_and_trace_bookings(budget)
        for b in bookings:
            if b["id"] == self.state.target_booking_id:
                return b
        raise ProviderAPIError(f"target booking {self.state.target_booking_id} not found")
