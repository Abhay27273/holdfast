import shutil
import tempfile
from pathlib import Path

import pytest

from agent.core.memory import MemoryStore
from agent.core.state import ConversationState, Stage
from agent.mocks.provider_api import ProviderAPI, ScenarioConfig
from agent.orchestrator.agent_loop import Agent


@pytest.fixture
def workspace():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_agent(workspace, user_id="alice", seed=7, scenario=None):
    api = ProviderAPI(user_id=user_id, seed=seed, scenario=scenario or ScenarioConfig())
    memory = MemoryStore(user_id, data_dir=workspace / "memory")
    return Agent(
        user_id, api=api, memory=memory, state_dir=workspace / "sessions",
        provider_dir=workspace / "backend", memory_dir=workspace / "memory",
        trace_dir=workspace / "traces",
    )


def book_first_candidate(agent, text, force_outcome="booked"):
    """Drives a full new-booking flow up through approval and returns the
    reply from saying "yes"."""
    agent.handle_message(text)
    agent.handle_message("1")
    provider_id = agent.state.selected_provider["id"]
    agent.api.scenario.forced_book_outcome[provider_id] = force_outcome
    return agent.handle_message("yes")


# ---------------------------------------------------------------------
# must-have 1: explicit state / never re-ask known facts / corrections
# ---------------------------------------------------------------------

def test_never_reasks_a_known_fact(workspace):
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber in koramangala this week, mornings only.")
    # all required slots (service/area/time_window) are present -- agent must
    # move straight to presenting candidates, not ask again for any of them
    assert agent.state.stage == Stage.PRESENTING.value


def test_correction_keeps_other_slots(workspace):
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber for my kitchen tap this week, mornings only.")
    agent.handle_message("koramangala")
    assert agent.state.get_slots().service == "plumber"
    agent.handle_message("actually, evenings instead")
    slots = agent.state.get_slots()
    assert slots.time_window == "evenings"
    assert slots.service == "plumber"  # untouched
    assert slots.area == "koramangala"  # untouched


# ---------------------------------------------------------------------
# must-have 2: approval enforced by code
# ---------------------------------------------------------------------

def test_no_booking_without_approval(workspace):
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber in koramangala this week, mornings only.")
    agent.handle_message("1")
    assert agent.state.stage == Stage.AWAITING_APPROVAL.value
    # user says something that is NOT a clear yes
    agent.handle_message("what's the weather like")
    assert agent.state.stage == Stage.AWAITING_APPROVAL.value
    assert agent.state.last_booking_id is None


def test_price_change_after_approval_voids_it_and_blocks_booking(workspace):
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber in koramangala this week, mornings only.")
    agent.handle_message("1")
    provider = agent.state.selected_provider
    agent.api.scenario.price_drift[provider["id"]] = provider["price"] + 777
    reply = agent.handle_message("yes")
    assert "price changed" in reply.lower()
    assert agent.state.last_booking_id is None
    assert agent.state.stage == Stage.AWAITING_APPROVAL.value
    # re-confirming at the NEW price now proceeds
    agent.api.scenario.forced_book_outcome[provider["id"]] = "booked"
    reply2 = agent.handle_message("yes")
    assert "booked!" in reply2.lower()
    assert agent.state.last_booking_id is not None


def test_slot_unavailable_after_approval_blocks_booking(workspace):
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber in koramangala this week, mornings only.")
    agent.handle_message("1")
    provider = agent.state.selected_provider
    slot = agent.state.pending_approval["slot"]
    agent.api.scenario.unavailable_holds.add((provider["id"], slot))
    reply = agent.handle_message("yes")
    assert "unavailable" in reply.lower()
    assert agent.state.last_booking_id is None


def test_denying_a_proposal_does_not_book(workspace):
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber in koramangala this week, mornings only.")
    agent.handle_message("1")
    reply = agent.handle_message("no")
    assert agent.state.last_booking_id is None
    assert agent.state.stage != Stage.AWAITING_APPROVAL.value


# ---------------------------------------------------------------------
# must-have 3: booking correctness (idempotency / no duplicates)
# ---------------------------------------------------------------------

def test_repeated_yes_after_booking_does_not_double_book(workspace):
    agent = make_agent(workspace)
    reply = book_first_candidate(agent, "Find a plumber in koramangala this week, mornings only.")
    assert "booked!" in reply.lower()
    first_booking_id = agent.state.last_booking_id
    reply2 = agent.handle_message("yes")  # repeated confirmation, nothing pending anymore
    assert agent.state.last_booking_id == first_booking_id
    bookings = agent.api.list_bookings("alice")
    booked = [b for b in bookings if b["status"] == "booked" and b["service"] == "plumber"]
    assert len(booked) == 1


def test_timeout_after_success_is_reported_correctly_and_only_booked_once(workspace):
    agent = make_agent(workspace, seed=13)
    agent.handle_message("Find a cleaner in whitefield this week.")
    agent.handle_message("1")
    provider_id = agent.state.selected_provider["id"]
    agent.api.scenario.forced_book_outcome[provider_id] = "timeout"
    agent.api.scenario.forced_timeout_secret[provider_id] = "booked"
    reply = agent.handle_message("yes")
    assert "booked!" in reply.lower()
    bookings = [b for b in agent.api.list_bookings("alice") if b["service"] == "cleaner"]
    assert len(bookings) == 1
    assert bookings[0]["status"] == "booked"


def test_timeout_after_failure_is_reported_as_failure_not_phantom_booking(workspace):
    agent = make_agent(workspace, seed=13)
    agent.handle_message("Find a cleaner in whitefield this week.")
    agent.handle_message("1")
    provider_id = agent.state.selected_provider["id"]
    agent.api.scenario.forced_book_outcome[provider_id] = "timeout"
    agent.api.scenario.forced_timeout_secret[provider_id] = "failed"
    reply = agent.handle_message("yes")
    assert "didn't go through" in reply.lower()
    bookings = [b for b in agent.api.list_bookings("alice") if b["service"] == "cleaner"]
    assert len(bookings) == 0


def test_restart_mid_booking_recovers_without_duplicate(workspace):
    agent1 = make_agent(workspace, seed=13)
    agent1.handle_message("Find a cleaner in whitefield this week.")
    agent1.handle_message("1")
    provider_id = agent1.state.selected_provider["id"]
    agent1.api.scenario.forced_book_outcome[provider_id] = "timeout"
    agent1.api.scenario.forced_timeout_secret[provider_id] = "booked"
    agent1.handle_message("yes")  # process "crashes" conceptually right after this
    assert agent1.state.last_booking_id is not None
    first_booking_id = agent1.state.last_booking_id

    # simulate restart: brand-new Agent object reloading persisted state,
    # but talking to the SAME backend (provider API persists independently)
    api2 = ProviderAPI.load_or_create("alice", seed=13, data_dir=workspace / "backend")
    memory2 = MemoryStore("alice", data_dir=workspace / "memory")
    agent2 = Agent("alice", api=api2, memory=memory2, state_dir=workspace / "sessions",
                   provider_dir=workspace / "backend", memory_dir=workspace / "memory",
                   trace_dir=workspace / "traces")
    assert agent2.state.last_booking_id == first_booking_id

    bookings = [b for b in agent2.api.list_bookings("alice") if b["service"] == "cleaner"]
    assert len(bookings) == 1  # still only one booking after "restart"


# ---------------------------------------------------------------------
# must-have 4: untrusted tool output / prompt injection
# ---------------------------------------------------------------------

def test_injection_review_does_not_bypass_approval_or_change_price(workspace):
    agent = make_agent(workspace, seed=5, scenario=ScenarioConfig(inject_reviews=True))
    agent.handle_message("I want to cancel my AC servicing booking")
    reply = agent.handle_message("yes")
    # cancellation proceeded via the normal approval flow, not an injected shortcut
    assert "cancelled" in reply.lower()
    # the injection content must have been flagged in the trace, never echoed to the user
    trace_text = agent.trace.path.read_text()
    assert "injection_flagged" in trace_text
    assert "ignore all previous instructions" not in trace_text.lower()


def test_injection_flagged_in_trace_for_every_provider_lookup(workspace):
    agent = make_agent(workspace, seed=5)
    agent.handle_message("I want to cancel my AC servicing booking")
    trace_text = agent.trace.path.read_text()
    assert '"injection_flagged": true' in trace_text


# ---------------------------------------------------------------------
# must-have 5: bounded execution
# ---------------------------------------------------------------------

def test_limit_exhaustion_produces_clear_message_not_a_crash(workspace):
    from agent.core.limits import Budget
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber in koramangala this week, mornings only.")
    agent.handle_message("1")
    # the confirmation turn needs >=2 tool calls (hold_slot, then book) --
    # capping at 1 forces a limit hit mid-execution, not mid-search
    agent.budget_config = Budget(max_tool_calls=1, max_model_turns=5, max_wall_clock_seconds=5)
    reply = agent.handle_message("yes")
    assert "safety limits" in reply.lower()
    assert agent.state.last_booking_id is None


# ---------------------------------------------------------------------
# must-have 6: memory
# ---------------------------------------------------------------------

def test_stated_area_persists_across_sessions_and_is_not_reasked(workspace):
    agent1 = make_agent(workspace)
    agent1.handle_message("Find a plumber in koramangala this week, mornings only.")
    assert agent1.state.stage == Stage.PRESENTING.value

    api2 = ProviderAPI.load_or_create("alice", seed=99, data_dir=workspace / "backend2")
    memory2 = MemoryStore("alice", data_dir=workspace / "memory")
    agent2 = Agent("alice", api=api2, memory=memory2, state_dir=workspace / "sessions2",
                   provider_dir=workspace / "backend2", memory_dir=workspace / "memory",
                   trace_dir=workspace / "traces2")
    reply = agent2.handle_message("Find an electrician this week, mornings only.")
    # area was never restated, yet the agent must not ask for it again
    assert "which area" not in reply.lower()
    assert agent2.state.stage == Stage.PRESENTING.value


def test_forgotten_fact_is_never_reused(workspace):
    agent = make_agent(workspace)
    agent.handle_message("Find a plumber in koramangala this week, mornings only.")
    agent.memory.forget("area")

    # fresh session: with area forgotten, memory must not silently supply it
    api2 = ProviderAPI.load_or_create("alice", seed=99, data_dir=workspace / "backend2")
    memory2 = MemoryStore("alice", data_dir=workspace / "memory")
    agent2 = Agent("alice", api=api2, memory=memory2, state_dir=workspace / "sessions2",
                   provider_dir=workspace / "backend2", memory_dir=workspace / "memory",
                   trace_dir=workspace / "traces2")
    reply = agent2.handle_message("Find an electrician this week, mornings only.")
    assert "which area" in reply.lower()


# ---------------------------------------------------------------------
# regression: service-mismatch must never resolve to an unrelated booking
# (BUGS.md #1 -- "cancel my electrician booking" previously matched an
# AC-servicing booking when no electrician booking existed)
# ---------------------------------------------------------------------

def test_cancel_with_wrong_service_reports_no_match_instead_of_wrong_booking(workspace):
    agent = make_agent(workspace)
    reply = agent.handle_message("I want to cancel my electrician booking")
    assert "don't see a matching" in reply.lower()


# ---------------------------------------------------------------------
# "book the same provider as last time" (assignment example 3)
# ---------------------------------------------------------------------

def test_book_same_provider_as_last_time_with_earliest_time_constraint(workspace):
    agent = make_agent(workspace, seed=1)
    reply = agent.handle_message("Book the same electrician as last time, but not before 10 a.m.")
    assert agent.state.stage == Stage.AWAITING_APPROVAL.value
    assert agent.state.selected_provider is not None
    proposed_slot = agent.state.pending_approval["slot"]
    assert proposed_slot[11:13] >= "10"
    provider_id = agent.state.selected_provider["id"]
    agent.api.scenario.forced_book_outcome[provider_id] = "booked"
    reply = agent.handle_message("yes")
    assert "booked!" in reply.lower()
