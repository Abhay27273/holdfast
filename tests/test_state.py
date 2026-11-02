import shutil
import tempfile
from pathlib import Path

import pytest

from agent.core.state import ConversationState, Slots, Stage, Intent


@pytest.fixture
def tmp_session_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_slots_update_only_sets_provided_fields():
    slots = Slots(service="plumber", area="koramangala")
    changed = slots.update(time_window="mornings")
    assert changed == ["time_window"]
    assert slots.service == "plumber"
    assert slots.area == "koramangala"
    assert slots.time_window == "mornings"


def test_correction_overwrites_only_the_mentioned_slot():
    slots = Slots(service="plumber", area="koramangala", time_window="evenings")
    slots.update(time_window="mornings")
    assert slots.time_window == "mornings"
    assert slots.service == "plumber"  # untouched
    assert slots.area == "koramangala"  # untouched


def test_missing_reports_only_unfilled_required_fields():
    slots = Slots(service="plumber")
    assert slots.missing(["service", "area", "time_window"]) == ["area", "time_window"]


def test_state_persists_and_reloads_mid_booking(tmp_session_dir):
    state = ConversationState.new("alice")
    state.start_new_intent(Intent.NEW_BOOKING)
    slots = state.get_slots()
    slots.update(service="plumber", area="koramangala")
    state.set_slots(slots)
    state.stage = Stage.HOLDING.value
    state.hold = {"hold_id": "hold_1", "provider_id": "p1", "slot": "s1", "quoted_price": 500}
    state.current_idempotency_key = "idem_abc"
    state.save(data_dir=tmp_session_dir)

    reloaded = ConversationState.load("alice", data_dir=tmp_session_dir)
    assert reloaded is not None
    assert reloaded.stage == Stage.HOLDING.value
    assert reloaded.hold["hold_id"] == "hold_1"
    assert reloaded.current_idempotency_key == "idem_abc"
    assert reloaded.get_slots().service == "plumber"


def test_load_returns_none_when_no_session_exists(tmp_session_dir):
    assert ConversationState.load("nobody", data_dir=tmp_session_dir) is None


def test_start_new_intent_resets_in_flight_fields_but_keeps_slots():
    state = ConversationState.new("alice")
    slots = state.get_slots()
    slots.update(area="koramangala", service="plumber")
    state.set_slots(slots)
    state.hold = {"hold_id": "h1"}
    state.pending_approval = {"action": "book"}
    state.selected_provider = {"id": "p1"}

    state.start_new_intent(Intent.RESCHEDULE)

    assert state.hold is None
    assert state.pending_approval is None
    assert state.selected_provider is None
    assert state.get_slots().area == "koramangala"  # stable preference survives
    assert state.get_slots().service is None  # request-specific, cleared on intent switch
