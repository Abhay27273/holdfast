from agent.core.state import ConversationState, Stage, Intent
from agent.nlu.rule_model import RuleBasedModel

model = RuleBasedModel()


def fresh_state(stage=Stage.IDLE, intent=Intent.UNKNOWN):
    s = ConversationState.new("u1")
    s.stage = stage.value
    s.intent = intent.value
    return s


def test_example_request_plumber():
    u = model.understand(
        "Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.",
        fresh_state(),
    )
    assert u.intent == Intent.NEW_BOOKING.value
    assert u.slot_updates["service"] == "plumber"
    assert u.slot_updates["budget"] == 1500
    assert u.slot_updates["time_window"] == "this_week mornings"


def test_example_request_reschedule():
    u = model.understand("Move my AC servicing to Saturday.", fresh_state())
    assert u.intent == Intent.RESCHEDULE.value
    assert u.slot_updates["service"] == "ac_technician"
    assert u.slot_updates["time_window"] == "saturday"


def test_example_request_same_provider_with_time_constraint():
    u = model.understand("Book the same electrician as last time, but not before 10 a.m.", fresh_state())
    assert u.slot_updates["provider_preference"] == "SAME_AS_LAST"
    assert u.slot_updates["earliest_time"] == "10:00"


def test_correction_only_updates_mentioned_slot():
    u = model.understand("Actually, mornings only.", fresh_state(intent=Intent.NEW_BOOKING))
    assert u.slot_updates == {"time_window": "mornings"}
    assert "service" not in u.slot_updates


def test_confirmation_yes_and_no():
    assert model.understand("yes", fresh_state()).confirmation is True
    assert model.understand("no", fresh_state()).confirmation is False
    assert model.understand("sounds fine to me, go for it", fresh_state()).confirmation is None


def test_bare_cancel_while_awaiting_approval_is_a_denial_not_a_new_cancel_intent():
    state = fresh_state(stage=Stage.AWAITING_APPROVAL, intent=Intent.NEW_BOOKING)
    u = model.understand("cancel", state)
    assert u.confirmation is False
    assert u.intent is None


def test_cancel_booking_intent_when_not_awaiting_approval():
    u = model.understand("I want to cancel my AC servicing booking", fresh_state())
    assert u.intent == Intent.CANCEL.value


def test_view_bookings_intent():
    u = model.understand("show my bookings", fresh_state())
    assert u.intent == Intent.VIEW_BOOKINGS.value


def test_memory_forget_and_show():
    assert model.understand("forget my area", fresh_state()).memory_forget_key == "area"
    assert model.understand("what do you remember about me", fresh_state()).memory_show_requested is True


def test_selection_index_only_parsed_while_presenting():
    presenting = fresh_state(stage=Stage.PRESENTING)
    assert model.understand("option 2", presenting).selection_index == 1
    assert model.understand("the second one", presenting).selection_index == 1
    assert model.understand("2", presenting).selection_index == 1
    idle = fresh_state(stage=Stage.IDLE)
    assert model.understand("2", idle).selection_index is None
