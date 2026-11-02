from agent.core.approval import (
    create_pending_approval, parse_confirmation, check_approval, PendingApproval,
)


def test_parse_confirmation_recognizes_yes_and_no():
    assert parse_confirmation("yes") is True
    assert parse_confirmation("Yes, go ahead") is True
    assert parse_confirmation("book it") is True
    assert parse_confirmation("no") is False
    assert parse_confirmation("never mind") is False


def test_parse_confirmation_returns_none_for_unrelated_text():
    assert parse_confirmation("what's the weather like") is None
    assert parse_confirmation("make it mornings only") is None


def test_check_approval_false_when_nothing_pending():
    assert check_approval(None, "book", provider_id="p1", slot="s1", price=100) is False


def test_check_approval_false_when_not_confirmed():
    pending = create_pending_approval("book", "proposal", provider_id="p1", slot="s1", price=100)
    assert check_approval(pending, "book", provider_id="p1", slot="s1", price=100) is False


def test_check_approval_true_only_for_exact_match_after_confirmation():
    pending = create_pending_approval("book", "proposal", provider_id="p1", slot="s1", price=100)
    pending.confirmed = True
    assert check_approval(pending, "book", provider_id="p1", slot="s1", price=100) is True


def test_approval_voided_by_price_change():
    """This is the core anti-manipulation guarantee: if hold_slot quotes a
    different price than what the user approved, the approval no longer
    matches and book() must not be allowed to proceed."""
    pending = create_pending_approval("book", "proposal", provider_id="p1", slot="s1", price=100)
    pending.confirmed = True
    assert check_approval(pending, "book", provider_id="p1", slot="s1", price=150) is False


def test_approval_voided_by_slot_change():
    pending = create_pending_approval("book", "proposal", provider_id="p1", slot="s1", price=100)
    pending.confirmed = True
    assert check_approval(pending, "book", provider_id="p1", slot="DIFFERENT_SLOT", price=100) is False


def test_approval_voided_by_provider_change():
    pending = create_pending_approval("book", "proposal", provider_id="p1", slot="s1", price=100)
    pending.confirmed = True
    assert check_approval(pending, "book", provider_id="p2", slot="s1", price=100) is False


def test_approval_does_not_cross_actions():
    pending = create_pending_approval("book", "proposal", provider_id="p1", slot="s1", price=100)
    pending.confirmed = True
    assert check_approval(pending, "cancel", provider_id="p1", slot="s1", price=100) is False


def test_approval_round_trips_through_dict_for_persistence():
    pending = create_pending_approval("reschedule", "proposal", booking_id="b1", slot="s2")
    pending.confirmed = True
    restored = PendingApproval.from_dict(pending.to_dict())
    assert restored.matches("reschedule", booking_id="b1", slot="s2")
