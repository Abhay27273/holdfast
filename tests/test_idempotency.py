import pytest

from agent.mocks.provider_api import ProviderAPI, ScenarioConfig, TransientError503
from agent.core.idempotency import resolve_book_outcome, retry_cancel, new_idempotency_key


def setup_hold(api):
    results = api.search_providers("plumber", "area", ("x", "y"))
    provider_id = results[0]["id"]
    hold = api.hold_slot(provider_id, results[0]["free_slots"][0])
    return provider_id, hold


def test_resolve_book_outcome_direct_success():
    api = ProviderAPI(user_id="u1", seed=1)
    provider_id, hold = setup_hold(api)
    api.scenario.forced_book_outcome[provider_id] = "booked"
    result = resolve_book_outcome(api, hold["hold_id"], new_idempotency_key())
    assert result["status"] == "booked"
    assert result["booking_id"]


def test_resolve_book_outcome_timeout_then_booked_never_calls_book_twice():
    api = ProviderAPI(user_id="u1", seed=1)
    provider_id, hold = setup_hold(api)
    api.scenario.forced_book_outcome[provider_id] = "timeout"
    api.scenario.forced_timeout_secret[provider_id] = "booked"

    call_count = {"n": 0}
    original_book = api.book

    def counting_book(*args, **kwargs):
        call_count["n"] += 1
        return original_book(*args, **kwargs)

    api.book = counting_book
    result = resolve_book_outcome(api, hold["hold_id"], "idem-x")
    assert result["status"] == "booked"
    assert call_count["n"] == 1  # book() called exactly once; status checked via get_booking_status


def test_resolve_book_outcome_timeout_then_failed():
    api = ProviderAPI(user_id="u1", seed=1)
    provider_id, hold = setup_hold(api)
    api.scenario.forced_book_outcome[provider_id] = "timeout"
    api.scenario.forced_timeout_secret[provider_id] = "failed"
    result = resolve_book_outcome(api, hold["hold_id"], "idem-y")
    assert result == {"status": "failed"}


def test_resolve_book_outcome_slot_taken_passthrough():
    api = ProviderAPI(user_id="u1", seed=1)
    provider_id, hold = setup_hold(api)
    api.scenario.forced_book_outcome[provider_id] = "slot_taken"
    result = resolve_book_outcome(api, hold["hold_id"], "idem-z")
    assert result == {"status": "slot_taken"}


def test_retry_cancel_recovers_from_transient_503():
    api = ProviderAPI(user_id="u1", seed=1)
    provider_id, hold = setup_hold(api)
    api.scenario.forced_book_outcome[provider_id] = "booked"
    booked = resolve_book_outcome(api, hold["hold_id"], "idem-cancel-test")
    result = retry_cancel(api, booked["booking_id"], new_idempotency_key("cancel"))
    assert result == {"status": "cancelled"}


def test_retry_cancel_gives_up_after_max_attempts_if_always_failing(monkeypatch):
    api = ProviderAPI(user_id="u1", seed=1)
    provider_id, hold = setup_hold(api)
    api.scenario.forced_book_outcome[provider_id] = "booked"
    booked = resolve_book_outcome(api, hold["hold_id"], "idem-cancel-test-2")

    def always_fail(*args, **kwargs):
        raise TransientError503("still down")

    monkeypatch.setattr(api, "cancel", always_fail)
    with pytest.raises(TransientError503):
        retry_cancel(api, booked["booking_id"], "k", max_attempts=2)
