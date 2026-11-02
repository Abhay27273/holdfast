import time

import pytest

from agent.mocks.provider_api import ProviderAPI, ScenarioConfig, TransientError503, ProviderAPIError


def make_api(**kwargs):
    return ProviderAPI(user_id="u1", seed=7, **kwargs)


def test_search_returns_3_to_8_providers_and_is_stable():
    api = make_api()
    results = api.search_providers("plumber", "koramangala", ("2026-11-01", "2026-11-07"))
    assert 3 <= len(results) <= 8
    again = api.search_providers("plumber", "koramangala", ("2026-11-01", "2026-11-07"))
    assert [p["id"] for p in results] == [p["id"] for p in again]


def test_get_provider_always_includes_an_injection_review():
    from agent.mocks.fixtures import INJECTION_REVIEWS
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    detail = api.get_provider(results[0]["id"])
    assert any(r in INJECTION_REVIEWS for r in detail["reviews"])


def test_list_bookings_seeds_a_completed_job_and_an_upcoming_booking():
    api = make_api()
    bookings = api.list_bookings("u1")
    assert len(bookings) == 2
    by_status = {b["status"]: b for b in bookings}
    assert by_status["completed"]["service"] == "electrician"
    assert by_status["booked"]["service"] == "ac_technician"


def test_hold_slot_unavailable_override():
    api = make_api(scenario=ScenarioConfig(unavailable_holds={("prov_x", "slot_1")}))
    api._providers["prov_x"] = api._make_provider("plumber", "area")
    api._providers["prov_x"].id = "prov_x"
    result = api.hold_slot("prov_x", "slot_1")
    assert result == {"status": "slot_unavailable"}


def test_book_is_idempotent_on_key_and_never_double_books():
    api = make_api(scenario=ScenarioConfig(forced_book_outcome={}))
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[results[0]["id"]] = "booked"
    first = api.book(hold["hold_id"], "idem-1")
    second = api.book(hold["hold_id"], "idem-1")  # retry with same key
    assert first == second
    assert len(api._bookings) == 3  # 2 seeded history bookings + this one


def test_book_timeout_then_booked_is_discoverable_via_status_not_a_second_booking():
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[results[0]["id"]] = "timeout"
    api.scenario.forced_timeout_secret[results[0]["id"]] = "booked"
    book_result = api.book(hold["hold_id"], "idem-2")
    assert book_result["status"] == "timeout"
    assert "booking_id" not in book_result
    status = api.get_booking_status("idem-2")
    assert status["status"] == "booked"
    assert status["booking_id"]
    # retrying book() with the same key must not create a second booking
    retry = api.book(hold["hold_id"], "idem-2")
    assert retry == book_result


def test_book_timeout_then_failed():
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[results[0]["id"]] = "timeout"
    api.scenario.forced_timeout_secret[results[0]["id"]] = "failed"
    api.book(hold["hold_id"], "idem-3")
    status = api.get_booking_status("idem-3")
    assert status == {"status": "failed"}


def test_get_booking_status_not_found_for_unknown_key():
    api = make_api()
    assert api.get_booking_status("never-seen") == {"status": "not_found"}


def test_book_past_hold_expiry_returns_hold_expired():
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api._holds[hold["hold_id"]].expires_at = time.time() - 1
    result = api.book(hold["hold_id"], "idem-4")
    assert result == {"status": "hold_expired"}


def test_cancel_fails_first_then_succeeds_on_retry():
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[results[0]["id"]] = "booked"
    booked = api.book(hold["hold_id"], "idem-5")
    booking_id = booked["booking_id"]

    with pytest.raises(TransientError503):
        api.cancel(booking_id, "cancel-idem-1")

    result = api.cancel(booking_id, "cancel-idem-1")
    assert result == {"status": "cancelled"}


def test_cancel_retry_is_idempotent_after_success():
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[results[0]["id"]] = "booked"
    booked = api.book(hold["hold_id"], "idem-6")
    booking_id = booked["booking_id"]
    with pytest.raises(TransientError503):
        api.cancel(booking_id, "cancel-idem-2")
    first_success = api.cancel(booking_id, "cancel-idem-2")
    second_call = api.cancel(booking_id, "cancel-idem-2")
    assert first_success == second_call == {"status": "cancelled"}


def test_reschedule_slot_conflict_override():
    api = make_api(scenario=ScenarioConfig(slot_conflicts={"bad-slot"}))
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[results[0]["id"]] = "booked"
    booked = api.book(hold["hold_id"], "idem-7")
    result = api.reschedule(booked["booking_id"], "bad-slot", "resched-1")
    assert result == {"status": "slot_conflict"}


def test_reschedule_success_updates_slot():
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    hold = api.hold_slot(results[0]["id"], results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[results[0]["id"]] = "booked"
    booked = api.book(hold["hold_id"], "idem-8")
    result = api.reschedule(booked["booking_id"], "2026-12-01T10:00:00", "resched-2")
    assert result["status"] == "updated"
    assert result["booking"]["slot"] == "2026-12-01T10:00:00"


def test_list_bookings_rejects_mismatched_user():
    api = make_api()
    with pytest.raises(ProviderAPIError):
        api.list_bookings("someone-else")


def test_price_drift_changes_quoted_price_vs_search_price():
    api = make_api()
    results = api.search_providers("plumber", "area", ("x", "y"))
    provider_id = results[0]["id"]
    search_price = results[0]["price"]
    api.scenario.price_drift[provider_id] = search_price + 500
    hold = api.hold_slot(provider_id, results[0]["free_slots"][0])
    assert hold["quoted_price"] == search_price + 500
