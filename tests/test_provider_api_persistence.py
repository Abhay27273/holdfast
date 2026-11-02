import shutil
import tempfile
from pathlib import Path

import pytest

from agent.mocks.provider_api import ProviderAPI


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_snapshot_round_trip_preserves_holds_and_bookings(tmp_dir):
    api = ProviderAPI(user_id="u1", seed=3)
    results = api.search_providers("plumber", "area", ("x", "y"))
    provider_id = results[0]["id"]
    hold = api.hold_slot(provider_id, results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[provider_id] = "timeout"
    api.scenario.forced_timeout_secret[provider_id] = "booked"
    api.book(hold["hold_id"], "idem-persist-1")
    api.save(data_dir=tmp_dir)

    restarted = ProviderAPI.load_or_create("u1", seed=3, data_dir=tmp_dir)
    status = restarted.get_booking_status("idem-persist-1")
    assert status["status"] == "booked"
    assert status["booking_id"]
    # the backend must not have re-seeded history (that would look like a fresh user)
    assert len(restarted.list_bookings("u1")) == 3  # 2 seeded + this one


def test_load_or_create_seeds_history_only_when_no_snapshot_exists(tmp_dir):
    api = ProviderAPI.load_or_create("fresh_user", seed=1, data_dir=tmp_dir)
    assert len(api.list_bookings("fresh_user")) == 2  # seeded history


def test_restart_does_not_allow_double_booking_same_hold(tmp_dir):
    api = ProviderAPI(user_id="u2", seed=5)
    results = api.search_providers("plumber", "area", ("x", "y"))
    provider_id = results[0]["id"]
    hold = api.hold_slot(provider_id, results[0]["free_slots"][0])
    api.scenario.forced_book_outcome[provider_id] = "booked"
    api.book(hold["hold_id"], "idem-restart-dup")
    api.save(data_dir=tmp_dir)

    restarted = ProviderAPI.load_or_create("u2", seed=5, data_dir=tmp_dir)
    # simulate the client retrying the same logical action after "restart"
    # with the same idempotency key -- must replay, not double-book
    replay = restarted.book(hold["hold_id"], "idem-restart-dup")
    assert replay["status"] == "booked"
    assert len(restarted.list_bookings("u2")) == 3  # 2 seeded + the one booking, not two of it
