from agent.orchestrator.matching import matching_slots, best_matching_slot

SLOTS = [
    "2030-03-17T09:00:00",  # Sunday morning
    "2030-03-22T16:00:00",  # Friday afternoon
    "2030-03-23T08:00:00",  # Saturday morning
    "2030-03-23T17:00:00",  # Saturday evening
]


def test_filters_by_day_name():
    assert matching_slots(SLOTS, "saturday", None) == ["2030-03-23T08:00:00", "2030-03-23T17:00:00"]


def test_filters_by_daypart():
    assert matching_slots(SLOTS, "mornings", None) == ["2030-03-17T09:00:00", "2030-03-23T08:00:00"]


def test_filters_by_day_and_daypart_combined():
    assert matching_slots(SLOTS, "saturday mornings", None) == ["2030-03-23T08:00:00"]


def test_filters_by_earliest_time():
    assert matching_slots(SLOTS, None, "10:00") == ["2030-03-22T16:00:00", "2030-03-23T17:00:00"]


def test_combined_day_and_earliest_time():
    assert matching_slots(SLOTS, "saturday", "10:00") == ["2030-03-23T17:00:00"]


def test_no_constraints_returns_everything():
    assert matching_slots(SLOTS, None, None) == SLOTS


def test_best_matching_slot_returns_first_or_none():
    assert best_matching_slot(SLOTS, "saturday mornings", None) == "2030-03-23T08:00:00"
    assert best_matching_slot(SLOTS, "sunday evenings", None) is None
