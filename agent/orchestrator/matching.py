"""Pure helpers for matching a provider's free_slots against the loose
time_window / earliest_time constraints the NLU extracts. Kept separate from
agent_loop.py so the matching logic can be unit-tested without spinning up a
whole conversation.
"""
from __future__ import annotations

import time
from typing import Optional

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAYPART_RANGES = {
    "mornings": (0, 12),
    "afternoons": (12, 17),
    "evenings": (17, 24),
}


def _weekday(slot_iso: str) -> str:
    return time.strftime("%A", time.strptime(slot_iso[:10], "%Y-%m-%d")).lower()


def _minutes_of_day(slot_iso: str) -> int:
    hh, mm = slot_iso[11:16].split(":")
    return int(hh) * 60 + int(mm)


def matching_slots(free_slots: list, time_window: Optional[str], earliest_time: Optional[str]) -> list:
    candidates = list(free_slots)

    if time_window:
        tw = time_window.lower()
        for day in DAY_NAMES:
            if day in tw:
                day_matches = [s for s in candidates if _weekday(s) == day]
                if day_matches:
                    candidates = day_matches
                break
        for part, (start_hour, end_hour) in DAYPART_RANGES.items():
            if part in tw or part.rstrip("s") in tw:
                part_matches = [s for s in candidates if start_hour * 60 <= _minutes_of_day(s) < end_hour * 60]
                candidates = part_matches
                break

    if earliest_time:
        hh, mm = earliest_time.split(":")
        threshold = int(hh) * 60 + int(mm)
        candidates = [s for s in candidates if _minutes_of_day(s) >= threshold]

    return candidates


def best_matching_slot(free_slots: list, time_window: Optional[str], earliest_time: Optional[str]) -> Optional[str]:
    matches = matching_slots(free_slots, time_window, earliest_time)
    return matches[0] if matches else None
