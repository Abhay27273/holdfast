"""Deterministic rule-based NLU -- the default agent "brain".

No API key required, fully deterministic, and this is exactly what the
evals run against (satisfying the assignment's "offline and reproducible,
run against a scripted ... model rather than a live LLM" -- this parser IS
the scripted model). An optional real-LLM adapter (anthropic_model.py)
implements the same Model protocol for more natural conversation, but nnone
of the graded safety behaviour depends on it.

This module only extracts structure from text. It never reads provider
reviews (untrusted free text) at all -- it only ever looks at the raw user
chat message -- so it is structurally immune to injection content that
might appear in tool output (must-have #4).
"""
from __future__ import annotations

import re
from typing import Optional

from agent.core.approval import parse_confirmation
from agent.core.state import Intent, Stage
from .model_protocol import Model, Understanding

SERVICE_SYNONYMS = {
    "plumber": ["plumber", "plumbing", "tap", "leak", "pipe", "faucet"],
    "electrician": ["electrician", "electrical", "wiring", "switch board", "fuse", "mcb"],
    "ac_technician": ["ac servicing", "ac repair", "air conditioner", "ac technician", "air-conditioner", " ac "],
    "carpenter": ["carpenter", "furniture repair", "woodwork"],
    "cleaner": ["cleaner", "cleaning", "deep clean", "housekeeping"],
    "painter": ["painter", "painting"],
    "pest_control": ["pest control", "cockroach", "termite"],
}

AREA_BANK = [
    "koramangala", "indiranagar", "hsr layout", "whitefield", "jayanagar",
    "btm layout", "electronic city", "marathahalli",
]

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

DAYPART_KEYWORDS = {
    "mornings": ["morning"],
    "afternoons": ["afternoon"],
    "evenings": ["evening"],
}

BUDGET_RE = re.compile(r"(?:under|below|less than|up to|max(?:imum)?)\s*₹?\s*([\d,]+)", re.I)
EARLIEST_TIME_RE = re.compile(r"(?:not before|after|from)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.I)
SAME_AS_LAST_RE = re.compile(
    r"same (?:provider|electrician|plumber|technician|person|one)?\s*as (?:last time|before)", re.I,
)
SELECTION_RE = re.compile(r"(?:option|number|#)\s*(\d+)|^\s*(\d+)\s*$", re.I)
ORDINAL_WORDS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}

CANCEL_BOOKING_RE = re.compile(r"\bcancel\b", re.I)
RESCHEDULE_RE = re.compile(r"\breschedule\b|\bmove\b.*\b(appointment|booking|servicing)\b|\bpush\b.*\bto\b", re.I)
VIEW_BOOKINGS_RE = re.compile(r"\b(my|show|list) bookings?\b|\bwhat.*booked\b", re.I)
MEMORY_SHOW_RE = re.compile(r"what do you (remember|know about me)|show my preferences|show my memory", re.I)
MEMORY_FORGET_RE = re.compile(r"forget (?:my |that )?(\w+)", re.I)
MEMORY_CONFIRM_RE = re.compile(r"remember (?:my |that )?(\w+)", re.I)

MEMORY_KEY_ALIASES = {
    "area": "area", "location": "area", "budget": "budget",
    "technician": "preferred_technician", "provider": "preferred_technician",
    "everything": "__all__", "all": "__all__",
    "time": "earliest_time", "earliest": "earliest_time",
    "preferences": "__all__",
}


def _extract_service(text: str) -> Optional[str]:
    for service, keywords in SERVICE_SYNONYMS.items():
        for kw in keywords:
            if kw in text:
                return service
    return None


def _extract_area(text: str) -> Optional[str]:
    for area in AREA_BANK:
        if area in text:
            return area
    return None


def _extract_budget(text: str) -> Optional[int]:
    m = BUDGET_RE.search(text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def _extract_time_window(text: str) -> Optional[str]:
    parts = []
    if "this week" in text:
        parts.append("this_week")
    for day in DAY_NAMES:
        if day in text:
            parts.append(day)
            break
    if "today" in text:
        parts.append("today")
    if "tomorrow" in text:
        parts.append("tomorrow")
    for canonical, keywords in DAYPART_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            parts.append(canonical)
            break
    if not parts:
        return None
    return " ".join(parts)


def _extract_earliest_time(text: str) -> Optional[str]:
    m = EARLIEST_TIME_RE.search(text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = m.group(2) or "00"
    meridiem = (m.group(3) or "").lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute}"


def _extract_provider_preference(text: str) -> Optional[str]:
    if SAME_AS_LAST_RE.search(text):
        return "SAME_AS_LAST"
    return None


def _extract_selection_index(text: str, stage: str) -> Optional[int]:
    if stage != Stage.PRESENTING.value:
        return None
    for word, idx in ORDINAL_WORDS.items():
        if word in text:
            return idx
    m = SELECTION_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        if raw:
            return int(raw) - 1
    return None


class RuleBasedModel:
    name = "rule_based_v1"

    def understand(self, text: str, state) -> Understanding:
        lowered = f" {text.lower().strip()} "

        u = Understanding(raw_text=text)

        # Confirmation is resolved relative to what's actually pending: a
        # lone "cancel"/"no" while a proposal is awaiting approval means
        # "reject this proposal", not "cancel my booking" -- context from
        # `state` disambiguates this correctly.
        u.confirmation = parse_confirmation(text)

        memory_show = bool(MEMORY_SHOW_RE.search(lowered))
        forget_match = MEMORY_FORGET_RE.search(lowered)
        confirm_match = MEMORY_CONFIRM_RE.search(lowered)
        if memory_show:
            u.memory_show_requested = True
        elif forget_match:
            alias = forget_match.group(1).lower()
            u.memory_forget_key = MEMORY_KEY_ALIASES.get(alias, alias)
        elif confirm_match:
            alias = confirm_match.group(1).lower()
            u.memory_confirm_key = MEMORY_KEY_ALIASES.get(alias, alias)

        awaiting_approval = state.stage == Stage.AWAITING_APPROVAL.value

        if not awaiting_approval or u.confirmation is None:
            if CANCEL_BOOKING_RE.search(lowered) and not awaiting_approval:
                u.intent = Intent.CANCEL.value
            elif RESCHEDULE_RE.search(lowered):
                u.intent = Intent.RESCHEDULE.value
            elif VIEW_BOOKINGS_RE.search(lowered):
                u.intent = Intent.VIEW_BOOKINGS.value
            elif _extract_service(lowered) is not None and state.intent in (None, Intent.UNKNOWN.value, ""):
                u.intent = Intent.NEW_BOOKING.value

        u.slot_updates = {
            k: v for k, v in {
                "service": _extract_service(lowered),
                "area": _extract_area(lowered),
                "budget": _extract_budget(lowered),
                "time_window": _extract_time_window(lowered),
                "earliest_time": _extract_earliest_time(lowered),
                "provider_preference": _extract_provider_preference(lowered),
            }.items() if v is not None
        }

        u.selection_index = _extract_selection_index(lowered, state.stage)

        return u
