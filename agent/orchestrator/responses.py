"""Deterministic response templates. Kept separate from the NLU layer so
swapping the understanding model never changes what the agent actually says
for a given state -- this is what makes eval assertions and the recorded
demo reproducible regardless of which Model is plugged in.
"""
from __future__ import annotations

SLOT_QUESTIONS = {
    "service": "What kind of service do you need (e.g. plumber, electrician, AC servicing)?",
    "area": "Which area should I search in?",
    "time_window": "What time window works for you (e.g. this week, mornings, Saturday)?",
}


def ask_for_slot(field: str) -> str:
    return SLOT_QUESTIONS.get(field, f"Could you tell me the {field.replace('_', ' ')}?")


def greeting() -> str:
    return "Hi! I can help you book, reschedule, or cancel a home-service appointment. What do you need?"


def present_candidates(candidates: list) -> str:
    lines = ["Here's what I found:"]
    for i, c in enumerate(candidates, start=1):
        slots_str = ", ".join(c["free_slots"][:3])
        lines.append(f"{i}. {c['name']} — ₹{c['price']}, {c['rating']}★, slots: {slots_str}")
    lines.append("Which one would you like (reply with a number)?")
    return "\n".join(lines)


def present_bookings_for_selection(bookings: list) -> str:
    lines = ["Which booking do you mean?"]
    for i, b in enumerate(bookings, start=1):
        lines.append(f"{i}. {b['service']} on {b['slot']} (booking {b['id']})")
    return "\n".join(lines)


def no_candidates_found() -> str:
    return "I couldn't find any providers matching that. Want to loosen the budget or time window?"


def no_matching_booking() -> str:
    return "I don't see a matching upcoming booking for that."


def propose_booking_approval(provider_name: str, slot: str, price: int) -> str:
    return (
        f"I'll book {provider_name} for {slot} at ₹{price}. "
        f"Reply 'yes' to confirm, or tell me what to change."
    )


def propose_cancel_approval(provider_name: str, slot: str, booking_id: str) -> str:
    return f"I'll cancel your booking with {provider_name} for {slot} (booking {booking_id}). Confirm?"


def propose_reschedule_approval(provider_name: str, old_slot: str, new_slot: str) -> str:
    return f"I'll move your booking with {provider_name} from {old_slot} to {new_slot}. Confirm?"


def approval_voided_price_changed(old_price: int, new_price: int) -> str:
    return (
        f"Heads up — the price changed from ₹{old_price} to ₹{new_price} since you approved this. "
        f"I won't book it without your OK on the new price. Proceed at ₹{new_price}? (yes/no)"
    )


def approval_voided_slot_unavailable() -> str:
    return (
        "That slot just became unavailable, so I can't proceed with the booking you approved. "
        "Want me to look for another slot?"
    )


def booking_confirmed(provider_name: str, slot: str, price: int, booking_id: str) -> str:
    return f"Booked! {provider_name} on {slot} for ₹{price}. Booking ID: {booking_id}."


def booking_failed(reason: str = "") -> str:
    return f"That booking didn't go through. {reason}".strip()


def slot_taken() -> str:
    return "Sorry, that slot was just taken by someone else. Want me to look for another one?"


def hold_expired() -> str:
    return "The hold on that slot expired before I could book it. Want me to try again?"


def cancelled(booking_id: str) -> str:
    return f"Done — booking {booking_id} is cancelled."


def rescheduled(provider_name: str, new_slot: str) -> str:
    return f"Done — your booking with {provider_name} is now set for {new_slot}."


def reschedule_slot_conflict() -> str:
    return "That new time conflicts with another booking. Want to pick a different slot?"


def reschedule_no_matching_slot() -> str:
    return "That provider doesn't have a free slot matching that time. Want to try a different day or time?"


def denied() -> str:
    return "No problem — what would you like to change?"


def limit_exceeded(kind: str) -> str:
    return (
        "I wasn't able to finish this within my safety limits, so I stopped rather than keep retrying "
        f"(hit the {kind.replace('_', ' ')} limit). Nothing was booked beyond what you already approved."
    )


def unexpected_error() -> str:
    return "Something went wrong on my end and I had to stop. Nothing further was booked or changed."


def bookings_list(bookings: list) -> str:
    if not bookings:
        return "You have no bookings yet."
    lines = ["Here are your bookings:"]
    for b in bookings:
        lines.append(f"- {b['service']} on {b['slot']} [{b['status']}] (booking {b['id']})")
    return "\n".join(lines)


def memory_summary(facts: list) -> str:
    if not facts:
        return "I don't have any saved preferences for you yet."
    lines = ["Here's what I remember about you:"]
    for f in facts:
        tag = "confirmed" if f["confirmed"] else "inferred, unconfirmed"
        lines.append(f"- {f['key']}: {f['value']} [{f['source']}, {tag}]")
    return "\n".join(lines)


def memory_forgotten(key: str) -> str:
    return f"Done — I've forgotten your {key.replace('_', ' ')}."
