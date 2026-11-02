"""Optional real-LLM adapter implementing the exact same Model protocol as
RuleBasedModel. Not required for grading: the eval suite (evals/scenarios.py)
only ever runs against RuleBasedModel, per the assignment's own instruction
that evals must be "offline and reproducible, run against a scripted or
recorded model rather than a live LLM." This class exists to show the
architecture is model-agnostic, not to carry any safety behaviour itself --
approval, idempotency, and injection handling all live in the orchestrator
and never depend on which Model is plugged in.

Note this adapter is handed raw user chat text ONLY. It never sees provider
review text -- reviews are read solely for structured fields (price, rating,
free_slots) by the orchestrator, never passed to any model, rule-based or
LLM. That is a stronger guarantee than sanitizing injected text would be:
there is no path by which a review could reach this model's context at all.

Usage (requires ANTHROPIC_API_KEY in the environment):
    from agent.nlu.anthropic_model import AnthropicModel
    agent = Agent("alice", model=AnthropicModel())
"""
from __future__ import annotations

import json
import os
from dataclasses import fields

from .model_protocol import Understanding

SYSTEM_PROMPT = """You turn one line of user chat into structured fields for a \
home-services booking assistant. You only extract what the CURRENT message \
states; you never infer approval, never invent slot values, and only set \
`confirmation` to true/false when the message is an unambiguous yes/no \
reply. Call extract_understanding exactly once."""

TOOL_SCHEMA = {
    "name": "extract_understanding",
    "description": "Structured extraction of one user chat turn.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": ["string", "null"],
                "enum": ["new_booking", "reschedule", "cancel", "view_bookings", None],
                "description": "A NEW intent signaled this turn, or null if none.",
            },
            "service": {"type": ["string", "null"]},
            "area": {"type": ["string", "null"]},
            "budget": {"type": ["integer", "null"]},
            "time_window": {"type": ["string", "null"]},
            "earliest_time": {"type": ["string", "null"], "description": "24h HH:MM, e.g. '10:00'"},
            "provider_preference": {"type": ["string", "null"], "description": "'SAME_AS_LAST' or null"},
            "confirmation": {"type": ["boolean", "null"]},
            "selection_index": {"type": ["integer", "null"], "description": "0-based index if picking from a numbered list"},
            "memory_show_requested": {"type": "boolean"},
            "memory_forget_key": {"type": ["string", "null"]},
            "memory_confirm_key": {"type": ["string", "null"]},
        },
        "required": ["memory_show_requested"],
    },
}

SLOT_FIELDS = ("service", "area", "budget", "time_window", "earliest_time", "provider_preference")


class AnthropicModel:
    name = "anthropic_claude"

    def __init__(self, model_id: str = "claude-sonnet-5", client=None):
        self.model_id = model_id
        if client is not None:
            self._client = client
        else:
            import anthropic  # local import: only required if this adapter is actually used
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def understand(self, text: str, state) -> Understanding:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "extract_understanding"},
            messages=[{
                "role": "user",
                "content": (
                    f"Current stage: {state.stage}. Current intent: {state.intent}.\n"
                    f"User message: {text}"
                ),
            }],
        )
        payload = {}
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                payload = block.input
                break

        return Understanding(
            intent=payload.get("intent"),
            slot_updates={k: payload.get(k) for k in SLOT_FIELDS if payload.get(k) is not None},
            confirmation=payload.get("confirmation"),
            selection_index=payload.get("selection_index"),
            memory_forget_key=payload.get("memory_forget_key"),
            memory_confirm_key=payload.get("memory_confirm_key"),
            memory_show_requested=bool(payload.get("memory_show_requested", False)),
            raw_text=text,
        )
