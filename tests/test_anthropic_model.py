"""Tests the AnthropicModel adapter's parsing logic with a fake client --
no network access, no API key required. This only proves the adapter
correctly translates a tool_use response into an Understanding; it is not
part of the graded eval suite (which runs exclusively against
RuleBasedModel, per the assignment's offline/reproducible requirement)."""
from dataclasses import dataclass

from agent.core.state import ConversationState
from agent.nlu.anthropic_model import AnthropicModel


@dataclass
class FakeToolUseBlock:
    type: str
    input: dict


@dataclass
class FakeResponse:
    content: list


class FakeMessages:
    def __init__(self, input_payload):
        self._input_payload = input_payload

    def create(self, **kwargs):
        return FakeResponse(content=[FakeToolUseBlock(type="tool_use", input=self._input_payload)])


class FakeClient:
    def __init__(self, input_payload):
        self.messages = FakeMessages(input_payload)


def make_model(payload):
    return AnthropicModel(client=FakeClient(payload))


def test_translates_tool_use_payload_into_understanding():
    model = make_model({
        "intent": "new_booking", "service": "plumber", "area": "koramangala",
        "budget": 1500, "time_window": "this_week mornings", "earliest_time": None,
        "provider_preference": None, "confirmation": None, "selection_index": None,
        "memory_show_requested": False, "memory_forget_key": None, "memory_confirm_key": None,
    })
    u = model.understand("Find a plumber in koramangala this week, mornings only, under 1500", ConversationState.new("u1"))
    assert u.intent == "new_booking"
    assert u.slot_updates == {"service": "plumber", "area": "koramangala", "budget": 1500,
                               "time_window": "this_week mornings"}
    assert u.confirmation is None


def test_null_fields_are_excluded_from_slot_updates():
    model = make_model({
        "intent": None, "service": None, "area": None, "budget": None,
        "time_window": None, "earliest_time": None, "provider_preference": None,
        "confirmation": True, "selection_index": None, "memory_show_requested": False,
        "memory_forget_key": None, "memory_confirm_key": None,
    })
    u = model.understand("yes", ConversationState.new("u1"))
    assert u.slot_updates == {}
    assert u.confirmation is True


def test_memory_fields_pass_through():
    model = make_model({
        "intent": None, "service": None, "area": None, "budget": None,
        "time_window": None, "earliest_time": None, "provider_preference": None,
        "confirmation": None, "selection_index": None, "memory_show_requested": True,
        "memory_forget_key": "area", "memory_confirm_key": None,
    })
    u = model.understand("what do you remember about me", ConversationState.new("u1"))
    assert u.memory_show_requested is True
    assert u.memory_forget_key == "area"
