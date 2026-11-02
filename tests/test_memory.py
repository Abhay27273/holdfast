import shutil
import tempfile
from pathlib import Path

import pytest

from agent.core.memory import MemoryStore


@pytest.fixture
def tmp_memory_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_stated_fact_is_usable_immediately(tmp_memory_dir):
    store = MemoryStore("alice", data_dir=tmp_memory_dir)
    store.set_fact("area", "koramangala", source="stated")
    assert store.get_fact("area") == "koramangala"


def test_inferred_fact_not_used_until_confirmed(tmp_memory_dir):
    store = MemoryStore("alice", data_dir=tmp_memory_dir)
    store.set_fact("preferred_technician", "prov_42", source="inferred")
    assert store.get_fact("preferred_technician") is None  # not confirmed yet
    assert store.get_fact("preferred_technician", only_confirmed=False) == "prov_42"
    store.confirm("preferred_technician")
    assert store.get_fact("preferred_technician") == "prov_42"


def test_facts_persist_across_store_instances(tmp_memory_dir):
    MemoryStore("alice", data_dir=tmp_memory_dir).set_fact("area", "hsr layout", source="stated")
    reloaded = MemoryStore("alice", data_dir=tmp_memory_dir)
    assert reloaded.get_fact("area") == "hsr layout"


def test_forgotten_fact_is_never_reused(tmp_memory_dir):
    store = MemoryStore("alice", data_dir=tmp_memory_dir)
    store.set_fact("area", "koramangala", source="stated")
    store.forget("area")
    assert store.get_fact("area") is None
    assert store.is_forgotten("area")


def test_forgotten_fact_is_never_silently_reinferred(tmp_memory_dir):
    store = MemoryStore("alice", data_dir=tmp_memory_dir)
    store.set_fact("preferred_technician", "prov_1", source="inferred")
    store.forget("preferred_technician")
    # agent re-infers the same pattern later in the session -- must be ignored
    store.set_fact("preferred_technician", "prov_1", source="inferred")
    assert store.get_fact("preferred_technician", only_confirmed=False) is None
    assert store.is_forgotten("preferred_technician")


def test_forgotten_fact_can_be_reestablished_by_explicit_user_statement(tmp_memory_dir):
    store = MemoryStore("alice", data_dir=tmp_memory_dir)
    store.set_fact("area", "koramangala", source="stated")
    store.forget("area")
    store.set_fact("area", "indiranagar", source="stated")  # user explicitly states it again
    assert store.get_fact("area") == "indiranagar"
    assert not store.is_forgotten("area")


def test_list_facts_and_unconfirmed_inferred(tmp_memory_dir):
    store = MemoryStore("bob", data_dir=tmp_memory_dir)
    store.set_fact("area", "area1", source="stated")
    store.set_fact("preferred_technician", "prov_9", source="inferred")
    unconfirmed = store.unconfirmed_inferred()
    assert len(unconfirmed) == 1
    assert unconfirmed[0]["key"] == "preferred_technician"
