"""Per-user persistent preference memory.

Facts are tagged `stated` (the user said it directly) or `inferred` (the
agent noticed a pattern, e.g. booking the same provider twice). Inferred
facts start unconfirmed and must never be *used* until a human confirms
them. `forget()` writes a tombstone: a forgotten key is never silently
reused or re-inferred, even if the pattern that produced it recurs -- it
only comes back if the user states it again directly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "memory"


@dataclass
class Fact:
    key: str
    value: object
    source: str  # "stated" | "inferred"
    confirmed: bool
    created_at: float

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryStore:
    def __init__(self, user_id: str, data_dir: Optional[Path] = None):
        self.user_id = user_id
        self.dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{user_id}.json"
        self._facts: dict[str, Fact] = {}
        self._forgotten: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for key, fd in raw.get("facts", {}).items():
            self._facts[key] = Fact(**fd)
        self._forgotten = set(raw.get("forgotten", []))

    def _save(self) -> None:
        payload = {
            "facts": {k: f.to_dict() for k, f in self._facts.items()},
            "forgotten": sorted(self._forgotten),
        }
        self.path.write_text(json.dumps(payload, indent=2))

    def set_fact(self, key: str, value, source: str = "stated") -> None:
        if key in self._forgotten:
            if source == "inferred":
                return  # tombstoned: never silently re-infer a forgotten fact
            self._forgotten.discard(key)  # explicit re-statement overrides a forget
        confirmed = source == "stated"
        self._facts[key] = Fact(key=key, value=value, source=source, confirmed=confirmed, created_at=time.time())
        self._save()

    def get_fact(self, key: str, only_confirmed: bool = True):
        fact = self._facts.get(key)
        if fact is None:
            return None
        if only_confirmed and not fact.confirmed:
            return None
        return fact.value

    def confirm(self, key: str) -> None:
        fact = self._facts.get(key)
        if fact is not None:
            fact.confirmed = True
            self._save()

    def forget(self, key: str) -> None:
        self._facts.pop(key, None)
        self._forgotten.add(key)
        self._save()

    def is_forgotten(self, key: str) -> bool:
        return key in self._forgotten

    def list_facts(self) -> list:
        return [f.to_dict() for f in self._facts.values()]

    def unconfirmed_inferred(self) -> list:
        return [f.to_dict() for f in self._facts.values() if f.source == "inferred" and not f.confirmed]
