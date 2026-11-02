"""Mock implementation of the fixed provider-tool contract from the assignment.

Design notes (the "pick a reasonable reading, write it down" calls):
- One ProviderAPI instance represents one authenticated user's session, so
  `user_id` is supplied at construction (mirroring how a real backend derives
  identity from auth rather than from a param on every call). `list_bookings`
  still takes `user_id` per the fixed contract and is validated against it.
- `book`/`cancel`/`reschedule` are idempotent on `idempotency_key`: replaying
  the same key returns the cached outcome instead of re-rolling random
  behaviour or re-mutating state. This is what makes "no duplicate bookings
  across retries" possible at all.
- `cancel`'s "first call fails, retry succeeds" is deterministic per
  booking_id (not randomised) per the spec's wording.
- Random outcomes (`book`'s 70/15/15 split, the hidden timeout resolution)
  come from a seeded `random.Random` for reproducibility, and can be forced
  per call via `ScenarioConfig` so evals don't depend on RNG luck.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

HOLD_TTL_SECONDS = 10 * 60
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVIDER_STATE_DIR = REPO_ROOT / "data" / "provider_backend"


def _to_tuples(value):
    """json round-trips tuples as lists; random.setstate() requires the
    exact nested-tuple shape random.getstate() produced, so convert back."""
    if isinstance(value, list):
        return tuple(_to_tuples(v) for v in value)
    return value


class ProviderAPIError(Exception):
    """Raised for malformed calls (unknown ids) -- not part of the scripted
    failure-behaviour contract, just defensive programming."""


class TransientError503(Exception):
    """Simulates a transient backend failure. The caller is expected to retry
    with the same idempotency_key."""


@dataclass
class Provider:
    id: str
    name: str
    service: str
    area: str
    price: int
    rating: float
    free_slots: list
    attributes: list

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "service": self.service,
            "area": self.area,
            "price": self.price,
            "rating": self.rating,
            "free_slots": list(self.free_slots),
            "attributes": list(self.attributes),
        }


@dataclass
class Booking:
    id: str
    user_id: str
    provider_id: str
    service: str
    slot: str
    price: int
    status: str  # "booked" | "cancelled"

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "service": self.service,
            "slot": self.slot,
            "price": self.price,
            "status": self.status,
        }


@dataclass
class Hold:
    id: str
    provider_id: str
    service: str
    slot: str
    quoted_price: int
    created_at: float
    expires_at: float
    resolved_booking_id: Optional[str] = None


@dataclass
class ScenarioConfig:
    """Knobs a test/eval can set to force specific mock behaviour instead of
    relying on the RNG. Every field is optional; unset fields fall back to
    seeded-random behaviour matching the assignment's fixed percentages.
    """
    # hold_slot overrides
    unavailable_holds: set = field(default_factory=set)       # {(provider_id, slot)}
    price_drift: dict = field(default_factory=dict)           # provider_id -> quoted_price override

    # book overrides, keyed by provider_id (not hold_id -- hold_id is generated
    # internally and unknown to the caller before hold_slot() runs, whereas
    # tests/evals always know which provider they're forcing an outcome for)
    forced_book_outcome: dict = field(default_factory=dict)   # provider_id -> "booked"|"slot_taken"|"timeout"
    forced_timeout_secret: dict = field(default_factory=dict)  # provider_id -> "booked"|"failed"

    # reschedule overrides
    slot_conflicts: set = field(default_factory=set)          # set of new_slot strings that always conflict

    inject_reviews: bool = True


class ProviderAPI:
    def __init__(self, user_id: str, seed: int = 42, scenario: Optional[ScenarioConfig] = None,
                 seed_history: bool = True):
        self.user_id = user_id
        self._rng = random.Random(seed)
        self.scenario = scenario or ScenarioConfig()

        self._providers: dict = {}          # provider_id -> Provider
        self._search_cache: dict = {}       # (service, area) -> [provider_id, ...]
        self._holds: dict = {}              # hold_id -> Hold
        self._bookings: dict = {}           # booking_id -> Booking
        self._idempotency_results: dict = {}  # idempotency_key -> last returned dict (book/cancel/reschedule, namespaced)
        self._idempotency_secret: dict = {}   # idempotency_key -> {"secret_status":..., "booking_id":...}
        self._cancel_attempts: dict = {}      # booking_id -> attempt count

        if seed_history:
            self._seed_history()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    def _make_provider(self, service: str, area: str) -> Provider:
        from .fixtures import PROVIDER_NAME_BANK, SERVICE_NAME_BANK, ATTRIBUTE_BANK
        pid = self._new_id("prov")
        name = self._rng.choice(SERVICE_NAME_BANK.get(service, PROVIDER_NAME_BANK))
        price = self._rng.randint(400, 3000)
        rating = round(self._rng.uniform(3.0, 5.0), 1)
        n_slots = self._rng.randint(2, 4)
        free_slots = sorted({self._random_slot_iso() for _ in range(n_slots)})
        n_attrs = self._rng.randint(0, 3)
        attributes = self._rng.sample(ATTRIBUTE_BANK, n_attrs) if n_attrs else []
        return Provider(pid, name, service, area, price, rating, free_slots, attributes)

    def _random_slot_iso(self) -> str:
        day_offset = self._rng.randint(0, 6)
        hour = self._rng.choice([8, 9, 10, 11, 14, 15, 16, 17])
        # Fixed reference "today" so output is stable across runs for a given seed.
        base = 1_900_000_000  # arbitrary fixed epoch anchor
        ts = base + day_offset * 86400 + hour * 3600
        return time.strftime("%Y-%m-%dT%H:%M:00", time.gmtime(ts))

    def _seed_history(self) -> None:
        """Pre-populate booking history so a fresh session has something to
        act on: one *completed* electrician job (for "book the same provider
        as last time") and one *upcoming* AC-servicing booking (so "Move my
        AC servicing to Saturday" -- the assignment's own example -- has an
        existing booking to reschedule/cancel without first creating one)."""
        past_provider = self._make_provider("electrician", "default")
        self._providers[past_provider.id] = past_provider
        past_booking = Booking(
            id=self._new_id("bkg"), user_id=self.user_id, provider_id=past_provider.id,
            service="electrician", slot=self._random_slot_iso(), price=past_provider.price,
            status="completed",
        )
        self._bookings[past_booking.id] = past_booking

        upcoming_provider = self._make_provider("ac_technician", "default")
        self._providers[upcoming_provider.id] = upcoming_provider
        upcoming_booking = Booking(
            id=self._new_id("bkg"), user_id=self.user_id, provider_id=upcoming_provider.id,
            service="ac_technician", slot=self._random_slot_iso(), price=upcoming_provider.price,
            status="booked",
        )
        self._bookings[upcoming_booking.id] = upcoming_booking

    def _get_provider_or_raise(self, provider_id: str) -> Provider:
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ProviderAPIError(f"unknown provider_id: {provider_id}")
        return provider

    # ------------------------------------------------------------------
    # fixed contract
    # ------------------------------------------------------------------
    def search_providers(self, service: str, area: str, date_range: tuple) -> list:
        key = (service, area)
        if key not in self._search_cache:
            count = self._rng.randint(3, 8)
            ids = []
            for _ in range(count):
                p = self._make_provider(service, area)
                self._providers[p.id] = p
                ids.append(p.id)
            self._search_cache[key] = ids
        return [self._providers[pid].to_public_dict() for pid in self._search_cache[key]]

    def get_provider(self, provider_id: str) -> dict:
        from .fixtures import CLEAN_REVIEWS, INJECTION_REVIEWS
        provider = self._get_provider_or_raise(provider_id)
        reviews = self._rng.sample(CLEAN_REVIEWS, min(3, len(CLEAN_REVIEWS)))
        if self.scenario.inject_reviews:
            reviews.insert(self._rng.randint(0, len(reviews)), self._rng.choice(INJECTION_REVIEWS))
        data = provider.to_public_dict()
        data["reviews"] = reviews
        return data

    def list_bookings(self, user_id: str) -> list:
        if user_id != self.user_id:
            raise ProviderAPIError(f"session is scoped to {self.user_id}, not {user_id}")
        return [b.to_public_dict() for b in self._bookings.values() if b.user_id == user_id]

    def hold_slot(self, provider_id: str, slot: str) -> dict:
        provider = self._get_provider_or_raise(provider_id)
        if (provider_id, slot) in self.scenario.unavailable_holds:
            return {"status": "slot_unavailable"}
        quoted_price = self.scenario.price_drift.get(provider_id, provider.price)
        hold_id = self._new_id("hold")
        now = time.time()
        hold = Hold(
            id=hold_id, provider_id=provider_id, service=provider.service, slot=slot,
            quoted_price=quoted_price, created_at=now, expires_at=now + HOLD_TTL_SECONDS,
        )
        self._holds[hold_id] = hold
        return {"status": "ok", "hold_id": hold_id, "expires_at": hold.expires_at, "quoted_price": quoted_price}

    def book(self, hold_id: str, idempotency_key: str) -> dict:
        ns_key = f"book:{idempotency_key}"
        if ns_key in self._idempotency_results:
            return self._idempotency_results[ns_key]

        hold = self._holds.get(hold_id)
        if hold is None:
            raise ProviderAPIError(f"unknown hold_id: {hold_id}")

        if time.time() > hold.expires_at:
            result = {"status": "hold_expired"}
            self._idempotency_results[ns_key] = result
            return result

        if hold.resolved_booking_id is not None:
            result = {"status": "slot_taken"}
            self._idempotency_results[ns_key] = result
            return result

        outcome = self.scenario.forced_book_outcome.get(hold.provider_id)
        if outcome is None:
            r = self._rng.random()
            if r < 0.70:
                outcome = "booked"
            elif r < 0.85:
                outcome = "slot_taken"
            else:
                outcome = "timeout"

        if outcome == "booked":
            booking_id = self._new_id("bkg")
            self._bookings[booking_id] = Booking(
                id=booking_id, user_id=self.user_id, provider_id=hold.provider_id,
                service=hold.service, slot=hold.slot, price=hold.quoted_price, status="booked",
            )
            hold.resolved_booking_id = booking_id
            result = {"status": "booked", "booking_id": booking_id}

        elif outcome == "slot_taken":
            result = {"status": "slot_taken"}

        else:  # timeout -- resolve a hidden secret outcome the caller cannot see yet
            secret = self.scenario.forced_timeout_secret.get(hold.provider_id)
            if secret is None:
                secret = "booked" if self._rng.random() < 0.5 else "failed"
            if secret == "booked":
                booking_id = self._new_id("bkg")
                self._bookings[booking_id] = Booking(
                    id=booking_id, user_id=self.user_id, provider_id=hold.provider_id,
                    service=hold.service, slot=hold.slot, price=hold.quoted_price, status="booked",
                )
                hold.resolved_booking_id = booking_id
                self._idempotency_secret[ns_key] = {"secret_status": "booked", "booking_id": booking_id}
            else:
                self._idempotency_secret[ns_key] = {"secret_status": "failed"}
            result = {"status": "timeout"}

        self._idempotency_results[ns_key] = result
        return result

    def get_booking_status(self, idempotency_key: str) -> dict:
        """Authoritative source of truth for a book() attempt. Must be
        consulted after any 'timeout' before retrying or reporting to the
        user."""
        ns_key = f"book:{idempotency_key}"
        result = self._idempotency_results.get(ns_key)
        if result is None:
            return {"status": "not_found"}
        if result["status"] == "booked":
            return {"status": "booked", "booking_id": result["booking_id"]}
        if result["status"] == "timeout":
            secret = self._idempotency_secret.get(ns_key, {"secret_status": "failed"})
            if secret["secret_status"] == "booked":
                return {"status": "booked", "booking_id": secret["booking_id"]}
            return {"status": "failed"}
        # slot_taken / hold_expired -> no booking ever existed under this key
        return {"status": "failed"}

    def cancel(self, booking_id: str, idempotency_key: str) -> dict:
        ns_key = f"cancel:{idempotency_key}"
        if ns_key in self._idempotency_results:
            return self._idempotency_results[ns_key]

        booking = self._bookings.get(booking_id)
        if booking is None:
            raise ProviderAPIError(f"unknown booking_id: {booking_id}")

        attempts = self._cancel_attempts.get(booking_id, 0)
        self._cancel_attempts[booking_id] = attempts + 1
        if attempts == 0:
            raise TransientError503(f"transient failure cancelling {booking_id}; retry")

        booking.status = "cancelled"
        result = {"status": "cancelled"}
        self._idempotency_results[ns_key] = result
        return result

    def reschedule(self, booking_id: str, new_slot: str, idempotency_key: str) -> dict:
        ns_key = f"reschedule:{idempotency_key}"
        if ns_key in self._idempotency_results:
            return self._idempotency_results[ns_key]

        booking = self._bookings.get(booking_id)
        if booking is None:
            raise ProviderAPIError(f"unknown booking_id: {booking_id}")

        if new_slot in self.scenario.slot_conflicts:
            result = {"status": "slot_conflict"}
            self._idempotency_results[ns_key] = result
            return result

        booking.slot = new_slot
        result = {"status": "updated", "booking": booking.to_public_dict()}
        self._idempotency_results[ns_key] = result
        return result

    # ------------------------------------------------------------------
    # persistence -- models the provider backend as a service that keeps its
    # own state independently of the chat client. This is what makes the
    # "restart mid-booking" guarantee demonstrable against a real-feeling
    # backend instead of only inside the eval harness: if the CLI process is
    # killed and restarted, the backend still knows about any hold/booking
    # already created, so get_booking_status(idempotency_key) keeps working.
    # ------------------------------------------------------------------
    def to_snapshot(self) -> dict:
        return {
            "user_id": self.user_id,
            "rng_state": self._rng.getstate(),
            "providers": {pid: asdict(p) for pid, p in self._providers.items()},
            "search_cache": {f"{k[0]}||{k[1]}": v for k, v in self._search_cache.items()},
            "holds": {hid: asdict(h) for hid, h in self._holds.items()},
            "bookings": {bid: asdict(b) for bid, b in self._bookings.items()},
            "idempotency_results": self._idempotency_results,
            "idempotency_secret": self._idempotency_secret,
            "cancel_attempts": self._cancel_attempts,
        }

    def load_snapshot(self, snapshot: dict) -> None:
        self._rng.setstate(_to_tuples(snapshot["rng_state"]))
        self._providers = {pid: Provider(**p) for pid, p in snapshot["providers"].items()}
        self._search_cache = {tuple(k.split("||", 1)): v for k, v in snapshot["search_cache"].items()}
        self._holds = {hid: Hold(**h) for hid, h in snapshot["holds"].items()}
        self._bookings = {bid: Booking(**b) for bid, b in snapshot["bookings"].items()}
        self._idempotency_results = snapshot["idempotency_results"]
        self._idempotency_secret = snapshot["idempotency_secret"]
        self._cancel_attempts = snapshot["cancel_attempts"]

    def save(self, data_dir: Optional[Path] = None) -> None:
        d = Path(data_dir) if data_dir else DEFAULT_PROVIDER_STATE_DIR
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.user_id}.json"
        path.write_text(json.dumps(self.to_snapshot(), indent=2))

    @classmethod
    def load_or_create(cls, user_id: str, seed: int = 42, scenario: Optional[ScenarioConfig] = None,
                        data_dir: Optional[Path] = None) -> "ProviderAPI":
        d = Path(data_dir) if data_dir else DEFAULT_PROVIDER_STATE_DIR
        path = d / f"{user_id}.json"
        api = cls(user_id=user_id, seed=seed, scenario=scenario, seed_history=not path.exists())
        if path.exists():
            api.load_snapshot(json.loads(path.read_text()))
        return api
