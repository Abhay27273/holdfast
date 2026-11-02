"""One JSONL trace file per run (must-have #7): state transitions, model
calls, tool calls with sanitised args/results, latency, and (when available)
tokens/cost. Secrets are redacted defensively even though the mocks never
carry real credentials.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = REPO_ROOT / "traces"

SECRET_KEYS = {"api_key", "apikey", "authorization", "token", "password", "secret"}


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("[redacted]" if k.lower() in SECRET_KEYS else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


class TraceWriter:
    def __init__(self, run_id: str, trace_dir: Optional[Path] = None):
        self.run_id = run_id
        self.dir = Path(trace_dir) if trace_dir else DEFAULT_TRACE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"
        self._start = time.monotonic()

    def log(self, event_type: str, **fields) -> dict:
        entry = {
            "ts": time.time(),
            "elapsed_s": round(time.monotonic() - self._start, 4),
            "type": event_type,
        }
        entry.update(_redact(fields))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def tool_call(self, name: str, args: dict, result=None, error: str | None = None,
                  latency_ms: float | None = None) -> dict:
        return self.log(
            "tool_call", name=name, args=args, result=result, error=error, latency_ms=latency_ms,
        )

    def model_call(self, model_name: str, input_text: str, output: dict,
                    tokens: Optional[int] = None, cost_usd: Optional[float] = None) -> dict:
        return self.log(
            "model_call", model=model_name, input=input_text, output=output,
            tokens=tokens, cost_usd=cost_usd,
        )
