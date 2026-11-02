"""Interactive REPL on top of the orchestrator. Run with:

    python -m agent.cli --user-id demo

State, memory, and the mock provider "backend" all persist to ./data under
the repo root between runs for the same --user-id, so you can genuinely kill
(Ctrl+C) and restart the process mid-booking to see the recovery guarantee
for yourself -- not just in the eval harness.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from agent.orchestrator.agent_loop import Agent

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reset_user_data(user_id: str) -> None:
    for sub in ("sessions", "memory", "provider_backend"):
        path = REPO_ROOT / "data" / sub / f"{user_id}.json"
        if path.exists():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Home-services booking agent CLI")
    parser.add_argument("--user-id", default="demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true", help="wipe this user's saved state/memory/backend first")
    args = parser.parse_args()

    if args.reset:
        _reset_user_data(args.user_id)

    agent = Agent(args.user_id, seed=args.seed)

    print("Home-services booking agent. Type 'exit' to quit.")
    print("Try: 'Find a plumber for my kitchen tap this week, mornings only, under ₹1500.'")
    print(f"(trace: {agent.trace.path})")
    print()

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        reply = agent.handle_message(text)
        print(f"agent> {reply}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
