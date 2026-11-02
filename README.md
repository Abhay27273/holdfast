# Home-Services Booking Agent

A chat agent that books, reschedules, and cancels home-service errands
against a mocked provider API -- built so it never double-books, never acts
without approval, and never gets talked into anything by a provider review.

See [DESIGN_NOTE.md](DESIGN_NOTE.md) for the architecture, [BUGS.md](BUGS.md)
for three real bugs hit while building this, and [eval_table.md](eval_table.md)
for the generated results of the 20 scripted eval scenarios.

## Run it (3 commands)

```bash
pip install -r requirements.txt
pytest && python -m evals.runner
python -m agent.cli
```

The first command installs dependencies (only `pytest` is required; `anthropic`
is only needed if you opt into the optional live-LLM adapter -- see below).
The second runs the full unit test suite and the 20-scenario eval harness,
regenerating `eval_table.md` and `eval_results.json`. The third starts an
interactive chat session.

Try the assignment's own example requests:

```
you> Find a plumber for my kitchen tap this week, mornings only, under ₹1500.
you> Move my AC servicing to Saturday.
you> Book the same electrician as last time, but not before 10 a.m.
```

State, memory, and the mock provider "backend" persist under `./data/` per
`--user-id` (default `demo`), so you can genuinely `Ctrl+C` mid-booking and
restart the CLI to see the recovery guarantee for yourself:

```bash
python -m agent.cli --user-id demo --reset   # wipe this user's saved state first
```

## Project layout

```
agent/
  mocks/provider_api.py   # the fixed tool contract (search_providers, hold_slot, book, ...)
  core/                     # state machine, approval records, idempotency, memory, sanitization, limits
  nlu/                      # pluggable "understanding" models (rule-based default + optional Anthropic)
  orchestrator/             # the deterministic turn loop that owns all tool calls and approval checks
  trace.py                  # one JSONL trace file per run
  cli.py                    # interactive REPL
evals/
  scenarios.py              # 20 scripted scenarios covering every must-have
  runner.py                 # runs them all, generates eval_table.md / eval_results.json
tests/                      # unit + integration tests for every mechanism above
traces/sample_timeout_after_success.jsonl   # committed end-to-end trace (see DESIGN_NOTE.md)
```

## Optional: live LLM

By default the agent's "brain" is a deterministic rule-based parser
(`agent/nlu/rule_model.py`) -- no API key needed, fully reproducible, and
exactly what the eval suite runs against. An optional Anthropic adapter
implements the identical `Model` protocol:

```bash
pip install -r requirements-optional.txt   # only needed for this path
```

```python
from agent.nlu.anthropic_model import AnthropicModel
from agent.orchestrator.agent_loop import Agent

agent = Agent("alice", model=AnthropicModel())  # requires ANTHROPIC_API_KEY
```

Swapping models never changes approval, idempotency, injection handling, or
limits -- those live entirely in the orchestrator.

## Memory commands (in the CLI, plain English)

- `what do you remember about me` -- lists saved preferences and whether
  each is user-stated or agent-inferred (and confirmed or not)
- `forget my area` -- deletes a saved preference permanently (it will not be
  silently re-inferred later)
- `remember my technician` -- confirms an inferred preference so it starts
  being used
