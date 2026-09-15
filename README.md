# GIM: Geometric Intergenerational Memory

The central research question, adopted on 2026-09-14, is:

> Under an audited persistent-byte cap, does utility-aware retention improve successor-agent performance, and do local evidence and information gain predict that utility better than source-episode reward?

The revised working hypothesis is that utility-aware retention improves successor
performance under that cap, and that local evidence and information gain predict
memory utility better than source-episode reward. **This hypothesis is untested.**
H1 and H3 are retained as labels for those two parts, not as a replacement for the
revised research question. Follow the architecture's G0–G8 build order, compare
policies at the same declared cap, and report actual occupied bytes separately.

The deployment priorities are useful transfer with less persistent storage,
fewer model tokens, and lower latency. Compare these costs at measured task
quality; a smaller record alone does not establish a better learning system.
Experience reinforces retained memories under the architecture's frozen agent
weights. Training the agent's weights is outside this memory design.

This repository has executable Python infrastructure and the approved workshop
environment with a frozen policy. G0–G1 mechanics are verified for this adapter.
The G4 prototype Split design remains proposed. **No inheritance result or H1/H3
finding is claimed.**

Approved option 2 keeps memories grouped by source episode outcome and uses the
existing local `works` fact separately for H3. The terminal workshop distiller
implements this separation without adding another vector. See the
[source outcome and local evidence decision](docs/11-source-outcome-and-local-evidence.md).

## Open in Visual Studio

Open `gim-agent-memory.sln` in Visual Studio with Python development support.
The project selects the existing `.venv` interpreter, includes `src` in the search
path, and runs the workshop demo with **Ctrl+F5**. Its Python context menu includes
the workshop, all tests, status and storage benchmark. See the
[Visual Studio guide](docs/12-visual-studio.md).

## Run the checks

Use Python 3.11 or newer. There are **no third-party runtime dependencies**.
The code was verified with Python 3.12.14 on Windows.

```powershell
cd path\to\gim-agent-memory
python -m venv .venv
.\.venv\Scripts\python.exe main.py check
.\.venv\Scripts\python.exe main.py status
```

If your Python launcher is `py`, use `py -3 -m venv .venv` for the first command.
Activation is optional: using the environment's executable directly is explicit
and avoids PowerShell activation-policy issues. The checkout launcher makes the
`src/gim` package available without downloading or installing anything.

`check` runs the unit and integration fixtures. Those fixtures exercise the
architecture's mechanics; their scores are not a Phase A inheritance result.

Run the approved workshop with a reproducible task schedule and fresh agents:

```powershell
.\.venv\Scripts\python.exe main.py workshop-demo --seed 2026 --workshops 3 --episodes 12 --output runs/workshop-g0-g1-demo.json
```

Each agent tries to open three locks with four available tools within five
attempts. The demo uses no inherited memory. Its JSON records seeds, hidden task
definitions, public trajectories and evaluator checks for review; agents never
receive this report. See the [workshop guide](docs/10-workshop-g0-g1.md).

To inspect a store written with `MemoryStore.save`:

```powershell
.\.venv\Scripts\python.exe main.py inspect 'path\to\memory.gim'
```

For editable package installation and optional development checks:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m mypy src/gim
.\.venv\Scripts\python.exe -m ruff check src tests benchmarks main.py
.\.venv\Scripts\python.exe -m ruff format --check src tests benchmarks main.py
```

Only these optional development steps require package downloads. Run `gim status`
after installation; `gim check` is intended for an editable source checkout.

## How the pieces connect

`run_generation` reconstructs declared memory from bytes, checks its inherited
budget, and creates a fresh environment and agent policy from caller-supplied
factories. Before each action, retrieval reads a detached immutable snapshot with
independent positive and negative quotas. The agent cannot call the store writer.

After termination, the runner freezes the trajectory, calls the distiller once,
checks every cited event ID, normalizes embeddings from the explicitly supplied
encoder, and inserts the candidates. FIFO compaction then measures the complete
store plus index representation and enforces the boundary cap. It returns new
memory bytes only if the boundary is valid. The caller can persist that result
atomically; the input bytes are unchanged if any step fails.

The separate JSON Lines logger records trajectories, retrievals, writes,
evictions and boundary measurements. Agents never receive the logger. A returned
`boundary_ready` record means the caller may commit memory; it does not claim that
a file commit already happened. Retrying across crashes still needs an explicit
experiment commit protocol.

Each run records the configured policy, environment, distiller and embedding
revisions, retrieval quotas, pruning seed/coverage settings, and input/output
memory SHA-256 hashes. The experiment must archive the matching source, model
artifacts and task schedule; a revision label cannot recover a missing artifact.

`GenerationResult.online_cost` reports setup, retrieval (including query
embedding), and terminal write-path durations in nanoseconds. The write path
includes distillation, embedding, insertion, compaction and serialization.
Agent/environment execution and audit-file I/O are outside these intervals.
Offline labeling cost is unavailable (`null`) because G6 does not run here; it
is never added to online cost. Timings naturally vary, while seeds, decisions
and canonical memory bytes remain reproducible.

## What is implemented

| Files | Responsibility |
| --- | --- |
| `src/gim/trajectory.py`, `environment.py`, `agent.py`, `distiller.py` | Immutable event evidence, pluggable task/policy interfaces, fresh one-episode agent, citation validation |
| `src/gim/workshop_contract.py`, `workshop_environment.py`, `workshop_policy.py`, `workshop_demo.py` | Approved deterministic workshop, separate ground-truth evaluator, frozen policy, reproducible G0–G1 demo |
| `src/gim/workshop_distiller.py` | Terminal per-event evidence, source episode labels, local `works` facts and failure tags; IG remains unavailable |
| `src/gim/memory_item.py`, `memory_store.py`, `embedder.py` | Typed memories, immutable provenance, deterministic storage, atomic save/load, explicit pinned encoder interface |
| `src/gim/vector_index.py`, `retrieval_policy.py` | Exact cosine search, stale-cache checks, separate polarity quotas |
| `src/gim/consolidation_policy.py`, `pruning_policy.py` | Insert-only writer, FIFO and seeded-random pruning with explicit coverage inputs |
| `src/gim/generation_runner.py`, `experiment_logger.py` | Terminal pipeline, inherited/boundary budget checks, isolated experiment records |
| `src/gim/experiment_runner.py`, `config.py`, `cli.py` | Descriptive paired comparisons, success-per-byte, explicit settings and offline commands |
| `tests/test_*.py` | Reproducible mechanics and failure-path checks |

The versioned binary store (`GMS2`) writes canonical UTF-8 JSON metadata and one
little-endian float32 vector per item: exactly **1,536 vector bytes at 384
dimensions**. The rebuildable index shares the authoritative coordinate tuples
and charges its ID/revision/dimension mapping without duplicating vector payloads.
Metadata, provenance, framing and that mapping also count toward B. These are
representation bytes; Python process RAM is a separate measurement. A zero-byte
no-memory control bypasses store and index serialization entirely. G4 prototypes
and their scalar statistics are still proposed, not implemented.

Old JSON snapshots require an explicit migration and are rejected with a clear
format error. No prior snapshots were found in this project. The current code
does not supply a migration tool or silently reinterpret older memory costs.

Inherited bytes must be the canonical bytes emitted by `MemoryStore`. The runner
rejects padded or reformatted input before acting, so normalization cannot hide
extra disk bytes from B. `inspect` reports actual file bytes, vector bytes,
metadata/framing bytes, canonical size, and whether the input is canonical.

The [storage benchmark](docs/09-storage-benchmark.md) reports the measured byte
and retrieval changes on a deterministic fixture. It does not measure transferred
task performance or LLM tokens.

Provenance remains immutable while its memory exists and is removed with that
memory, as approved. Coverage names must be supplied explicitly; the implemented
rule protects one smallest record per named failure family. Scientific choices
about which families count as rare and their quotas remain open. Seeded random
is available for mechanics checks; it is not a completed H1 comparison.

Python callbacks are trusted adapters, not a security sandbox. Factories must not
return predecessor state, and a concrete model needs its own reset-isolation
audit. The tests verify the supplied infrastructure and fixtures only.

## Research documents and next review

The hypotheses document records the adopted central research question, its
aligned working abstract, and the earlier approved H3 option-2 amendment. The
architecture and bibliography retain their existing contents:

- [Abstract and hypotheses](docs/00-abstract-and-hypotheses.md)
- [Architecture and build gates](docs/03-architecture-diagram.md)
- [Bibliography](docs/04-bibliography.md)

New implementation notes:

- [Implementation, research gaps, and C++ suitability](docs/05-implementation-and-research.md)
- [Approved Phase A environment and remaining evidence proposals](docs/06-proposed-phase-a-and-split.md)
- [Centroid storage and the remaining Split decision](docs/08-centroid-storage-design.md)
- [Verification record](docs/07-verification.md)
- [Storage and exact-retrieval benchmark](docs/09-storage-benchmark.md)
- [Run and inspect the approved G0–G1 workshop](docs/10-workshop-g0-g1.md)
- [Approved option 2: source outcome and local evidence](docs/11-source-outcome-and-local-evidence.md)

The architecture governs G5 inheritance, followed by G6 utility for H1/H3. G4
merge/split, G5 evaluation and G6 utility are not implemented or passed. The
utility files remain clearly marked planning notes. G7 ANN and G8 LLM transfer
follow later. Nothing substitutes a fabricated scientific result for a gate.

## Performance iteration suite

Run `python main.py check`, then
`python benchmarks/iteration_suite.py --label baseline --profile`.
After an implementation change, use a new label and
`--compare runs/iterations/baseline/summary.json` to check replay, success,
stored bytes, and timing. Visual Studio also exposes **GIM - Iteration suite**.
The six-arm workshop engineering study and C++ candidate review are documented
in [Performance and ablation](docs/13-performance-and-ablation.md).
This exercises insert-only memory and FIFO/random budgets, not formal H1/H3 gates.

## Current evidence and research status

The engineering workshop study provides preliminary evidence that inherited memory
can help: FIFO at 32 KiB completed 39/72 tasks, compared with 17/72 without memory.
These are 24 tasks per lineage across only three independent seeds, using a fixed
task-specific encoder. They are not 72 independent statistical replicates or a
formal inheritance-gate result.

Success-only storage completed 38/72 tasks while occupying fewer bytes on average.
This is a comparison condition, not a refutation of the revised hypothesis: that
hypothesis does not require failed episodes to be inherently more valuable.
Utility-aware retention, counterfactual utility labels, and measured information
gain have not yet been evaluated. The speed and allocation improvements establish
engineering progress, not evidence for the revised hypothesis or deployed LLM
token savings. See the [protocol and limitations](docs/13-performance-and-ablation.md).
