# Approved workshop: deterministic tasks and fresh agents

The user approved this environment and frozen policy for G0–G1 on 2026-09-14.
The Python implementation now runs in the CLion project. The original architecture
and hypothesis documents remain unchanged.

## Run it

From the project terminal:

```powershell
.\.venv\Scripts\python.exe main.py workshop-demo --seed 2026 --workshops 3 --episodes 12 --output runs/workshop-g0-g1-demo.json
.\.venv\Scripts\python.exe main.py check
```

Omit `--output` to print the complete JSON. The demo uses no model, API key or
third-party runtime dependency. It creates a new policy and agent per episode.
No inherited memories or write path are connected in this G0–G1 command.

## Environment and policy

Each workshop has six lock types and four tools. Every lock type has exactly one
working tool, fixed across that workshop's generations. An episode shows three
distinct locks in sequence; opening all three within five attempts earns reward
1. Otherwise the fifth attempt ends with reward 0. Opening the final lock on the
fifth attempt succeeds. Invalid tool names are errors and consume no attempts.

The policy sees only the workshop ID, current lock, four tool names, attempts
remaining and last tool feedback. It rules out tools disproven during this episode
and draws among the remaining tools. The action rule stays fixed. Its observations
and conflict diagnostics disappear with the policy; no weights are updated.

Hidden rules and future locks are experiment-side values. The separate evaluator
replays every recorded action and rejects inconsistent observations, rewards or
termination. It reports opened locks, success, wrong-tool counts and failure
triples. Repeated-failure counts here are within each episode; cross-generation
recurrence is a later analysis, not silently inferred from these counts.

## Reproducibility and boundary checks

- Versioned SHA-256 counter draws use integer rejection to avoid modulo bias.
  Candidate tools have stable ordering. Python hash/set order cannot change actions.
- Workshop mappings are sampled without replacement from all 4,096 possibilities.
  IDs label sample positions and do not encode the selected mapping. A study must
  keep its mapping manifest with its scoped memories; labels are local to that study.
- The demo fixes a round-robin family schedule before any episode runs. Separate
  namespaces derive mapping, task and policy seeds. Only its own policy seed is
  supplied to the policy. Changing outcomes cannot change later tasks.
- Tests enumerate all mappings for transition boundaries and evaluator replay.
  Separate Python processes with different hash seeds emit the same complete demo.
- Tests exercise the real policy's local beliefs, retrieved-evidence conflict
  fallback and reuse rejection. Weak-reference checks establish that each policy
  is released before its successor is constructed. The no-memory retrieval
  function captures no store, logger, evaluator or previous agent.

These checks establish G0–G1 behavior for the supplied trusted Python adapter.
They do not make arbitrary Python callbacks a security sandbox. A future LLM or
native runtime needs its own reset/isolation verification.

## Recorded demonstration

Seed 2026, three workshops and twelve episodes produced **2 successes out of 12**
and 60 attempts. The report includes the complete fixed schedule, mapping manifest,
public trajectories, evaluator labels, revision strings and Python version.
It is an audit artifact; the policy never receives it.

Persistent inherited memory is 0 bytes and model-token use is 0 because this policy
makes no model calls. Neither number measures deployed LLM savings. The small
no-memory run is a replayable example, not a powered success-rate estimate or
evidence that memory transfers. G5 still requires the actual inheritance comparison.

## Interface prepared for G2

The policy can consume already-retrieved `MemoryItem` values with exact workshop
scope and type `workshop-tool-rule-v1`. Their content is canonical JSON with three
fields, for example:

```json
{"lock_type":"triangle","tool":"blue","works":true}
```

This describes a local observation. The policy reads `works` from content and
does not reinterpret the existing `outcome_class` field. Compatible positive facts
select the working tool; negative facts eliminate a tool. Contradictory inherited
facts are ignored as a batch for that decision and logged as local diagnostics.
Only direct episode observations survive into the next decision; inherited facts
must be retrieved again. The current demo supplies no such records; tests use
explicit fixtures, without a hidden or unbudgeted transfer channel.

The user subsequently approved option 2. A terminal workshop distiller now keeps
source episode outcome and the local `works` fact separate, as described in
[decision 11](11-source-outcome-and-local-evidence.md). This G0–G1 demo still uses
no inherited memory. The IG estimator and research embedder remain unfinished;
G4 Split and later utility comparisons retain their existing review requirements.
