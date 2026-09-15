# Verification record — 2026-09-14

This record covers the infrastructure and approved G0–G1 workshop checks. It does
not establish a successful inheritance study or completed G0–G8 research program.

## Checks performed

- Python 3.12.14, Windows, isolated virtual environment.
- `python main.py check`: 138 tests passed.
- `python -m mypy src/gim`: strict typing passed for 24 source files.
- `python -m ruff check src tests benchmarks main.py`: passed.
- `python -m ruff format --check src tests benchmarks main.py`: 35 files formatted.
- Editable installation with pinned setuptools 80.9.0 succeeded; installed
  `gim status` ran successfully.

No third-party runtime libraries, downloaded model weights or API keys are
needed for these checks. The development-tool versions are pinned in
`requirements-dev.txt`. Commands are documented in the README.

## What the tests establish

| Property | Evidence |
| --- | --- |
| G0 approved workshop | All 4,096 mappings checked for success/failure boundaries and evaluator replay; arbitrary task order, fifth-attempt success, reset, invalid actions and forged evidence covered. The full demo reproduces across fresh processes with different Python hash seeds. |
| G1 workshop policy | Fresh real policies release before successors are constructed; episode observations and conflict diagnostics are instance-local. Policies receive only their draw seed, public observation and current retrieval values. Hidden-rule variations with identical public evidence produce identical public observations. |
| Option-2 extraction | Successful-source warnings and failed-source working rules retain source-based quotas and actual source reward; every candidate cites its observed event. Coverage can protect a successful-source warning. Public sequence contradictions, duplicate JSON fields, truncation and impossible tool feedback are rejected. IG remains unavailable. |
| Frozen evidence and memory | Mutable input containers are detached; event IDs, terminal summaries, finite values and nested immutable records are checked. |
| First decision context | The actual reset observation is preserved in the frozen trajectory and audit records; later decision inputs are reconstructed from previous events. |
| Terminal extraction | Distillation rejects truncations and nonexistent citations; the integration fixture invokes one terminal batch per generation. |
| Retrieval | Exact cosine comparisons, deterministic ties, separate polarity quotas, scope/type filters, stale-cache rejection, revision/dimension validation even with empty quotas. |
| Actual representation cost | Binary store round-trips match measured file bytes; each 384-dimensional vector occupies exactly 1,536 bytes. The charged ID/revision/dimension mapping shares immutable authoritative coordinate tuples, without another persisted vector block. |
| Float32 and format integrity | Values are rounded before insertion and preserved across save/load with stable retrieval; invalid/truncated framing, nonfinite coordinates, duplicate items and legacy JSON are rejected. |
| Persistence failures | Failed atomic replacement retains the previous store and cleans its temporary file. |
| Generation boundary | A successor reads serialized memories; the previous fixture policy is collectible. No-memory bypasses embedding, distillation and serialization. |
| Inherited budget | Oversized inherited memory is rejected before the policy/environment factories run, not merely pruned after it has influenced actions. |
| Canonical input and file measurement | Padded input is rejected before acting; CLI inspection counts actual saved bytes and reports canonical size separately. |
| Boundary compaction | FIFO, seeded-random reproducibility, index overhead, smallest coverage witness, infeasible coverage with unchanged inputs. |
| Audit records | Invalid JSON values do not append a partial record; failed coverage preserves trajectory evidence without logging a valid boundary. |
| Measurements | Paired differences respect matching; zero-byte efficiency is undefined; malformed/empty measurements are rejected. |
| Online costs and reproducibility | A controlled clock proves that setup/retrieval/write costs exclude agent, environment and audit I/O; logs record adapter revisions, quotas, pruning inputs, memory hashes, inserted records, and failed-boundary peak bytes. Offline labeling remains unavailable and is never added to online cost. |

## What remains unproven or unimplemented

- The workshop task, frozen policy and option-2 terminal evidence extraction are
  implemented. The IG estimator and research embedder remain unfinished G2 work.
- Trusted Python adapters can retain hidden global state if written incorrectly.
  Interface tests are not process isolation or a hostile-code sandbox.
- G4 merge/split remains unimplemented. The revised design preserves one float32
  centroid; the historical evidence and partition rule for Split await agreement.
- G5 inheritance has no approved statistical decision or empirical result.
- G6 utility labels, learned predictor, H1 comparison and H3 feature importance
  remain unimplemented in accordance with the preceding gates.
- No access-recency, raw-trajectory, reflection, success-only, utility-aware,
  quantization, symbolic or ANN experiment has been reported as completed.
- G7/G8 scaling and LLM transfer are later work.
- The serializer validates the sketch's 4d vector cost for basic items. It does
  not establish prototype quality, runtime RAM savings, or a complete prototype's
  byte cost. Scalar prototype statistics and consolidation remain unimplemented.
- The selected coverage family list, scientific budget sweep, model choice,
  utility score and sample-size/significance rules still need specification.

The separate [storage benchmark](09-storage-benchmark.md) compares 200 identical
items and ten queries with the preceding reference implementation: 83.37% less
charged storage and 1.87 times faster retrieval in the recorded local run, with
the same selected IDs and ordering. This is a mechanics measurement, not a
transfer experiment or a token-cost result.

The [workshop report](10-workshop-g0-g1.md) records the approved G0–G1 implementation
and a deterministic 12-episode no-memory demo. Its 2 successes are a reproducible
example, not an inheritance comparison or a powered estimate of performance.

The objective remains open. Tests and code presence do not substitute for these
research requirements or authorize a change to the original plan.
