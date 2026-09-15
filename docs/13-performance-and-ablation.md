# Performance, memory efficiency, and iteration testing

The acceptance requirement is **preserve task success**. A speed or storage win
does not pass if it loses task success. For the current internal optimization,
require identical trajectory and memory hashes for every matched lineage as well.

## Concrete agent goal

A fresh agent must open three distinct locks using four available tools within
five attempts. Workshop families have fixed hidden tool mappings. The public
observation reveals the current lock and action feedback, never the answer map.
Each episode constructs a new agent and environment. Only serialized memory is
inherited, with provenance retained for the life of each memory.

`benchmarks/iteration_suite.py` exercises the actual `run_generation`, workshop
policy, evidence distiller, scoped retrieval, store, and boundary compactor.
Schedules and policy seeds are fixed before execution; no evaluator state or
audit log is supplied to the successor. Every trajectory is independently replayed.

The engineering encoder maps public lock identity to a unit vector in 384
dimensions. It does not encode tools, rewards, or hidden answers. This deliberately
simple fixture isolates memory plumbing; it is **not a selected semantic research
encoder**. It persists one float32 vector (1,536 bytes) per item. Insert-only
consolidation, missing IG (`None`), and no protected rare-family list are explicit.
G4 prototypes/Split, formal G5 inheritance validation, H1/H3 utility experiments,
ANN, and LLM transfer remain outside this suite. This respects the build order in
`03-architecture-diagram.md`; engineering controls do not certify later gates.

## Ablation matrix

| Arm | Difference | Interpretation |
|---|---|---|
| `no_memory` | Zero bytes, no reads or writes | Learning-free task control |
| `reference` | All extracted observations retained; cap asserted never to prune | Diagnostic unrestricted-retention reference |
| `fifo` | Whole-record FIFO eviction at B | Budget baseline |
| `random` | Seeded random eviction at the same B | Retention-policy ablation against FIFO |
| `no_read` | Same FIFO writes, both retrieval quotas zero | Reading ablation; must replay no-memory actions exactly |
| `success_only` | Extract only from source episodes that succeeded, then FIFO | Source-outcome write ablation; not a utility predictor |

The default caps are 8,192 and 32,768 bytes, with quotas 5/5 except `no_read`.
Three seeds (2026–2028), three workshop families, 24 generations, and three timing
repeats produce 90 fresh worker processes. Workers run serially in a fixed shuffled
order, with two untimed no-memory warmup episodes. Each measured lineage starts
empty, so cold memory initialization is intentionally included. These are small
engineering runs, not a powered statistical study or a tuned evaluation set.

Equal cap does not imply equal occupancy. Matched tasks do not imply equal source
candidates: different actions create different evidence. Comparisons with the
unrestricted reference deliberately use different caps. Treat whole lineages as
independent units; neither successive episodes nor timing repeats increase the
statistical sample size. Report per-seed success before aggregating.

## Run and track an iteration

From the project directory in PowerShell:

```powershell
& .\.venv\Scripts\python.exe main.py check
& .\.venv\Scripts\python.exe benchmarks\iteration_suite.py --label baseline --profile
# Make one implementation change, then use a new label:
& .\.venv\Scripts\python.exe benchmarks\iteration_suite.py --label candidate --compare runs\iterations\baseline\summary.json --profile
```

Run the same command again with a different label for a new baseline. Existing
iteration directories cannot be overwritten. `--seeds`, `--budgets`, `--episodes`,
`--workshops`, and `--repeats` are configurable; changing them starts a different
protocol and makes the built-in iteration comparator reject the pairing.
Different machines/Python versions are also rejected for timing comparison.

Visual Studio includes the new files and a **GIM - Iteration suite** Python command
with a timestamped label. The existing **GIM - All tests** command discovers the
new tests. The project continues to use its existing Python interpreter.

Each `runs/iterations/<label>` contains:

- `manifest.json`: protocol, Python/machine, SHA-256 source inventory, completion status.
- `job-order.json`: exact serial worker schedule.
- `worker-*.json`: episode metrics, frozen evidence, retrieval/eviction audit, replay hashes.
- `lineages.csv`: per-lineage results, including all repeat times.
- `summary.json`: medians, paired ablation contrasts, and optional iteration comparison.
- `report.md`: readable outcome/storage/time table and optional iteration deltas.
- `profile.txt` and `profile.pstats` when requested, from a separate profiling pass.

A completed run means its measurements finished, not that an optimization passed.
Inspect `comparison[].accepted_semantics_preserving`: all entries must be true for
this optimization category. A representation change can legitimately alter memory
hashes, but then requires a separate declared equivalence test; this strict check
must not be silently relaxed. Timings are descriptive, with no fragile millisecond
thresholds in unit tests. Repeat an A/B/A sequence before a deployment performance
claim to check machine drift and background load.

## Measurement definitions

| Field | What is included |
|---|---|
| Success / attempts | Three locks opened within five attempts / number of actions |
| `generation_ns` | Full generation call: setup, agent/environment, reads, write/compaction, in-memory audit buffering |
| `successful_generation_ms` | Mean generation latency for successful tasks only, then median across repeats; denominator is `successes` |
| `setup_ns`, `retrieval_ns`, `write_ns` | Existing runner management phases; not summed with offline labeling |
| `store_bytes` | Actual serialized bytes, including provenance, text, metadata, framing and vectors |
| `index_bytes` | Derived canonical index overhead, charged to the cap |
| `boundary_bytes` | Store plus index after compaction, independently audited and required <= B |
| `peak_bytes` | Persistent representation size before boundary compaction; may exceed B by design |
| `mean_bytes_saved` | Control mean boundary bytes minus candidate mean boundary bytes, per matched lineage |
| `speedup` | Control complete-lineage time divided by candidate time; interpret only alongside success |

Task generation, evaluator replay, evidence serialization to disk, process launch,
profiling, and report generation are outside `generation_ns`. Audit data is separate
and never agent-readable. It does consume experiment disk/RAM, which is not hidden
inside the adaptive-store budget. No model calls occur, so this suite establishes
**no deployed LLM token savings**. Persistent bytes are not total process RAM.

For dense-vector compaction and temporary allocation measurement:

```powershell
& .\.venv\Scripts\python.exe benchmarks\benchmark_compaction.py --output runs\compaction-current.json
# Optional: use an older checkout's src in a separate process:
& .\.venv\Scripts\python.exe benchmarks\benchmark_compaction.py --source C:\path\to\older\src --output runs\compaction-older.json
```

This 200-record, 384D dense-vector fixture uses seven unprofiled timing passes and
a separate `tracemalloc` pass. Its peak covers Python allocations during compaction,
excluding the preexisting store/items/index; it is not whole-process RSS or the
native heap of a future C++ backend. See the [Python allocation tracing documentation](https://docs.python.org/3.12/library/tracemalloc.html).

## Code review: C++ candidates, in order

The measured first change stays in Python: `PruningPolicy.enforce` now constructs
an independent planning container sharing immutable records, rather than serializing
and decoding every record. Deleting from that container cannot mutate the original
store or its frozen records. Persistence decoding and validation are unchanged.

| Priority | Candidate | Reason and required boundary |
|---|---|---|
| 1 | Batched vector validation / float32 decode in `embedder.py` and `MemoryStore.from_bytes` | The baseline profile is dominated by repeated Python coordinate checks. First remove redundant work; a native batch must preserve finite/type/norm checks, binary32 rounding, malformed-input rejection, and canonical bytes. |
| 2 | Exact cosine scan and top-k in `VectorIndex.search` | Python scans every coordinate and sorts hits. A contiguous float32 buffer and batched native call may help at larger N. Preserve scope and outcome filtering, true norm handling, stable ID tie-breaking, and deterministic retrieval; benchmark near-ties as well as this simple fixture. |
| 3 | Serialization / compaction size accounting | Repeated whole-store serialization remains expensive under eviction pressure. First consider exact per-record costs and Python container reuse; native serialization must reproduce the same format and audit totals. |
| Keep Python | Experiment runner, workshop, distiller policy rules, metrics, analysis, CLI | These encode research choices and benefit from easy iteration. The current profile does not justify a wholesale port. |

For a later small native extension, pass whole buffers through a supported buffer
interface rather than crossing the language boundary once per coordinate; see
[pybind11 buffer/array interfaces](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html).
No C++ dependency or compiler requirement has been added. Moving a vector to C++
does not make its persisted 384 × 4-byte representation smaller. It may reduce
Python object overhead in RAM; verify ownership and copies before claiming this.

Profiles identify call costs, not fair Python-versus-C++ speedups. Native code and
Python have different profiler overhead; use separate unprofiled timings, as
required by the [Python profiler guidance](https://docs.python.org/3.12/library/profile.html).

## Tests and remaining coverage

| Area | Test type / target | Examples |
|---|---|---|
| Task and boundary semantics | Existing unit + integration tests | Success on fifth attempt, fresh policy, scoped evidence, valid citations, no writes during episode |
| Ablation integrity | End-to-end deterministic fixtures | No-read exactly matches no-memory trajectory; source-success filtering; useful inherited evidence |
| Storage | Byte audit + boundary tests | Sum vector/metadata/index, 1,536 vector bytes per item, eviction pressure, no-memory zero bytes |
| Compaction optimization | Differential tests | Serialized-copy reference versus shared records over seeds, modes, variable record sizes, budgets and protected coverage; infeasible plans preserve originals |
| Iteration artifacts | Unit tests | Refuse overwrite/path escape, reject protocol/machine mismatch, detect changed replay and lost success |

Full checks: `main.py check`; developer tools, if installed:
`python -m ruff check src tests benchmarks` and `python -m mypy src/gim`.
No statement/line-coverage percentage is claimed. Remaining gaps are native-backend
parity, process RSS/native allocations, larger-scale dense retrieval, general semantic
encoders, held-out task-family generalization, utility/IG and formal hypothesis tests,
LLM latency/token accounting, and powered confidence/noninferiority analysis.
