# Storage and exact-retrieval comparison

The user prioritized useful transfer with less storage, fewer deployed tokens,
and lower latency. This measurement checks the storage and retrieval mechanics.
It does not establish H1/H3, transfer quality, deployed token savings, or G7 ANN
performance. Both sides use exact search and the same 384-dimensional fixture.

## Measured on this Windows host, Python 3.12.14

The fixture contains 200 items, 384 coordinates each, and ten deterministic query
vectors. Retrieval allows five positive and five negative memories per query.
The table reports seven-call medians after warm-up. These timings are a small
local benchmark, not a universal speed guarantee or a powered latency study.

| Measurement | Previous reference | Compact float32 reference |
| --- | ---: | ---: |
| Authoritative store | 1,666,971 B | 376,882 B |
| Charged index | 617,437 B | 3,037 B |
| Store + index | 2,284,408 B | 379,919 B |
| Ten retrievals | 61.70 ms | 33.08 ms |
| Serialization | 27.57 ms | 1.82 ms |
| Deserialization | 24.48 ms | 20.92 ms |

The combined representation is **83.37% smaller** and the measured retrieval
batch is **1.87 times faster**. All 100 selected IDs (ten queries, ten memories
each) matched, including ordering. That finite check does not prove every possible
near-tie ranks identically after float32 rounding. Within the new format, items
are rounded before insertion so decisions remain stable across save/reload.

## Why the result changed

The earlier implementation wrote vector coordinates as decimal JSON and duplicated
binary64 coordinates in its index. The new store writes one binary32 block per
item, and the index references the immutable authoritative coordinates. The index
charges its ID/revision/dimension mapping without another vector payload. Its
staleness check also avoids repacking every coordinate on every query.

A 384-coordinate vector now occupies exactly **1,536 bytes** in the store.
The 200 vectors total 307,200 bytes; metadata/framing contribute 69,682 bytes and
the derived mapping another 3,037 bytes in this fixture. All count toward B.
Python object overhead and transient allocations are RAM, not hidden persisted
payload; this benchmark does not report a RAM reduction.

No dimensions, memory items, evidence records, or retrieval quotas were removed.
No token savings are inferred from byte savings. Those need the deployed model's
actual tokenizer and quality evaluation at the appropriate architecture gate.

## Reproduce the current mechanics

```powershell
.\.venv\Scripts\python.exe benchmarks/benchmark_storage.py --output runs/storage-benchmark.json
```

To compare a saved prior source checkout, run a separate process with
`--source PATH_TO_ITS_SRC_DIRECTORY`. Use the same interpreter and machine for
both sides. Save the raw JSON, including returned IDs, rather than comparing only
wall-clock time. Retest at realistic sizes before making a C++ or ANN decision.

The binary format is versioned (`GMS2`). Old JSON snapshots require an explicit
migration; silently rewriting them could change costs and near-tie rankings.
No saved memory snapshots were present in the project when this change was made.
The current implementation does not claim to supply a legacy migration tool.
