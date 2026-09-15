"""Measure compaction time and temporary Python allocations in separate passes."""

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns
import tracemalloc

from benchmark_storage import vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from gim.memory_item import MemoryItem, Provenance
    from gim.memory_store import MemoryStore
    from gim.pruning_policy import PruningPolicy
    from gim.vector_index import VectorIndex

    items = tuple(MemoryItem(
        item_id=f"memory-{i:04d}", content=f"Fixed dense-vector assertion {i}",
        memory_type="fixture", outcome_class="success", scope=("benchmark",),
        provenance=(Provenance(f"episode-{i}", ("event-1",), 1.0),),
        vector=vector(i), embedding_revision="benchmark-analytic-v1", created_generation=i,
    ) for i in range(200))
    pruning = PruningPolicy(mode="fifo", seed=2026, required_failure_families=frozenset())
    full_size = MemoryStore(items).bytes() + VectorIndex(items).overhead_bytes()
    results = []
    for fraction in (1.0, 0.75):
        budget = int(full_size * fraction)

        def setup():
            return MemoryStore(items), VectorIndex(items)

        store, index = setup()
        pruning.enforce(store, index, budget)  # untimed warmup
        timings, fingerprints = [], []
        for _ in range(7):
            store, index = setup()
            start = perf_counter_ns()
            evictions = pruning.enforce(store, index, budget)
            timings.append(perf_counter_ns() - start)
            fingerprints.append(sha256(store.to_bytes() + index.to_bytes()).hexdigest())
        if len(set(fingerprints)) != 1:
            raise AssertionError("compaction is nondeterministic")
        # Independent tracing pass: excludes initial items/store/index and includes
        # temporary allocations during enforce only. This is not whole-process RSS.
        store, index = setup()
        gc.collect()
        tracemalloc.start()
        pruning.enforce(store, index, budget)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({"budget": budget, "median_compaction_ns": statistics.median(timings),
                        "peak_temporary_python_bytes": peak, "evictions": len(evictions),
                        "boundary_bytes": store.bytes() + index.overhead_bytes(),
                        "fingerprint": fingerprints[0]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"fixture": "200 records, dense 384D binary32 vectors, FIFO",
                   "python": sys.version, "source": str(args.source.resolve()),
                   "full_boundary_bytes": full_size, "results": results}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
