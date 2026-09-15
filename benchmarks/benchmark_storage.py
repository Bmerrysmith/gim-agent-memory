"""A deterministic storage/search fixture, not a task-transfer experiment.

Run each checkout in a fresh process. --source selects its src directory so that
the same fixture can compare implementations. Warm up once, then report medians
of seven calls. No LLM, model download, or token estimate is involved.
"""

import argparse
from collections.abc import Callable
import gc
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns


def vector(seed: int, dimension: int = 384) -> tuple[float, ...]:
    values = tuple(math.sin((seed + 1) * (i + 1) * 0.137) for i in range(dimension))
    norm = math.hypot(*values)
    return tuple(value / norm for value in values)


def median_ns(operation: Callable[[], object], repeats: int = 7) -> int:
    timings = []
    for _ in range(repeats):
        start = perf_counter_ns()
        operation()
        timings.append(perf_counter_ns() - start)
    return int(statistics.median(timings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # A checkout selector is deliberate here: each side runs in a fresh process,
    # preventing Python's module cache from mixing two implementations.
    sys.path.insert(0, str(args.source.resolve()))
    from gim.memory_item import MemoryItem, Provenance
    from gim.memory_store import MemoryStore
    from gim.retrieval_policy import RetrievalPolicy, RetrievalResult
    from gim.vector_index import VectorIndex

    items = tuple(
        MemoryItem(
            item_id=f"memory-{number:04d}",
            content=f"A fixed benchmark assertion {number}.",
            memory_type="fixture",
            outcome_class="failure" if number % 2 else "success",
            scope=("benchmark",),
            provenance=(Provenance(f"episode-{number}", ("event-1",), 0.0),),
            vector=vector(number),
            embedding_revision="benchmark-analytic-v1",
            created_generation=number,
        )
        for number in range(200)
    )
    store = MemoryStore(items)
    index = VectorIndex(store.items())
    snapshot = store.snapshot()
    retrieval = RetrievalPolicy(5, 5)
    queries = tuple(vector(number + 500) for number in range(10))

    def read_batch() -> tuple[RetrievalResult, ...]:
        return tuple(
            retrieval.retrieve(
                snapshot, index, query, "benchmark-analytic-v1", scope=("benchmark",)
            )
            for query in queries
        )

    encoded = store.to_bytes()
    read_batch()
    gc.collect()
    results = read_batch()
    measurements = {
        "fixture": "200 items, 384 dimensions, 10 queries, k+=5/k-=5",
        "python": sys.version.split()[0],
        "source": str(args.source.resolve()),
        "store_bytes": len(encoded),
        "index_bytes": index.overhead_bytes(),
        "total_bytes": len(encoded) + index.overhead_bytes(),
        "median_ten_retrievals_ns": median_ns(read_batch),
        "median_serialization_ns": median_ns(store.to_bytes),
        "median_deserialization_ns": median_ns(lambda: MemoryStore.from_bytes(encoded)),
        "retrieved_ids": [
            [item.item_id for item in result.positive + result.negative] for result in results
        ],
    }
    output = json.dumps(measurements, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
