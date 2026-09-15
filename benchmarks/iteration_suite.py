"""Serial, fresh-process workshop ablations and immutable iteration records.

Run with the project interpreter: benchmarks/iteration_suite.py --label baseline
This task-specific encoder is an engineering fixture, not a research encoder.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import platform
import pstats
import random
import re
import statistics
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gim.config import RunConfig  # noqa: E402
from gim.generation_runner import run_generation  # noqa: E402
from gim.memory_store import MemoryStore  # noqa: E402
from gim.pruning_policy import PruningPolicy  # noqa: E402
from gim.vector_index import VectorIndex  # noqa: E402
from gim.workshop_contract import LOCK_TYPES, MAX_ATTEMPTS, WORKSHOP_REVISION  # noqa: E402
from gim.workshop_contract import draw_index  # noqa: E402
from gim.workshop_distiller import WORKSHOP_DISTILLER_REVISION, WorkshopDistiller  # noqa: E402
from gim.workshop_environment import WorkshopEnvironment, WorkshopEvaluator  # noqa: E402
from gim.workshop_environment import generate_task, generate_workshops  # noqa: E402
from gim.workshop_policy import POLICY_REVISION, WorkshopPolicy  # noqa: E402

PROTOCOL = "workshop-engineering-ablation-v1"
ARMS = ("no_memory", "reference", "fifo", "random", "no_read", "success_only")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


class StructuralEmbedder:
    """384D lock-identity fixture; no tools, rewards, or hidden mappings encoded."""

    revision = "engineering-lock-onehot-384-v1"
    dimension = 384

    def embed(self, text):
        data = json.loads(text)
        lock = data.get("current_lock", data.get("lock_type"))
        if lock not in LOCK_TYPES:
            raise ValueError("fixture requires a public, nonterminal lock identity")
        position = LOCK_TYPES.index(lock)
        return tuple(1.0 if i == position else 0.0 for i in range(self.dimension))


class SourceSuccessDistiller:
    def distill(self, trajectory):
        if trajectory.terminal_reward != 1.0:
            return ()
        return WorkshopDistiller().distill(trajectory)


class AuditBuffer:
    """Evaluator-only evidence, flushed after timing; never passed to successors."""

    def __init__(self):
        self.records = []

    def record(self, kind, **fields):
        self.records.append({"kind": kind, **fields})


def run_lineage(*, arm, budget, seed, episodes, workshops):
    if arm not in ARMS or episodes < 1 or not 1 <= workshops <= 4096:
        raise ValueError("invalid lineage configuration")
    if budget < 0 or (arm == "no_memory") != (budget == 0):
        raise ValueError("only no_memory uses a zero budget")
    definitions = generate_workshops(draw_index(seed, "ablation-mappings", 0, 2**63), workshops)
    # Freeze schedule before running any policy. Policy draws do not use mapping seeds.
    schedule = tuple(
        (
            definitions[generation % workshops],
            draw_index(seed, "ablation-task", generation, 2**63),
            draw_index(seed, "ablation-policy", generation, 2**63),
        )
        for generation in range(episodes)
    )
    quota = 0 if arm == "no_read" else 5
    config = RunConfig(
        budget_bytes=budget, max_steps=MAX_ATTEMPTS, k_positive=quota, k_negative=quota,
        embedding_revision=StructuralEmbedder.revision, policy_revision=POLICY_REVISION,
        environment_revision=WORKSHOP_REVISION,
        distiller_revision=WORKSHOP_DISTILLER_REVISION + (
            "/source-success-only-v1" if arm == "success_only" else ""
        ),
    )
    memory = b""
    rows, audit, replay = [], [], []
    for generation, (rules, task_seed, policy_seed) in enumerate(schedule):
        task = generate_task(rules.family_id, task_seed)
        logger = AuditBuffer()
        start = perf_counter_ns()
        result = run_generation(
            config=config, memory_bytes=memory, episode_id=f"episode-{generation:04d}",
            generation=generation, seed=task_seed, scope=(rules.family_id,),
            policy_factory=lambda: WorkshopPolicy(policy_seed),
            environment_factory=lambda: WorkshopEnvironment(rules, task),
            embedder=StructuralEmbedder(),
            distiller=SourceSuccessDistiller() if arm == "success_only" else WorkshopDistiller(),
            pruning=PruningPolicy(
                mode="random" if arm == "random" else "fifo",
                seed=draw_index(seed, "ablation-pruning", generation, 2**63),
                required_failure_families=frozenset(),
            ),
            logger=logger,
        )
        elapsed = perf_counter_ns() - start
        memory = result.memory_bytes
        labels = dict(WorkshopEvaluator(rules, task, task_seed).evaluate(result.trajectory))
        store = MemoryStore.from_bytes(memory) if memory else MemoryStore()
        index_bytes = VectorIndex(store.items()).overhead_bytes() if memory else 0
        vector_bytes = sum(4 * len(item.vector) for item in store.items())
        if len(memory) + index_bytes != result.boundary_bytes or result.boundary_bytes > budget:
            raise AssertionError("independent boundary byte audit failed")
        evictions = sum(r.get("action") == "evict" for r in logger.records)
        if arm == "reference" and evictions:
            raise AssertionError("reference cap was not large enough; increase protocol cap")
        fingerprint = digest({"trajectory": asdict(result.trajectory),
                              "memory_sha256": sha256(memory).hexdigest()})
        replay.append(fingerprint)
        rows.append({
            "generation": generation, "success": labels["success"],
            "attempts": labels["attempts"], "wrong_tools": labels["wrong_tool_count"],
            "store_bytes": len(memory), "index_bytes": index_bytes,
            "vector_bytes": vector_bytes, "metadata_bytes": len(memory) - vector_bytes,
            "boundary_bytes": result.boundary_bytes, "peak_bytes": result.peak_bytes,
            "evictions": evictions, "generation_ns": elapsed,
            "setup_ns": result.online_cost.setup_ns,
            "retrieval_ns": result.online_cost.retrieval_ns,
            "write_ns": result.online_cost.write_path_ns, "fingerprint": fingerprint,
        })
        audit.append({"task": asdict(task), "task_seed": task_seed,
                      "policy_seed": policy_seed, "events": logger.records})
    successful_times = [r["generation_ns"] for r in rows if r["success"]]
    return {
        "arm": arm, "budget": budget, "seed": seed, "rows": rows, "audit": audit,
        "fingerprint": digest(replay),
        "metrics": {
            "successes": sum(r["success"] for r in rows), "episodes": episodes,
            "attempts": sum(r["attempts"] for r in rows),
            "generation_ms": sum(r["generation_ns"] for r in rows) / 1e6,
            "successful_generation_ms": (
                statistics.mean(successful_times) / 1e6 if successful_times else None
            ),
            "mean_boundary_bytes": statistics.mean(r["boundary_bytes"] for r in rows),
            "final_boundary_bytes": rows[-1]["boundary_bytes"],
            "max_peak_bytes": max(r["peak_bytes"] for r in rows),
            "evictions": sum(r["evictions"] for r in rows),
        },
    }


def compare_iterations(previous, current):
    if previous["protocol_hash"] != current["protocol_hash"]:
        raise ValueError("protocol mismatch: iteration comparison would be confounded")
    if previous["machine"] != current["machine"]:
        raise ValueError("machine/runtime mismatch: timing comparison is not controlled")
    old = {r["key"]: r for r in previous["lineages"]}
    new = {r["key"]: r for r in current["lineages"]}
    if old.keys() != new.keys():
        raise ValueError("iteration comparison requires identical lineage keys")
    rows = []
    for key, after in new.items():
        before = old[key]
        preserved = after["successes"] >= before["successes"]
        exact = before["fingerprint"] == after["fingerprint"]
        rows.append({
            "key": key, "success_preserved": preserved, "exact_replay": exact,
            "bytes_saved": before["mean_boundary_bytes"] - after["mean_boundary_bytes"],
            "speedup": before["generation_ms"] / after["generation_ms"],
            "accepted_semantics_preserving": preserved and exact,
        })
    return rows


def ablation_comparisons(lineages):
    """Paired contrasts; reference comparisons have different caps by design."""
    contrasts = []
    for after in lineages:
        controls = [r for r in lineages if r["seed"] == after["seed"] and (
            (r["arm"] == "reference" and after["arm"] != "reference") or
            (r["arm"] == "no_read" and after["arm"] == "fifo" and
             r["budget"] == after["budget"]) or
            (r["arm"] == "fifo" and after["arm"] in ("random", "success_only") and
             r["budget"] == after["budget"])
        )]
        for before in controls:
            contrasts.append({
                "candidate": after["key"], "control": before["key"],
                "equal_cap": after["budget"] == before["budget"],
                "success_delta": after["successes"] - before["successes"],
                "success_preserved": after["successes"] >= before["successes"],
                "mean_bytes_saved": before["mean_boundary_bytes"] - after["mean_boundary_bytes"],
                "speedup": before["generation_ms"] / after["generation_ms"],
            })
    return contrasts


def reserve_output(root, label):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", label):
        raise ValueError("label must be 1–80 letters/digits/dots/underscores/hyphens")
    path = Path(root) / label
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def run_suite(args):
    if args.episodes < 1 or args.repeats < 2 or not 1 <= args.workshops <= 4096:
        raise ValueError("use positive episodes, >=2 repeats, and 1–4096 workshops")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not args.budgets or min(args.budgets) < 12 or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("budgets must be unique and at least the 12-byte store header")
    output = reserve_output(args.output, args.label)
    protocol = {
        "revision": PROTOCOL, "seeds": args.seeds, "budgets": args.budgets,
        "episodes": args.episodes, "workshops": args.workshops, "repeats": args.repeats,
        "arms": ARMS, "encoder": StructuralEmbedder.revision, "quota_per_outcome": 5,
        "coverage": [], "reference_cap": max(1_000_000_000, max(args.budgets)),
        "warmup": "2 no-memory episodes per worker before measured lineage",
    }
    sources = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
        for folder in ("src", "tests", "benchmarks")
        for path in sorted((ROOT / folder).rglob("*.py"))
    }
    machine = {"python": sys.version, "platform": platform.platform(),
               "processor": platform.processor(), "host": platform.node()}
    manifest = {"status": "running", "protocol": protocol, "protocol_hash": digest(protocol),
                "sources": sources, "machine": machine, "label": args.label,
                "clock": "perf_counter_ns", "executable": sys.executable}
    write_json(output / "manifest.json", manifest)
    try:
        jobs = []
        for seed in args.seeds:
            conditions = [("no_memory", 0), ("reference", protocol["reference_cap"])]
            conditions += [(arm, b) for b in args.budgets for arm in ARMS[2:]]
            for arm, budget in conditions:
                for repeat in range(args.repeats):
                    jobs.append({"arm": arm, "budget": budget, "seed": seed,
                                 "episodes": args.episodes, "workshops": args.workshops,
                                 "repeat": repeat})
        random.Random(1701).shuffle(jobs)
        write_json(output / "job-order.json", jobs)
        results = {}
        for number, job in enumerate(jobs):
            destination = output / f"worker-{number:04d}.json"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                            canonical(job), "--worker-output", str(destination)], check=True)
            result = json.loads(destination.read_text(encoding="utf-8"))
            key = f'{job["arm"]}/{job["budget"]}/{job["seed"]}'
            results.setdefault(key, []).append(result)
            print(f"{number + 1}/{len(jobs)} {key}", flush=True)
        lineages = []
        for key, runs in sorted(results.items()):
            if len({r["fingerprint"] for r in runs}) != 1:
                raise AssertionError(f"nondeterministic replay: {key}")
            metrics = dict(runs[0]["metrics"])
            metrics["generation_ms"] = statistics.median(r["metrics"]["generation_ms"] for r in runs)
            times = [r["metrics"]["successful_generation_ms"] for r in runs]
            metrics["successful_generation_ms"] = statistics.median(times) if times[0] else None
            lineages.append({"key": key, "arm": runs[0]["arm"], "budget": runs[0]["budget"],
                             "seed": runs[0]["seed"], "fingerprint": runs[0]["fingerprint"],
                             "repeat_generation_ms": [r["metrics"]["generation_ms"] for r in runs],
                             **metrics})
        summary = {"protocol_hash": manifest["protocol_hash"], "machine": machine,
                   "lineages": lineages, "label": args.label,
                   "ablation_comparisons": ablation_comparisons(lineages)}
        if args.compare:
            summary["comparison"] = compare_iterations(
                json.loads(Path(args.compare).read_text(encoding="utf-8")), summary)
        write_json(output / "summary.json", summary)
        with (output / "lineages.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(lineages[0]))
            writer.writeheader()
            writer.writerows(lineages)
        report = [f"# Engineering iteration: {args.label}", "",
                  "Goal: open three locks within five attempts. Each episode starts a fresh agent.",
                  "Times are medians across fresh-process repeats of each complete lineage.",
                  "No LLM, utility/IG, prototypes, rare-family protection, or formal H1/H3 test.", "",
                  "| Arm / cap | Successes / tasks | Mean boundary bytes | Lineage ms |",
                  "|---|---:|---:|---:|"]
        for arm, budget in sorted({(r["arm"], r["budget"]) for r in lineages}):
            group = [r for r in lineages if (r["arm"], r["budget"]) == (arm, budget)]
            report.append(f"| {arm} / {budget} | {sum(r['successes'] for r in group)} / "
                          f"{sum(r['episodes'] for r in group)} | "
                          f"{statistics.mean(r['mean_boundary_bytes'] for r in group):.1f} | "
                          f"{statistics.mean(r['generation_ms'] for r in group):.2f} |")
        report += ["", "Bytes include serialized metadata, provenance, float32 vectors, and index.",
                   "The cap applies at boundaries; pre-compaction peaks are separately recorded.",
                   "These are persistent bytes, not process RAM. Audit files are evaluator-only.",
                   "Matched schedules do not imply matched candidate streams: agents act differently.",
                   "Seeds/lineages are the independent units; episodes and repeats are not replicates."]
        if args.compare:
            report += ["", "## Comparison to previous iteration", "",
                       "| Lineage | Exact replay | Bytes saved (mean) | Speedup |", "|---|---|---:|---:|"]
            for row in summary["comparison"]:
                report.append(f"| {row['key']} | {row['exact_replay']} | "
                              f"{row['bytes_saved']:.1f} | {row['speedup']:.3f}x |")
        (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        if args.profile:
            profiler = cProfile.Profile()
            profiler.runcall(run_lineage, arm="fifo", budget=max(args.budgets), seed=args.seeds[0],
                            episodes=args.episodes, workshops=args.workshops)
            profiler.dump_stats(str(output / "profile.pstats"))
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(35)
            stats.sort_stats("tottime").print_stats(25)
            (output / "profile.txt").write_text(stream.getvalue(), encoding="utf-8")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "iterations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--budgets", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--workshops", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker)
        job.pop("repeat")
        run_lineage(arm="no_memory", budget=0, seed=0, episodes=2, workshops=1)
        write_json(args.worker_output, run_lineage(**job))
    elif not args.label:
        parser.error("--label is required; each iteration must have a new label")
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
