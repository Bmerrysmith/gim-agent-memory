"""Offline commands for inspecting and validating the foundation."""

import argparse
import json
from pathlib import Path
import sys
import unittest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="run source-checkout invariant tests")
    commands.add_parser("status", help="show implementation/research boundaries")
    inspect_parser = commands.add_parser("inspect", help="measure a saved memory store")
    inspect_parser.add_argument("path", type=Path)
    workshop_parser = commands.add_parser("workshop-demo", help="run the approved G0-G1 workshop")
    workshop_parser.add_argument("--seed", type=int, default=2026)
    workshop_parser.add_argument("--workshops", type=int, default=3)
    workshop_parser.add_argument("--episodes", type=int, default=12)
    workshop_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "check":
        tests = Path(__file__).resolve().parents[2] / "tests"
        if not tests.is_dir():
            parser.error("check requires the source checkout, including tests/")
        suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_*.py")
        if suite.countTestCases() == 0:
            parser.error("no tests found; refusing an empty passing suite")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

    if args.command == "inspect":
        from gim.memory_store import MemoryStore
        from gim.vector_index import VectorIndex

        try:
            saved_bytes = args.path.read_bytes()
            store = MemoryStore.from_bytes(saved_bytes)
            index = VectorIndex(store.items())
        except (OSError, ValueError) as error:
            print(f"Cannot inspect memory: {error}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "items": len(store.items()),
                    "store_bytes": len(saved_bytes),
                    "vector_bytes": store.vector_bytes(),
                    "metadata_and_framing_bytes": len(saved_bytes) - store.vector_bytes(),
                    "canonical_store_bytes": store.bytes(),
                    "canonical": saved_bytes == store.to_bytes(),
                    "index_bytes": index.overhead_bytes(),
                    "total_persistent_bytes": len(saved_bytes) + index.overhead_bytes(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "workshop-demo":
        from gim.workshop_demo import run_workshop_demo

        try:
            report = run_workshop_demo(
                seed=args.seed, workshops=args.workshops, episodes=args.episodes
            )
            encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output is None:
                print(encoded, end="")
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded, encoding="utf-8")
                print(f"Workshop report saved to {args.output}")
                print(json.dumps(report["summary"], sort_keys=True))
                print("No-memory G0-G1 demonstration; this is not an inheritance result.")
        except (OSError, ValueError) as error:
            print(f"Cannot run workshop demo: {error}", file=sys.stderr)
            return 1
        return 0

    print("GIM infrastructure and approved G0-G1 workshop mechanics.")
    print("Run workshop-demo for reproducible episodes; no inheritance result is claimed.")
    print("Implementation and open decisions: docs/05-implementation-and-research.md")
    print("Architecture governs build order; H1 and H3 are the current focus.")
    return 0
