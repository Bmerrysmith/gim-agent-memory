"""Integrity and behavioral controls for the engineering ablation harness."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.iteration_suite import StructuralEmbedder, ablation_comparisons, compare_iterations
from benchmarks.iteration_suite import reserve_output, run_lineage


class IterationSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {
            arm: run_lineage(arm=arm, budget=0 if arm == "no_memory" else 32768,
                             seed=2026, episodes=12, workshops=3)
            for arm in ("no_memory", "no_read", "fifo", "success_only")
        }

    def test_fixture_encodes_only_public_lock_identity(self):
        encoder = StructuralEmbedder()
        observation = encoder.embed(json.dumps({"current_lock": "circle", "tools": ["red"]}))
        positive = encoder.embed(json.dumps({"lock_type": "circle", "tool": "blue", "works": True}))
        negative = encoder.embed(json.dumps({"lock_type": "circle", "tool": "red", "works": False}))
        self.assertEqual(observation, positive)
        self.assertEqual(positive, negative)
        self.assertEqual(len(observation), 384)
        self.assertEqual(sum(x * x for x in observation), 1)
        with self.assertRaises(ValueError):
            encoder.embed('{"current_lock": null}')

    def test_no_read_preserves_entire_no_memory_trajectory_despite_writes(self):
        def trajectories(run):
            return [event["trajectory"] for episode in run["audit"] for event in episode["events"]
                    if event.get("status") == "terminal"]
        self.assertEqual(trajectories(self.runs["no_memory"]), trajectories(self.runs["no_read"]))
        self.assertTrue(any(r["boundary_bytes"] > 0 for r in self.runs["no_read"]["rows"]))
        self.assertTrue(all(r["boundary_bytes"] == 0 for r in self.runs["no_memory"]["rows"]))

    def test_source_success_ablation_keeps_no_failed_episode_memories(self):
        memories = [event["memory"] for episode in self.runs["success_only"]["audit"]
                    for event in episode["events"] if event.get("action") == "insert"]
        self.assertTrue(memories)
        self.assertTrue(all(m["outcome_class"] == "success" for m in memories))
        self.assertTrue(all(p["source_reward"] == 1 for m in memories for p in m["provenance"]))

    def test_byte_components_and_eviction_pressure(self):
        run = self.runs["fifo"]
        self.assertGreater(run["metrics"]["evictions"], 0)
        for row in run["rows"]:
            self.assertLessEqual(row["boundary_bytes"], 32768)
            self.assertEqual(row["boundary_bytes"], row["store_bytes"] + row["index_bytes"])
            self.assertEqual(row["store_bytes"], row["vector_bytes"] + row["metadata_bytes"])
            self.assertEqual(row["vector_bytes"] % 1536, 0)
            self.assertGreaterEqual(row["peak_bytes"], row["boundary_bytes"])

    def test_restart_replay_and_useful_transfer_fixture(self):
        replay = run_lineage(arm="fifo", budget=32768, seed=2026, episodes=12, workshops=3)
        self.assertEqual(replay["fingerprint"], self.runs["fifo"]["fingerprint"])
        reference = run_lineage(arm="reference", budget=1000000, seed=2026,
                                episodes=24, workshops=3)
        control = run_lineage(arm="no_memory", budget=0, seed=2026, episodes=24, workshops=3)
        # A fixed plumbing regression fixture, not a scientific success claim.
        self.assertGreater(reference["metrics"]["successes"], control["metrics"]["successes"])
        self.assertEqual(reference["metrics"]["evictions"], 0)

    def test_reference_rejects_a_cap_that_actually_prunes(self):
        with self.assertRaises(AssertionError):
            run_lineage(arm="reference", budget=12, seed=2026, episodes=1, workshops=1)

    def test_comparison_rejects_confounds_and_detects_regression(self):
        baseline = {"protocol_hash": "p", "machine": {"host": "a"}, "lineages": [
            {"key": "fifo/32768/2026", "successes": 7, "fingerprint": "same",
             "mean_boundary_bytes": 200, "generation_ms": 20}]}
        candidate = deepcopy(baseline)
        candidate["lineages"][0]["generation_ms"] = 10
        row = compare_iterations(baseline, candidate)[0]
        self.assertTrue(row["accepted_semantics_preserving"])
        self.assertEqual(row["speedup"], 2)
        candidate["lineages"][0]["successes"] = 6
        candidate["lineages"][0]["fingerprint"] = "different"
        self.assertFalse(compare_iterations(baseline, candidate)[0]["accepted_semantics_preserving"])
        for field, value in (("protocol_hash", "other"), ("machine", {}), ("lineages", [])):
            altered = deepcopy(baseline)
            altered[field] = value
            with self.assertRaises(ValueError):
                compare_iterations(baseline, altered)

    def test_artifacts_cannot_escape_output_root_or_overwrite_baselines(self):
        with tempfile.TemporaryDirectory() as folder:
            output = reserve_output(folder, "baseline")
            self.assertEqual(output.parent, Path(folder))
            with self.assertRaises(FileExistsError):
                reserve_output(folder, "baseline")
            for label in ("../escape", "a/b", "a\\b", "", "C:\\escape", ".."):
                with self.assertRaises(ValueError):
                    reserve_output(folder, label)

    def test_ablation_contrasts_pair_seeds_caps_and_report_success_loss(self):
        def row(arm, budget, seed, success, size, duration):
            return {"key": f"{arm}/{budget}/{seed}", "arm": arm, "budget": budget,
                    "seed": seed, "successes": success,
                    "mean_boundary_bytes": size, "generation_ms": duration}
        rows = ablation_comparisons([
            row("reference", 1000000, 1, 10, 90000, 100),
            row("fifo", 32768, 1, 8, 30000, 50),
            row("no_read", 32768, 1, 3, 32000, 40),
            row("reference", 1000000, 2, 9, 80000, 90),
        ])
        reference = next(r for r in rows if r["candidate"] == "fifo/32768/1"
                         and r["control"] == "reference/1000000/1")
        self.assertEqual(reference["mean_bytes_saved"], 60000)
        self.assertEqual(reference["speedup"], 2)
        self.assertFalse(reference["success_preserved"])
        self.assertFalse(reference["equal_cap"])
        read = next(r for r in rows if r["control"] == "no_read/32768/1")
        self.assertTrue(read["equal_cap"])
        self.assertEqual(read["success_delta"], 5)
        self.assertTrue(all(r["candidate"].endswith("/1") for r in rows))


if __name__ == "__main__":
    unittest.main()
