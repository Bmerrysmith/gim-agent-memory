"""G0/G1 integration evidence for the approved task, separate from H1/H3."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

from gim.cli import main
from gim.workshop_contract import Feedback, WorkshopObservation, WorkshopRules, WorkshopTask
from gim.workshop_contract import draw_index
from gim.workshop_demo import no_memory, run_workshop_demo
from gim.workshop_policy import WorkshopPolicy


class WorkshopContractTests(unittest.TestCase):
    def test_setup_values_copy_mutable_inputs(self):
        tools = ["blue"] * 6
        locks = ["circle", "square", "triangle"]
        rules = WorkshopRules("opaque", tools)
        task = WorkshopTask("opaque", locks)
        tools.clear()
        locks.clear()
        self.assertEqual(len(rules.working_tools), 6)
        self.assertEqual(task.locks, ("circle", "square", "triangle"))
        with self.assertRaises(FrozenInstanceError):
            rules.family_id = "changed"

    def test_observation_is_a_strict_public_contract(self):
        observation = WorkshopObservation("opaque", "circle", 5)
        self.assertEqual(WorkshopObservation.from_json(observation.to_json()), observation)
        public = json.loads(observation.to_json())
        for modification in (
            {"hidden_map": ["blue"] * 6},
            {"future_locks": ["square", "triangle"]},
            {"revision": "unknown-v2"},
            {"tools": ["privileged-inspect"]},
            {"attempts_remaining": True},
            {"done": 1},
            {"current_lock": "unknown"},
        ):
            with self.subTest(change=modification), self.assertRaises(ValueError):
                WorkshopObservation.from_json(json.dumps(public | modification))

    def test_impossible_terminal_or_feedback_combinations_are_rejected(self):
        with self.assertRaises(ValueError):
            WorkshopObservation("opaque", None, 4, Feedback("circle", "blue", "opened"), True)
        with self.assertRaises(ValueError):
            WorkshopObservation("opaque", "circle", 4, Feedback("circle", "blue", "opened"))
        with self.assertRaises(ValueError):
            WorkshopObservation("opaque", "square", 4, Feedback("circle", "blue", "wrong_tool"))
        with self.assertRaises(ValueError):
            WorkshopObservation("opaque", "circle", 0, Feedback("circle", "blue", "wrong_tool"))

    def test_counter_draw_does_not_depend_on_previous_candidate_sizes(self):
        expected = draw_index(17, "policy-action", 3, 4)
        for size in (1, 2, 3, 4, 4096):
            self.assertIn(draw_index(17, "policy-action", 2, size), range(size))
        self.assertEqual(draw_index(17, "policy-action", 3, 4), expected)
        for kwargs in (
            {"seed": True},
            {"counter": -1},
            {"counter": True},
            {"size": 0},
            {"size": True},
            {"namespace": ""},
        ):
            values = dict(seed=17, namespace="policy-action", counter=3, size=4) | kwargs
            with self.subTest(values=values), self.assertRaises(ValueError):
                draw_index(**values)


class WorkshopDemoTests(unittest.TestCase):
    def test_complete_report_replays_exactly_without_transferred_state(self):
        first = run_workshop_demo(seed=17, workshops=3, episodes=18)
        second = run_workshop_demo(seed=17, workshops=3, episodes=18)
        self.assertEqual(first, second)
        self.assertEqual(first["persistent_memory_bytes"], 0)
        self.assertEqual(first["model_tokens"], 0)
        self.assertEqual(len(first["episodes"]), 18)
        families = [record["schedule"]["task"]["family_id"] for record in first["episodes"]]
        self.assertEqual(len(set(families)), 3)
        self.assertEqual(families[:3], families[3:6])
        for record in first["episodes"]:
            self.assertTrue(record["trajectory"]["terminated"])
            self.assertLessEqual(len(record["trajectory"]["events"]), 5)
            self.assertEqual(record["policy_conflicts"], [])
        self.assertIsNone(no_memory.__closure__)

    def test_previous_real_policy_is_released_before_the_next_is_constructed(self):
        references = []
        checks = self

        class TrackedPolicy(WorkshopPolicy):
            def __init__(self, seed):
                gc.collect()
                for reference in references:
                    checks.assertIsNone(reference(), "predecessor policy survived boundary")
                super().__init__(seed)
                references.append(weakref.ref(self))

            def choose_action(self, observation, memories):
                checks.assertEqual(memories, ())
                checks.assertIsInstance(observation, str)
                return super().choose_action(observation, memories)

        with patch("gim.workshop_demo.WorkshopPolicy", TrackedPolicy):
            run_workshop_demo(seed=23, workshops=2, episodes=8)
        gc.collect()
        self.assertEqual(len(references), 8)
        self.assertTrue(all(reference() is None for reference in references))

    def test_report_is_independent_of_python_hash_seed_in_fresh_processes(self):
        root = Path(__file__).resolve().parents[1]
        command = [
            sys.executable,
            "-B",
            str(root / "main.py"),
            "workshop-demo",
            "--seed",
            "31",
            "--workshops",
            "2",
            "--episodes",
            "6",
        ]
        reports = []
        for hash_seed in ("1", "931"):
            result = subprocess.run(
                command,
                cwd=root,
                env=dict(os.environ, PYTHONHASHSEED=hash_seed),
                capture_output=True,
                text=True,
                check=True,
            )
            reports.append(result.stdout)
        self.assertEqual(reports[0], reports[1])

    def test_cli_saves_replayable_audit_report_and_rejects_invalid_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            with redirect_stdout(io.StringIO()):
                result = main(["workshop-demo", "--episodes", "2", "--output", str(output)])
            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["episodes"], 2)
            self.assertEqual(report["condition"], "no-memory")
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(main(["workshop-demo", "--workshops", "0"]), 1)
        self.assertIn("workshops", error.getvalue())


if __name__ == "__main__":
    unittest.main()
