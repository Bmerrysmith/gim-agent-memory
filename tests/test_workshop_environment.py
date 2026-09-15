"""Workshop transition and replay checks, not evidence that H1/H3 hold."""

from dataclasses import replace
from itertools import product
import json
import unittest

from gim.trajectory import Event, Trajectory
from gim.workshop_contract import LOCK_TYPES, MAX_ATTEMPTS, TOOLS, WorkshopRules, WorkshopTask
from gim.workshop_environment import (
    WorkshopEnvironment,
    WorkshopEvaluator,
    generate_task,
    generate_workshops,
)


def sample_rules():
    return WorkshopRules("workshop-0000", ("blue", "green", "red", "yellow", "blue", "green"))


def sample_task():
    return WorkshopTask("workshop-0000", LOCK_TYPES[:3])


def episode(rules, task, actions, *, seed=41):
    environment = WorkshopEnvironment(rules, task)
    initial = environment.reset(seed)
    events = []
    for number, action in enumerate(actions, 1):
        result = environment.step(action)
        events.append(Event(f"episode:{number}", number, action, result.observation, result.reward))
    return Trajectory(
        "episode",
        0,
        tuple(events),
        result.observation,
        result.reward,
        terminated=result.done,
        initial_observation=initial,
    )


class WorkshopEnvironmentTests(unittest.TestCase):
    def test_all_4096_maps_open_exactly_the_required_three_locks(self):
        # Exhausting this finite ground-truth space catches wrong lock ordering
        # and accidental tool aliases without relying on statistical sampling.
        task = sample_task()
        for mapping in product(TOOLS, repeat=len(LOCK_TYPES)):
            rules = WorkshopRules(task.family_id, mapping)
            environment = WorkshopEnvironment(rules, task)
            environment.reset(0)
            for number, tool in enumerate(mapping[:3], 1):
                result = environment.step(tool)
                self.assertEqual(result.done, number == 3)
                self.assertEqual(result.reward, float(number == 3))
                observation = json.loads(result.observation)
                self.assertEqual(observation["attempts_remaining"], MAX_ATTEMPTS - number)
                self.assertEqual(observation["last_result"]["result"], "opened")
            self.assertIsNone(observation["current_lock"])

    def test_all_maps_allow_fifth_attempt_success_and_reject_fifth_attempt_failure(self):
        task = sample_task()
        for mapping in product(TOOLS, repeat=len(LOCK_TYPES)):
            rules = WorkshopRules(task.family_id, mapping)
            wrong = next(tool for tool in TOOLS if tool != mapping[0])
            environment = WorkshopEnvironment(rules, task)
            environment.reset(0)
            for action in (wrong, wrong, *mapping[:3]):
                result = environment.step(action)
            self.assertTrue(result.done)
            self.assertEqual(result.reward, 1.0)
            self.assertEqual(json.loads(result.observation)["attempts_remaining"], 0)

            environment.reset(0)
            for number in range(MAX_ATTEMPTS):
                result = environment.step(wrong)
                self.assertEqual(result.done, number == MAX_ATTEMPTS - 1)
                self.assertEqual(result.reward, 0.0)
            self.assertEqual(json.loads(result.observation)["current_lock"], task.locks[0])

    def test_observations_do_not_reveal_unvisited_rules_or_future_locks(self):
        rules = sample_rules()
        first = WorkshopEnvironment(rules, sample_task())
        alternate_rules = WorkshopRules(rules.family_id, ("blue",) + ("yellow",) * 5)
        alternate_task = WorkshopTask(
            rules.family_id, (LOCK_TYPES[0], LOCK_TYPES[4], LOCK_TYPES[5])
        )
        second = WorkshopEnvironment(alternate_rules, alternate_task)
        self.assertEqual(first.reset(3), second.reset(3))
        # Both worlds have the same visible lock and observed failed action.
        # Their hidden rules and future task sequences differ substantially.
        self.assertEqual(first.step("red"), second.step("red"))
        visible = json.loads(first.reset(3))
        self.assertEqual(
            set(visible),
            {
                "revision",
                "family_id",
                "current_lock",
                "attempts_remaining",
                "last_result",
                "done",
                "tools",
            },
        )
        self.assertEqual(visible["tools"], list(TOOLS))
        first_step = json.loads(first.step("blue").observation)
        self.assertEqual(first_step["current_lock"], LOCK_TYPES[1])
        self.assertEqual(first_step["last_result"]["lock_type"], LOCK_TYPES[0])

    def test_invalid_actions_and_lifecycle_calls_do_not_consume_attempts(self):
        environment = WorkshopEnvironment(sample_rules(), sample_task())
        with self.assertRaisesRegex(RuntimeError, "reset"):
            environment.step("blue")
        initial = environment.reset(7)
        for action in ("inspect", "BLUE", " blue", "", 0, True, None):
            with self.assertRaisesRegex(ValueError, "action"):
                environment.step(action)
        result = environment.step("blue")
        self.assertEqual(json.loads(result.observation)["attempts_remaining"], 4)
        environment.step("green")
        environment.step("red")
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            environment.step("blue")
        self.assertEqual(environment.reset(99), initial)

    def test_replay_determinism_and_reset_do_not_leak_prior_state(self):
        environment = WorkshopEnvironment(sample_rules(), sample_task())
        first_initial = environment.reset(-21)
        first = [environment.step(action) for action in ("yellow", "blue", "green", "red")]
        self.assertEqual(environment.reset(-21), first_initial)
        second = [environment.step(action) for action in ("yellow", "blue", "green", "red")]
        self.assertEqual(first, second)

    def test_task_order_can_differ_from_hidden_mapping_order(self):
        rules = sample_rules()
        task = WorkshopTask(rules.family_id, (LOCK_TYPES[5], LOCK_TYPES[0], LOCK_TYPES[3]))
        environment = WorkshopEnvironment(rules, task)
        self.assertEqual(json.loads(environment.reset(7))["current_lock"], LOCK_TYPES[5])
        for action, next_lock in (
            ("green", LOCK_TYPES[0]),
            ("blue", LOCK_TYPES[3]),
            ("yellow", None),
        ):
            result = environment.step(action)
            self.assertEqual(json.loads(result.observation)["current_lock"], next_lock)
        self.assertEqual(result.reward, 1.0)

    def test_constructor_rejects_cross_family_task_and_noninteger_seed(self):
        rules = sample_rules()
        with self.assertRaisesRegex(ValueError, "same workshop family"):
            WorkshopEnvironment(rules, WorkshopTask("other", sample_task().locks))
        environment = WorkshopEnvironment(rules, sample_task())
        for seed in (True, 1.0, "1", None):
            with self.assertRaisesRegex(ValueError, "seed"):
                environment.reset(seed)


class WorkshopGeneratorTests(unittest.TestCase):
    def test_generators_replay_and_preserve_sample_prefix(self):
        first = generate_workshops(420, 8)
        self.assertEqual(first, generate_workshops(420, 8))
        self.assertEqual(first[:3], generate_workshops(420, 3))
        self.assertNotEqual(first, generate_workshops(421, 8))
        self.assertEqual(
            tuple(rule.family_id for rule in first), tuple(f"workshop-{i:04d}" for i in range(8))
        )
        for rule in first:
            self.assertEqual(generate_task(rule.family_id, 23), generate_task(rule.family_id, 23))
            self.assertEqual(len(set(generate_task(rule.family_id, 23).locks)), 3)

    def test_full_population_is_sampled_once_and_labels_do_not_encode_map(self):
        sampled = generate_workshops(11, len(TOOLS) ** len(LOCK_TYPES))
        self.assertEqual(len({rules.working_tools for rules in sampled}), 4096)
        self.assertEqual({rules.working_tools for rules in sampled}, set(product(TOOLS, repeat=6)))
        other = generate_workshops(12, 1)[0]
        self.assertEqual(sampled[0].family_id, other.family_id)
        self.assertNotEqual(sampled[0].working_tools, other.working_tools)

    def test_generator_argument_boundaries(self):
        self.assertEqual(generate_workshops(1, 0), ())
        for count in (-1, 4097, True, 1.0, "2"):
            with self.assertRaisesRegex(ValueError, "count"):
                generate_workshops(1, count)
        for seed in (True, 1.0, "1", None):
            with self.assertRaisesRegex(ValueError, "seed"):
                generate_workshops(seed, 0)
            with self.assertRaisesRegex(ValueError, "seed"):
                generate_task("family", seed)
        for family in ("", "  ", None):
            with self.assertRaisesRegex(ValueError, "family_id"):
                generate_task(family, 1)


class WorkshopEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.rules = sample_rules()
        self.task = sample_task()
        self.evaluator = WorkshopEvaluator(self.rules, self.task, seed=41)

    def test_ground_truth_replays_every_mapping(self):
        for mapping in product(TOOLS, repeat=len(LOCK_TYPES)):
            rules = WorkshopRules(self.task.family_id, mapping)
            evidence = episode(rules, self.task, mapping[:3])
            labels = WorkshopEvaluator(rules, self.task, seed=41).evaluate(evidence)
            self.assertTrue(labels["success"])
            self.assertEqual(labels["locks_opened"], 3)
            self.assertEqual(labels["wrong_tool_count"], 0)

    def test_success_failure_and_repeated_failure_counts_use_observed_triples(self):
        trajectory = episode(self.rules, self.task, ("yellow", "yellow", "blue", "green", "red"))
        self.assertEqual(
            self.evaluator.evaluate(trajectory),
            {
                "success": True,
                "attempts": 5,
                "locks_opened": 3,
                "wrong_tool_count": 2,
                "repeated_failure_count": 1,
                "failure_families": ((self.task.family_id, LOCK_TYPES[0], "yellow"),),
                "horizon_exhausted": False,
                "terminated": True,
            },
        )
        failed = episode(self.rules, self.task, ("yellow", "yellow", "blue", "yellow", "yellow"))
        report = self.evaluator.evaluate(failed)
        self.assertFalse(report["success"])
        self.assertTrue(report["horizon_exhausted"])
        self.assertEqual(report["locks_opened"], 1)
        self.assertEqual(report["repeated_failure_count"], 2)
        self.assertEqual(len(report["failure_families"]), 2)

    def test_valid_truncation_is_neither_success_nor_horizon_failure(self):
        truncated = episode(self.rules, self.task, ("blue",))
        report = self.evaluator.evaluate(truncated)
        self.assertFalse(report["success"])
        self.assertFalse(report["terminated"])
        self.assertFalse(report["horizon_exhausted"])
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["locks_opened"], 1)

    def test_evaluator_rejects_tampered_initial_observation_action_reward_and_result(self):
        original = episode(self.rules, self.task, ("blue", "green", "red"))
        with self.assertRaisesRegex(ValueError, "initial observation"):
            self.evaluator.evaluate(replace(original, initial_observation="forged"))
        for event in (
            replace(original.events[0], action="yellow"),
            replace(original.events[0], reward=1.0),
            replace(original.events[0], observation=original.initial_observation),
        ):
            with self.assertRaisesRegex(ValueError, "differs from replay"):
                self.evaluator.evaluate(replace(original, events=(event, *original.events[1:])))
        with self.assertRaisesRegex(ValueError, "termination flag"):
            self.evaluator.evaluate(replace(original, terminated=False))

    def test_evaluator_rejects_extra_actions_and_false_terminal_flag(self):
        original = episode(self.rules, self.task, ("blue", "green", "red"))
        extra = Event("episode:4", 4, "red", original.terminal_state, original.terminal_reward)
        with self.assertRaisesRegex(ValueError, "after workshop termination"):
            self.evaluator.evaluate(replace(original, events=(*original.events, extra)))
        truncated = episode(self.rules, self.task, ("blue",))
        with self.assertRaisesRegex(ValueError, "termination flag"):
            self.evaluator.evaluate(replace(truncated, terminated=True))

    def test_evaluator_does_not_return_unobserved_map_values(self):
        trajectory = episode(self.rules, self.task, ("yellow",) * 5)
        report = self.evaluator.evaluate(trajectory)
        encoded = json.dumps(report, sort_keys=True)
        for unopened_lock in LOCK_TYPES[1:]:
            self.assertNotIn(unopened_lock, encoded)
        self.assertNotIn("working_tools", encoded)
        self.assertNotIn("rules", encoded)


if __name__ == "__main__":
    unittest.main()
