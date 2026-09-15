"""Frozen workshop choices and isolation, not transfer or H1/H3 evidence.

Memory records below are hand-built interface fixtures. They do not implement
distillation or change the project's existing outcome_class semantics.
"""

from dataclasses import FrozenInstanceError, replace
import gc
import json
import unittest
from unittest.mock import patch
import weakref

from gim.agent import Agent
from gim.environment import StepResult
from gim.memory_item import MemoryItem, Provenance
from gim.workshop_contract import TOOLS, Feedback, WorkshopObservation, draw_index
from gim.workshop_policy import POLICY_REVISION, RULE_MEMORY_TYPE, WorkshopPolicy


def rule(tool, works, *, lock="circle", family="workshop-a", **changes):
    fields = dict(
        item_id=f"fixture-{tool}-{works}",
        content=json.dumps(
            {"lock_type": lock, "tool": tool, "works": works},
            sort_keys=True,
            separators=(",", ":"),
        ),
        memory_type=RULE_MEMORY_TYPE,
        outcome_class="success",
        scope=(family,),
        provenance=(Provenance("fixture-episode", ("fixture-event",), 1.0),),
        vector=(1.0, 0.0),
        embedding_revision="fixture-only-v1",
        created_generation=0,
    )
    fields.update(changes)
    return MemoryItem(**fields)


def observation(lock="circle", *, attempts=5, feedback=None, family="workshop-a", done=False):
    return WorkshopObservation(family, lock, attempts, feedback, done).to_json()


class WorkshopChoiceTests(unittest.TestCase):
    def test_no_memory_uses_stable_sorted_candidates_and_keyed_draw(self):
        for seed in (-7, 0, 1, 28, 2**200):
            with self.subTest(seed=seed):
                policy = WorkshopPolicy(seed)
                expected = sorted(TOOLS)[draw_index(seed, "policy-action", 0, len(TOOLS))]
                self.assertEqual(policy.choose_action(observation(), ()), expected)
                self.assertEqual(policy.conflicts, ())
        self.assertEqual(POLICY_REVISION, "workshop-policy-v1")

    def test_one_positive_identifies_working_tool_regardless_of_outcome_class(self):
        # Assertion meaning comes only from works; terminal episode outcomes
        # remain independent and are not reinterpreted by this adapter.
        for outcome in ("success", "failure"):
            with self.subTest(outcome=outcome):
                policy = WorkshopPolicy(8)
                self.assertEqual(
                    policy.choose_action(
                        observation(), (rule("yellow", True, outcome_class=outcome),)
                    ),
                    "yellow",
                )

    def test_duplicate_support_does_not_change_the_selected_action(self):
        memory = rule("red", True)
        policy = WorkshopPolicy(10)
        self.assertEqual(
            policy.choose_action(observation(), (memory, replace(memory, item_id="copy"))),
            "red",
        )
        self.assertEqual(policy.conflicts, ())

    def test_compatible_negatives_remove_tools_for_either_outcome_class(self):
        memories = tuple(rule(tool, False, outcome_class="failure") for tool in TOOLS[:-1])
        for seed in range(12):
            with self.subTest(seed=seed):
                self.assertEqual(
                    WorkshopPolicy(seed).choose_action(observation(), memories), "yellow"
                )

    def test_current_observation_eliminates_a_wrong_tool(self):
        policy = WorkshopPolicy(30)
        first = policy.choose_action(observation(), ())
        second = policy.choose_action(
            observation(attempts=4, feedback=Feedback("circle", first, "wrong_tool")), ()
        )
        self.assertNotEqual(first, second)
        expected_candidates = sorted(set(TOOLS) - {first})
        self.assertEqual(second, expected_candidates[draw_index(30, "policy-action", 1, 3)])

    def test_candidates_reset_for_new_lock_without_shared_tool_elimination(self):
        policy = WorkshopPolicy(11)
        first = policy.choose_action(observation(), (rule("blue", True),))
        second = policy.choose_action(
            observation("diamond", attempts=4, feedback=Feedback("circle", first, "opened")),
            (rule("blue", True, lock="diamond"),),
        )
        self.assertEqual((first, second), ("blue", "blue"))

    def test_malformed_unrelated_and_unscoped_facts_are_ignored(self):
        baseline = WorkshopPolicy(25).choose_action(observation(), ())
        normal = rule("yellow", True)
        invalid_contents = (
            "not json",
            "null",
            "[]",
            '{"lock_type":"circle","tool":"yellow","works":1}',
            '{"lock_type":"circle","tool":"yellow","works":"true"}',
            '{"lock_type":"circle","tool":["yellow"],"works":true}',
            '{"lock_type":"circle","tool":"purple","works":true}',
            '{"lock_type":"unknown","tool":"yellow","works":true}',
            '{"lock_type":"circle","tool":"yellow","works":true,"extra":1}',
            '{"lock_type":"circle","tool":"red","tool":"yellow","works":true}',
            normal.content + " ",
            "[" * 2000 + "0" + "]" * 2000,
            '{"lock_type":"circle","tool":"yellow","works":' + "1" * 5000 + "}",
        )
        records = (
            rule("yellow", True, family="another-workshop"),
            rule("yellow", True, lock="square"),
            replace(normal, scope=()),
            replace(normal, memory_type="free-text"),
            *(replace(normal, content=content) for content in invalid_contents),
        )
        for memory in records:
            with self.subTest(content=memory.content[:100], scope=memory.scope):
                policy = WorkshopPolicy(25)
                self.assertEqual(policy.choose_action(observation(), (memory,)), baseline)
                self.assertEqual(policy.conflicts, ())

    def test_exact_family_can_be_one_of_multiple_explicit_scopes(self):
        memory = rule("green", True, scope=("other", "workshop-a"))
        self.assertEqual(WorkshopPolicy(7).choose_action(observation(), (memory,)), "green")

    def test_inherited_facts_are_not_retained_when_the_next_retrieval_omits_them(self):
        policy = WorkshopPolicy(4)
        # Initial memory rules out three tools, but only the resulting direct
        # failure is available when retrieval returns no evidence next time.
        memories = tuple(rule(tool, False) for tool in TOOLS if tool != "blue")
        first = policy.choose_action(observation(), memories)
        self.assertEqual(first, "blue")
        next_observation = observation(attempts=4, feedback=Feedback("circle", first, "wrong_tool"))
        actual = policy.choose_action(next_observation, ())
        expected = sorted(set(TOOLS) - {"blue"})[draw_index(4, "policy-action", 1, 3)]
        self.assertEqual(actual, expected)
        self.assertEqual(policy.conflicts, ())

    def test_equal_inputs_replay_actions_and_conflicts_exactly(self):
        def replay():
            policy = WorkshopPolicy(29)
            first = policy.choose_action(observation(), (rule("blue", True), rule("red", True)))
            second = policy.choose_action(
                observation(attempts=4, feedback=Feedback("circle", first, "wrong_tool")),
                (),
            )
            return (first, second), policy.conflicts

        self.assertEqual(replay(), replay())

    def test_later_random_key_does_not_depend_on_previous_candidate_count(self):
        unconstrained = WorkshopPolicy(21)
        constrained = WorkshopPolicy(21)
        with patch("gim.workshop_policy.draw_index", wraps=draw_index) as recorded:
            left_first = unconstrained.choose_action(observation(), ())
            right_first = constrained.choose_action(observation(), (rule("yellow", True),))
            left_next = unconstrained.choose_action(
                observation(
                    "diamond", attempts=4, feedback=Feedback("circle", left_first, "opened")
                ),
                (),
            )
            right_next = constrained.choose_action(
                observation(
                    "diamond", attempts=4, feedback=Feedback("circle", right_first, "opened")
                ),
                (),
            )
        self.assertEqual(left_next, right_next)
        self.assertEqual(
            [call.args for call in recorded.call_args_list],
            [
                (21, "policy-action", 0, 4),
                (21, "policy-action", 0, 1),
                (21, "policy-action", 1, 4),
                (21, "policy-action", 1, 4),
            ],
        )


class WorkshopConflictTests(unittest.TestCase):
    def assert_fallback(self, memories, reason):
        policy = WorkshopPolicy(12)
        actual = policy.choose_action(observation(), memories)
        expected = WorkshopPolicy(12).choose_action(observation(), ())
        self.assertEqual(actual, expected)
        (conflict,) = policy.conflicts
        self.assertEqual(
            (conflict.family_id, conflict.lock_type, conflict.decision_number, conflict.reason),
            ("workshop-a", "circle", 1, reason),
        )
        return policy

    def test_disagreeing_positives_discard_the_entire_inherited_batch(self):
        self.assert_fallback(
            (rule("red", True), rule("yellow", True), rule("blue", False)),
            "disagreeing_positive_assertions",
        )

    def test_positive_and_negative_same_tool_conflict(self):
        self.assert_fallback(
            (rule("green", True), rule("green", False)),
            "positive_and_negative_assertions",
        )

    def test_all_tools_ruled_out_fall_back_without_empty_random_draw(self):
        self.assert_fallback(
            tuple(rule(tool, False) for tool in TOOLS), "inherited_assertions_exhaust_candidates"
        )

    def test_disproven_positive_falls_back_to_local_observations(self):
        policy = WorkshopPolicy(6)
        first = policy.choose_action(observation(), ())
        next_observation = observation(attempts=4, feedback=Feedback("circle", first, "wrong_tool"))
        second = policy.choose_action(next_observation, (rule(first, True),))
        self.assertNotEqual(first, second)
        self.assertEqual(policy.conflicts[0].reason, "assertion_contradicts_observation")
        self.assertEqual(policy.conflicts[0].decision_number, 2)

    def test_negatives_exhausting_remaining_local_tools_fall_back(self):
        policy = WorkshopPolicy(15)
        first = policy.choose_action(observation(), ())
        second = policy.choose_action(
            observation(attempts=4, feedback=Feedback("circle", first, "wrong_tool")),
            tuple(rule(tool, False) for tool in TOOLS if tool != first),
        )
        self.assertNotEqual(first, second)
        self.assertEqual(policy.conflicts[0].reason, "inherited_assertions_exhaust_candidates")

    def test_conflicts_are_immutable_detached_diagnostics(self):
        policy = self.assert_fallback(
            (rule("red", True), rule("yellow", True)), "disagreeing_positive_assertions"
        )
        snapshot = policy.conflicts
        with self.assertRaises(FrozenInstanceError):
            snapshot[0].reason = "changed"
        with self.assertRaises(AttributeError):
            policy.conflicts = ()
        self.assertEqual(len(snapshot), 1)
        self.assertIsNot(snapshot, policy.conflicts)

    def test_a_later_clean_retrieval_can_be_used_after_conflict(self):
        policy = WorkshopPolicy(1)
        first = policy.choose_action(observation(), tuple(rule(tool, False) for tool in TOOLS))
        other = next(tool for tool in TOOLS if tool != first)
        second = policy.choose_action(
            observation(attempts=4, feedback=Feedback("circle", first, "wrong_tool")),
            (rule(other, True),),
        )
        self.assertEqual(second, other)
        self.assertEqual(len(policy.conflicts), 1)


class WorkshopIsolationTests(unittest.TestCase):
    def test_seed_type_is_explicit(self):
        for invalid in (True, 1.5, "1", None):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                WorkshopPolicy(invalid)

    def test_policy_reuse_or_skipped_steps_fail(self):
        for attempts, family in ((5, "workshop-a"), (3, "workshop-a"), (4, "different")):
            policy = WorkshopPolicy(20)
            first = policy.choose_action(observation(), ())
            next_observation = (
                observation()
                if attempts == 5
                else observation(
                    attempts=attempts,
                    family=family,
                    feedback=Feedback("circle", first, "wrong_tool"),
                )
            )
            with self.subTest(attempts=attempts, family=family), self.assertRaises(ValueError):
                policy.choose_action(next_observation, ())

    def test_starting_mid_episode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fresh policy"):
            WorkshopPolicy(2).choose_action(
                observation(attempts=4, feedback=Feedback("circle", "red", "wrong_tool")), ()
            )

    def test_feedback_for_another_action_is_rejected(self):
        policy = WorkshopPolicy(20)
        first = policy.choose_action(observation(), ())
        other = next(tool for tool in TOOLS if tool != first)
        with self.assertRaisesRegex(ValueError, "preceding action"):
            policy.choose_action(
                observation(attempts=4, feedback=Feedback("circle", other, "wrong_tool")), ()
            )

    def test_four_observed_wrong_tools_are_a_world_contradiction(self):
        policy = WorkshopPolicy(4)
        current = observation()
        # The fourth failed distinct tool would rule out every possible hidden
        # rule. Do not silently erase direct evidence to keep choosing actions.
        for attempts in (4, 3, 2, 1):
            action = policy.choose_action(current, ())
            current = observation(
                attempts=attempts, feedback=Feedback("circle", action, "wrong_tool")
            )
        with self.assertRaisesRegex(ValueError, "contradictory within-episode"):
            policy.choose_action(current, ())

    def test_reopening_a_previous_lock_is_rejected(self):
        policy = WorkshopPolicy(4)
        first = policy.choose_action(observation(), ())
        second = policy.choose_action(
            observation("diamond", attempts=4, feedback=Feedback("circle", first, "opened")), ()
        )
        with self.assertRaisesRegex(ValueError, "new distinct lock"):
            policy.choose_action(
                observation("circle", attempts=3, feedback=Feedback("diamond", second, "opened")),
                (),
            )

    def test_a_fourth_lock_is_rejected(self):
        policy = WorkshopPolicy(4)
        first = policy.choose_action(observation(), ())
        second = policy.choose_action(
            observation("diamond", attempts=4, feedback=Feedback("circle", first, "opened")), ()
        )
        third = policy.choose_action(
            observation("star", attempts=3, feedback=Feedback("diamond", second, "opened")), ()
        )
        with self.assertRaisesRegex(ValueError, "third opened lock"):
            policy.choose_action(
                observation("triangle", attempts=2, feedback=Feedback("star", third, "opened")), ()
            )

    def test_terminal_observation_is_not_actionable(self):
        with self.assertRaisesRegex(ValueError, "terminal"):
            WorkshopPolicy(0).choose_action(
                observation(
                    None, attempts=2, feedback=Feedback("circle", "red", "opened"), done=True
                ),
                (),
            )

    def test_fresh_policies_do_not_share_observations_or_conflicts(self):
        used = WorkshopPolicy(19)
        first = used.choose_action(observation(), tuple(rule(tool, False) for tool in TOOLS))
        used.choose_action(
            observation(attempts=4, feedback=Feedback("circle", first, "wrong_tool")), ()
        )
        fresh = WorkshopPolicy(19)
        self.assertEqual(fresh.choose_action(observation(), ()), first)
        self.assertEqual(fresh.conflicts, ())

    def test_fresh_agent_factory_can_release_policy_and_episode_state(self):
        class OneStepFixture:
            def reset(self, seed):
                return observation()

            def step(self, action):
                # A bounded Agent run can be truncated; no memory write is
                # performed. This fixture checks object lifetime only.
                return StepResult(
                    observation(attempts=4, feedback=Feedback("circle", action, "wrong_tool")),
                    0.0,
                    False,
                )

        policy = WorkshopPolicy(3)
        policy_ref = weakref.ref(policy)
        agent = Agent(policy)
        agent_ref = weakref.ref(agent)
        trajectory = agent.run(
            OneStepFixture(),
            seed=3,
            generation=0,
            episode_id="fixture",
            retrieve=lambda _: (),
            max_steps=1,
        )
        self.assertEqual(len(trajectory.events), 1)
        del agent, policy
        gc.collect()
        self.assertIsNone(agent_ref())
        self.assertIsNone(policy_ref())


if __name__ == "__main__":
    unittest.main()
