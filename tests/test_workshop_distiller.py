"""Source-outcome separation and public-evidence checks, not H3 results."""

from dataclasses import replace
import json
import unittest

from gim.config import RunConfig
from gim.embedder import NormalizingEmbedder
from gim.experiment_logger import NullLogger
from gim.generation_runner import run_generation
from gim.memory_store import MemoryStore
from gim.pruning_policy import PruningPolicy
from gim.retrieval_policy import RetrievalPolicy
from gim.trajectory import Event, Trajectory
from gim.vector_index import VectorIndex
from gim.workshop_contract import LOCK_TYPES, WORKSHOP_REVISION, WorkshopRules, WorkshopTask
from gim.workshop_distiller import WORKSHOP_DISTILLER_REVISION, WorkshopDistiller
from gim.workshop_environment import WorkshopEnvironment
from gim.workshop_policy import RULE_MEMORY_TYPE


def rules():
    return WorkshopRules("atelier-é", ("blue", "green", "red", "yellow", "blue", "green"))


def environment():
    return WorkshopEnvironment(rules(), WorkshopTask(rules().family_id, LOCK_TYPES[:3]))


def episode(actions):
    workshop = environment()
    initial = workshop.reset(41)
    events = []
    for number, action in enumerate(actions, 1):
        result = workshop.step(action)
        events.append(Event(f"episode:{number}", number, action, result.observation, result.reward))
    return Trajectory(
        "episode",
        0,
        tuple(events),
        events[-1].observation,
        events[-1].reward,
        terminated=result.done,
        initial_observation=initial,
    )


def with_events(trajectory, events):
    return replace(
        trajectory,
        events=tuple(events),
        terminal_state=events[-1].observation,
        terminal_reward=events[-1].reward,
    )


def change_observation(trajectory, position, **fields):
    events = list(trajectory.events)
    payload = json.loads(events[position].observation)
    payload.update(fields)
    events[position] = replace(events[position], observation=json.dumps(payload))
    return with_events(trajectory, events)


class WorkshopDistillerTests(unittest.TestCase):
    def setUp(self):
        self.distiller = WorkshopDistiller()
        self.success = episode(("yellow", "blue", "green", "red"))
        self.failure = episode(("blue", "yellow", "yellow", "yellow", "yellow"))

    def test_successful_episode_preserves_wrong_tool_warning_with_success_source_label(self):
        candidates = self.distiller.distill(self.success)
        self.assertEqual(len(candidates), 4)
        self.assertEqual({candidate.outcome_class for candidate in candidates}, {"success"})
        self.assertEqual(
            [json.loads(candidate.content)["works"] for candidate in candidates],
            [False, True, True, True],
        )
        self.assertEqual(candidates[0].failure_family, '["atelier-é","circle","yellow"]')
        self.assertTrue(all(candidate.failure_family is None for candidate in candidates[1:]))

    def test_failed_episode_preserves_working_tool_with_failure_source_label(self):
        candidates = self.distiller.distill(self.failure)
        self.assertEqual({candidate.outcome_class for candidate in candidates}, {"failure"})
        self.assertEqual(
            [json.loads(candidate.content)["works"] for candidate in candidates],
            [True, False, False, False, False],
        )
        self.assertIsNone(candidates[0].failure_family)
        self.assertEqual(candidates[1].failure_family, '["atelier-é","diamond","yellow"]')

    def test_exact_content_scope_and_each_event_citation_are_reproducible(self):
        first = self.distiller.distill(self.failure)
        second = WorkshopDistiller().distill(self.failure)
        self.assertEqual(first, second)
        for event, candidate in zip(self.failure.events, first, strict=True):
            feedback = json.loads(event.observation)["last_result"]
            self.assertEqual(candidate.event_ids, (event.event_id,))
            self.assertEqual(candidate.scope, (rules().family_id,))
            self.assertEqual(candidate.memory_type, RULE_MEMORY_TYPE)
            self.assertEqual(
                candidate.content,
                json.dumps(
                    {
                        "lock_type": feedback["lock_type"],
                        "tool": feedback["tool"],
                        "works": feedback["result"] == "opened",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
            self.assertIsNone(candidate.information_gain)
        # Equal repeated content remains separately cited for insert-only storage.
        self.assertEqual(first[1].content, first[4].content)
        self.assertNotEqual(first[1].event_ids, first[4].event_ids)

    def test_fifth_attempt_opening_is_success(self):
        trajectory = episode(("yellow", "yellow", "blue", "green", "red"))
        self.assertEqual(
            {candidate.outcome_class for candidate in self.distiller.distill(trajectory)},
            {"success"},
        )

    def test_extractor_does_not_infer_untried_working_tool(self):
        trajectory = episode(("yellow", "red", "green", "yellow", "red"))
        candidates = self.distiller.distill(trajectory)
        self.assertEqual(len(candidates), 5)
        self.assertTrue(all(not json.loads(candidate.content)["works"] for candidate in candidates))
        self.assertNotIn(
            "blue", {json.loads(candidate.content)["tool"] for candidate in candidates}
        )

    def test_requires_frozen_terminal_trajectory_and_complete_reset(self):
        with self.assertRaises(TypeError):
            self.distiller.distill({"episode": "fabricated"})
        truncated = episode(("blue",))
        for malformed in (
            truncated,
            replace(truncated, terminated=True),
            replace(self.success, initial_observation=""),
            replace(self.success, initial_observation=self.success.events[0].observation),
        ):
            with self.subTest(trajectory=malformed), self.assertRaises(ValueError):
                self.distiller.distill(malformed)

    def test_rejects_skipped_attempts_switched_family_and_wrong_reset_lock(self):
        payload = json.loads(self.success.initial_observation)
        payload["current_lock"] = "star"
        cases = (
            change_observation(self.success, 0, attempts_remaining=3),
            change_observation(self.success, 0, family_id="another-family"),
            replace(self.success, initial_observation=json.dumps(payload)),
        )
        for malformed in cases:
            with self.subTest(trajectory=malformed), self.assertRaises(ValueError):
                self.distiller.distill(malformed)

    def test_rejects_feedback_for_another_action_or_lock(self):
        first = self.success.events[0]
        for action in ("red", "inspect"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.distiller.distill(
                    with_events(
                        self.success, (replace(first, action=action), *self.success.events[1:])
                    )
                )
        payload = json.loads(first.observation)
        payload["last_result"]["lock_type"] = "star"
        payload["current_lock"] = "star"
        with self.assertRaisesRegex(ValueError, "preceding lock"):
            self.distiller.distill(
                with_events(
                    self.success,
                    (replace(first, observation=json.dumps(payload)), *self.success.events[1:]),
                )
            )

    def test_rejects_wrong_tool_advancement_and_repeated_opened_lock(self):
        cases = (
            change_observation(self.success, 0, current_lock="diamond"),
            change_observation(self.success, 2, current_lock="circle"),
        )
        for malformed in cases:
            with self.subTest(trajectory=malformed), self.assertRaises(ValueError):
                self.distiller.distill(malformed)

    def test_rejects_early_success_and_continuation_after_third_opening(self):
        early = episode(("yellow", "blue", "green"))
        early = replace(
            change_observation(early, -1, current_lock=None, done=True), terminated=True
        )
        late = change_observation(self.success, -1, current_lock="star", done=False)
        for malformed in (early, late):
            with self.subTest(trajectory=malformed), self.assertRaises(ValueError):
                self.distiller.distill(malformed)

    def test_rejects_actions_after_terminal_and_beyond_horizon(self):
        last = self.success.events[-1]
        extra = replace(last, event_id="episode:5", step=5)
        with self.assertRaisesRegex(ValueError, "after termination"):
            self.distiller.distill(with_events(self.success, (*self.success.events, extra)))
        last = self.failure.events[-1]
        extra = replace(last, event_id="episode:6", step=6)
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.distiller.distill(with_events(self.failure, (*self.failure.events, extra)))

    def test_rejects_reward_inconsistent_with_public_source_outcome(self):
        for trajectory, position, reward in (
            (self.success, 0, 1.0),
            (self.success, -1, 0.0),
            (self.failure, -1, 1.0),
        ):
            events = list(trajectory.events)
            events[position] = replace(events[position], reward=reward)
            with (
                self.subTest(position=position, reward=reward),
                self.assertRaisesRegex(ValueError, "reward"),
            ):
                self.distiller.distill(with_events(trajectory, events))

    def test_rejects_same_tool_failing_then_opening_same_lock(self):
        events = list(self.success.events)
        payload = json.loads(events[1].observation)
        payload["last_result"]["tool"] = "yellow"
        events[1] = replace(events[1], action="yellow", observation=json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "both fail and open"):
            self.distiller.distill(with_events(self.success, events))

    def test_rejects_all_four_tools_claimed_wrong_for_one_lock(self):
        trajectory = episode(("yellow",) * 5)
        events = list(trajectory.events)
        for position, tool in enumerate(("blue", "green", "red", "yellow")):
            payload = json.loads(events[position].observation)
            payload["last_result"]["tool"] = tool
            events[position] = replace(
                events[position], action=tool, observation=json.dumps(payload)
            )
        with self.assertRaisesRegex(ValueError, "one working tool"):
            self.distiller.distill(with_events(trajectory, events))

    def test_rejects_ambiguous_or_malformed_json_and_hidden_fields(self):
        first = self.success.events[0]
        duplicate = first.observation.replace('"done":false', '"done":true,"done":false')
        nested_duplicate = first.observation.replace(
            '"tool":"yellow"', '"tool":"red","tool":"yellow"'
        )
        for text in ("not JSON", duplicate, nested_duplicate, "[" * 2000):
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                self.distiller.distill(
                    with_events(
                        self.success, (replace(first, observation=text), *self.success.events[1:])
                    )
                )
        for fields in (
            {"hidden_rules": {"circle": "blue"}},
            {"revision": "workshop-unknown"},
            {"attempts_remaining": True},
            {"done": "false"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.distiller.distill(change_observation(self.success, 0, **fields))

    def test_terminal_extraction_retains_source_reward_and_source_based_quotas(self):
        # A fixed two-coordinate encoder and scripted actions isolate the storage
        # contract. This is deliberately not a research encoder or an H3 trial.
        class ScriptedPolicy:
            def __init__(self, actions):
                self.actions = iter(actions)

            def choose_action(self, observation, memories):
                return next(self.actions)

        embedder = NormalizingEmbedder("workshop-test-only-v1", 2, lambda text: (1.0, 0.0))
        config = RunConfig(
            100_000,
            5,
            4,
            5,
            embedder.revision,
            "scripted-test-only-v1",
            WORKSHOP_REVISION,
            WORKSHOP_DISTILLER_REVISION,
        )
        inherited = b""
        for generation, (name, actions) in enumerate(
            (
                ("failed", ("blue", "yellow", "yellow", "yellow", "yellow")),
                ("succeeded", ("yellow", "blue", "green", "red")),
            )
        ):
            result = run_generation(
                config=config,
                memory_bytes=inherited,
                episode_id=name,
                generation=generation,
                seed=41,
                scope=(rules().family_id,),
                policy_factory=lambda: ScriptedPolicy(actions),
                environment_factory=environment,
                embedder=embedder,
                distiller=self.distiller,
                pruning=PruningPolicy(mode="fifo", seed=41, required_failure_families=frozenset()),
                logger=NullLogger(),
            )
            inherited = result.memory_bytes
        store = MemoryStore.from_bytes(inherited)
        retrieved = RetrievalPolicy(4, 5).retrieve(
            store.snapshot(),
            VectorIndex(store.items()),
            (1.0, 0.0),
            embedder.revision,
            scope=(rules().family_id,),
        )
        self.assertEqual((len(retrieved.positive), len(retrieved.negative)), (4, 5))
        warning = next(item for item in retrieved.positive if not json.loads(item.content)["works"])
        useful = next(item for item in retrieved.negative if json.loads(item.content)["works"])
        self.assertEqual(warning.provenance[0].source_reward, 1.0)
        self.assertEqual(warning.provenance[0].event_ids, ("succeeded:1",))
        self.assertEqual(warning.failure_family, '["atelier-é","circle","yellow"]')
        self.assertEqual(useful.provenance[0].source_reward, 0.0)
        self.assertEqual(useful.provenance[0].event_ids, ("failed:1",))
        self.assertTrue(all(item.information_gain is None for item in store.items()))
        # The existing budgeted coverage tag also protects a warning whose
        # episode succeeded; it need not be relabeled into the negative quota.
        budget = MemoryStore((warning,)).bytes() + VectorIndex((warning,)).overhead_bytes()
        pruning = PruningPolicy(
            mode="fifo", seed=41, required_failure_families=frozenset({warning.failure_family})
        )
        index = VectorIndex(store.items())
        pruning.enforce(store, index, budget)
        self.assertEqual(store.items(), (warning,))
