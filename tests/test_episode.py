"""Deterministic contract checks, not a selected Phase A research environment.

The scripted adapter exists only in tests: it exercises lifecycle boundaries
without asserting anything about inheritance, utility, H1, or H3.
"""

from dataclasses import FrozenInstanceError, asdict, replace
import unittest

from gim.agent import Agent
from gim.distiller import CandidateMemory, validate_candidates
from gim.environment import StepResult
from gim.trajectory import Event, Trajectory


class ScriptedEnvironment:
    """Replay fixed responses while retaining calls for assertions."""

    def __init__(self, responses):
        self.responses = tuple(responses)
        self.actions = []
        self.seed = None

    def reset(self, seed):
        self.seed = seed
        self.actions = []
        return "initial observation"

    def step(self, action):
        response = self.responses[len(self.actions)]
        self.actions.append(action)
        return response


class RecordingPolicy:
    """Episode-local history makes incorrect policy reuse observable."""

    def __init__(self):
        self.calls = []

    def choose_action(self, observation, memories):
        self.calls.append((observation, memories))
        return f"action {len(self.calls)}"


def finished_trajectory():
    return Trajectory("episode", 0, (Event("event-1", 1, "action", "done", 1.0),), "done", 1.0)


def candidate(**changes):
    fields = dict(
        content="A fixture lesson",
        memory_type="fixture-rule",
        outcome_class="success",
        scope=("fixture",),
        event_ids=("event-1",),
    )
    fields.update(changes)
    return CandidateMemory(**fields)


class AgentEpisodeTests(unittest.TestCase):
    def run_episode(self, responses, *, max_steps=5, agent=None, queries=None):
        policy = RecordingPolicy()
        agent = Agent(policy) if agent is None else agent
        environment = ScriptedEnvironment(responses)
        retrieval_queries = [] if queries is None else queries

        def retrieve(observation):
            retrieval_queries.append(observation)
            return ()

        trajectory = agent.run(
            environment,
            seed=42,
            generation=2,
            episode_id="episode",
            retrieve=retrieve,
            max_steps=max_steps,
        )
        return trajectory, policy, environment, retrieval_queries

    def test_each_action_uses_current_public_observation_and_retrieval(self):
        trajectory, policy, environment, queries = self.run_episode(
            (
                StepResult("second observation", 0.25, False),
                StepResult("done", 1.0, True),
                StepResult("must not be reached", 10.0, True),
            )
        )
        self.assertEqual(environment.seed, 42)
        self.assertEqual(queries, ["initial observation", "second observation"])
        self.assertEqual(policy.calls, [("initial observation", ()), ("second observation", ())])
        self.assertEqual(environment.actions, ["action 1", "action 2"])
        self.assertEqual(
            [event.event_id for event in trajectory.events], ["episode:1", "episode:2"]
        )
        self.assertEqual(
            [event.observation for event in trajectory.events], ["second observation", "done"]
        )
        decision_inputs = [trajectory.initial_observation] + [
            event.observation for event in trajectory.events[:-1]
        ]
        self.assertEqual(decision_inputs, queries)
        self.assertEqual(trajectory.terminal_state, "done")
        self.assertEqual(trajectory.terminal_reward, 1.0)
        self.assertEqual(trajectory.total_reward, 1.25)
        self.assertTrue(trajectory.terminated)

    def test_first_decision_context_survives_in_one_step_evidence(self):
        trajectory, policy, _, queries = self.run_episode((StepResult("done", 1.0, True),))
        # A distiller/logger gets only this trajectory, not the live policy or
        # environment. The first decision's premise must survive that handoff.
        self.assertEqual(trajectory.initial_observation, policy.calls[0][0])
        self.assertEqual(trajectory.initial_observation, queries[0])
        self.assertEqual(asdict(trajectory)["initial_observation"], "initial observation")
        self.assertEqual(trajectory.events[0].observation, "done")
        self.assertNotEqual(trajectory.initial_observation, trajectory.events[0].observation)

    def test_equal_seed_actions_and_task_replay_exactly(self):
        responses = (StepResult("middle", 0.0, False), StepResult("done", 1.0, True))
        first = self.run_episode(responses)[0]
        second = self.run_episode(responses)[0]
        self.assertEqual(first, second)

    def test_step_limit_is_distinct_from_terminal_completion(self):
        trajectory = self.run_episode((StepResult("still running", 0.0, False),), max_steps=1)[0]
        self.assertFalse(trajectory.terminated)
        self.assertEqual(len(trajectory.events), 1)
        # Reaching a terminal state exactly at the limit is still completion.
        trajectory = self.run_episode((StepResult("done", 1.0, True),), max_steps=1)[0]
        self.assertTrue(trajectory.terminated)

    def test_same_agent_cannot_reuse_policy_context(self):
        policy = RecordingPolicy()
        agent = Agent(policy)
        responses = (StepResult("done", 1.0, True),)
        self.run_episode(responses, agent=agent)
        with self.assertRaisesRegex(RuntimeError, "fresh Agent"):
            self.run_episode(responses, agent=agent)
        self.assertEqual(len(policy.calls), 1)

    def test_failed_episode_also_requires_a_fresh_agent(self):
        agent = Agent(RecordingPolicy())
        with self.assertRaises(TypeError):
            self.run_episode(("malformed step response",), agent=agent)
        with self.assertRaises(RuntimeError):
            self.run_episode((StepResult("done", 1.0, True),), agent=agent)

    def test_zero_step_limit_is_rejected_before_acting(self):
        with self.assertRaisesRegex(ValueError, "max_steps"):
            self.run_episode((StepResult("done", 1.0, True),), max_steps=0)


class FrozenEvidenceTests(unittest.TestCase):
    def test_freeze_copies_external_list_and_freezes_members(self):
        event = Event("event-1", 1, "action", "done", 1.0)
        external_events = [event]
        trajectory = Trajectory("episode", 0, external_events, "done", 1.0)
        external_events.clear()
        self.assertEqual(trajectory.events, (event,))
        with self.assertRaises(FrozenInstanceError):
            event.reward = 2.0
        with self.assertRaises(FrozenInstanceError):
            trajectory.terminal_state = "rewritten"
        with self.assertRaises(FrozenInstanceError):
            trajectory.initial_observation = "rewritten first decision"

    def test_initial_observation_allows_empty_text_but_rejects_nontext(self):
        trajectory = finished_trajectory()
        self.assertEqual(trajectory.initial_observation, "")
        for invalid in (None, [], 1, True):
            with (
                self.subTest(value=invalid),
                self.assertRaisesRegex(ValueError, "initial_observation"),
            ):
                replace(trajectory, initial_observation=invalid)

    def test_duplicate_ids_and_noncontiguous_steps_are_rejected(self):
        event = Event("same", 1, "action", "first", 0.0)
        duplicate_id = Event("same", 2, "action", "done", 1.0)
        with self.assertRaisesRegex(ValueError, "unique"):
            Trajectory("episode", 0, (event, duplicate_id), "done", 1.0)
        skipped_step = Event("different", 3, "action", "done", 1.0)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            Trajectory("episode", 0, (event, skipped_step), "done", 1.0)

    def test_terminal_summary_cannot_disagree_with_evidence(self):
        trajectory = finished_trajectory()
        with self.assertRaisesRegex(ValueError, "terminal_state"):
            replace(trajectory, terminal_state="private ground truth")
        with self.assertRaisesRegex(ValueError, "terminal_reward"):
            replace(trajectory, terminal_reward=99.0)

    def test_nonfinite_rewards_and_nonboolean_termination_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StepResult("observation", value, True)
            with self.subTest(value=value), self.assertRaises(ValueError):
                Event("event", 1, "action", "observation", value)
        with self.assertRaises(ValueError):
            StepResult("observation", 1.0, 1)


class DistillationEvidenceTests(unittest.TestCase):
    def test_valid_batch_is_detached_and_cites_current_episode(self):
        scope = ["fixture"]
        citations = ["event-1"]
        lesson = candidate(scope=scope, event_ids=citations)
        scope.clear()
        citations.clear()
        external_batch = [lesson]
        batch = validate_candidates(finished_trajectory(), external_batch)
        external_batch.clear()
        self.assertEqual(batch, (lesson,))
        self.assertEqual(lesson.scope, ("fixture",))
        self.assertEqual(lesson.event_ids, ("event-1",))
        with self.assertRaises(FrozenInstanceError):
            lesson.content = "modified lesson"

    def test_candidate_with_unknown_citation_rejects_entire_batch(self):
        with self.assertRaisesRegex(ValueError, "unknown event IDs"):
            validate_candidates(
                finished_trajectory(), (candidate(), candidate(event_ids=("other",)))
            )

    def test_truncation_cannot_enter_write_path_even_with_empty_batch(self):
        trajectory = replace(finished_trajectory(), terminated=False)
        with self.assertRaisesRegex(ValueError, "truncated"):
            validate_candidates(trajectory, ())

    def test_required_type_scope_polarity_and_evidence_fields(self):
        for changes in (
            {"memory_type": ""},
            {"scope": "fixture"},
            {"outcome_class": "ambiguous"},
            {"event_ids": ()},
            {"event_ids": ("event-1", "event-1")},
            {"information_gain": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                candidate(**changes)

    def test_missing_information_gain_is_not_a_fake_zero_measurement(self):
        self.assertIsNone(candidate().information_gain)
        self.assertEqual(candidate(information_gain=0.0).information_gain, 0.0)

    def test_global_scope_and_failure_family_match_memory_item_contract(self):
        lesson = candidate(scope=(), outcome_class="failure", failure_family="fixture-failure")
        self.assertEqual(validate_candidates(finished_trajectory(), (lesson,)), (lesson,))
        self.assertEqual(lesson.scope, ())
        success_warning = candidate(outcome_class="success", failure_family="fixture-failure")
        self.assertEqual(
            validate_candidates(finished_trajectory(), (success_warning,)), (success_warning,)
        )


if __name__ == "__main__":
    unittest.main()
