"""End-to-end contract fixtures, explicitly not the Phase A research task."""

from dataclasses import asdict, replace
import gc
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import weakref

from gim.config import RunConfig
from gim.distiller import CandidateMemory
from gim.embedder import NormalizingEmbedder
from gim.environment import StepResult
from gim.experiment_logger import ExperimentLogger, NullLogger
from gim.generation_runner import run_generation
from gim.memory_item import MemoryItem, Provenance
from gim.memory_store import MemoryStore, RECORD_HEADER, STORE_HEADER
from gim.pruning_policy import BudgetInfeasible, PruningPolicy
from gim.trajectory import Trajectory
from gim.vector_index import VectorIndex


def fixture_item(
    name: str, generation: int, *, family: str | None = None, content: str = "fixture assertion"
) -> MemoryItem:
    return MemoryItem(
        name,
        content,
        "fixture",
        "failure" if family else "success",
        ("fixture",),
        (Provenance(name, ("event",), 0.0),),
        (1.0, 0.0),
        "fixture-v1",
        generation,
        failure_family=family,
    )


class OneStepFixture:
    def reset(self, seed: int) -> str:
        return "fixture observation"

    def step(self, action: str) -> StepResult:
        return StepResult("fixture terminal", float(action == "use-memory"), True)


class FixturePolicy:
    def choose_action(self, observation: str, memories: tuple[MemoryItem, ...]) -> str:
        return "use-memory" if memories else "no-memory"


class FixtureDistiller:
    def __init__(self) -> None:
        self.calls = 0

    def distill(self, trajectory: Trajectory) -> tuple[CandidateMemory, ...]:
        self.calls += 1
        return (
            CandidateMemory(
                "fixture assertion",
                "fixture",
                "success",
                ("fixture",),
                (trajectory.events[0].event_id,),
            ),
        )


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = NormalizingEmbedder("fixture-v1", 2, lambda text: (1.0, 0.0))
        self.config = RunConfig(
            10_000,
            2,
            1,
            1,
            "fixture-v1",
            "fixture-policy-v1",
            "fixture-environment-v1",
            "fixture-distiller-v1",
        )
        self.pruning = PruningPolicy(mode="fifo", seed=17, required_failure_families=frozenset())

    def test_successor_reads_only_serialized_memory_and_policy_is_released(self) -> None:
        references: list[weakref.ReferenceType[FixturePolicy]] = []

        def factory() -> FixturePolicy:
            policy = FixturePolicy()
            references.append(weakref.ref(policy))
            return policy

        distiller = FixtureDistiller()
        first = run_generation(
            config=self.config,
            memory_bytes=b"",
            episode_id="first",
            generation=0,
            seed=17,
            scope=("fixture",),
            policy_factory=factory,
            environment_factory=OneStepFixture,
            embedder=self.embedder,
            distiller=distiller,
            pruning=self.pruning,
            logger=NullLogger(),
        )
        second = run_generation(
            config=self.config,
            memory_bytes=first.memory_bytes,
            episode_id="second",
            generation=1,
            seed=17,
            scope=("fixture",),
            policy_factory=factory,
            environment_factory=OneStepFixture,
            embedder=self.embedder,
            distiller=distiller,
            pruning=self.pruning,
            logger=NullLogger(),
        )
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(distiller.calls, 2)  # one terminal batch per completed episode
        self.assertEqual(first.trajectory.terminal_reward, 0.0)
        self.assertEqual(second.trajectory.terminal_reward, 1.0)
        persisted = MemoryStore.from_bytes(second.memory_bytes)
        self.assertEqual(
            second.boundary_bytes,
            persisted.bytes() + VectorIndex(persisted.items()).overhead_bytes(),
        )
        self.assertLessEqual(second.boundary_bytes, self.config.budget_bytes)

    def test_zero_byte_control_does_not_distill_or_embed(self) -> None:
        class ForbiddenDistiller:
            def distill(self, trajectory: Trajectory) -> tuple[CandidateMemory, ...]:
                raise AssertionError("control must not distill")

        def forbidden_encoder(text: str) -> tuple[float, ...]:
            raise AssertionError("control must not embed")

        result = run_generation(
            config=replace(self.config, budget_bytes=0),
            memory_bytes=b"",
            episode_id="control",
            generation=0,
            seed=17,
            scope=(),
            policy_factory=FixturePolicy,
            environment_factory=OneStepFixture,
            embedder=NormalizingEmbedder("fixture-v1", 2, forbidden_encoder),
            distiller=ForbiddenDistiller(),
            pruning=self.pruning,
            logger=NullLogger(),
        )
        self.assertEqual(
            (result.memory_bytes, result.peak_bytes, result.boundary_bytes), (b"", 0, 0)
        )

    def test_blank_observation_is_a_valid_episode_without_a_query(self) -> None:
        class EmptyObservation(OneStepFixture):
            def reset(self, seed: int) -> str:
                return ""

        result = run_generation(
            config=self.config,
            memory_bytes=b"",
            episode_id="blank",
            generation=0,
            seed=17,
            scope=(),
            policy_factory=FixturePolicy,
            environment_factory=EmptyObservation,
            embedder=self.embedder,
            distiller=FixtureDistiller(),
            pruning=self.pruning,
            logger=NullLogger(),
        )
        self.assertTrue(result.trajectory.terminated)

    def test_oversized_inheritance_is_rejected_before_acting(self) -> None:
        original = MemoryStore((fixture_item("old", 0),)).to_bytes()

        def forbidden_factory() -> FixturePolicy:
            raise AssertionError("oversized inheritance must be rejected before construction")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with self.assertRaises(BudgetInfeasible):
                run_generation(
                    config=replace(self.config, budget_bytes=1),
                    memory_bytes=original,
                    episode_id="too-small",
                    generation=1,
                    seed=17,
                    scope=("fixture",),
                    policy_factory=forbidden_factory,
                    environment_factory=OneStepFixture,
                    embedder=self.embedder,
                    distiller=FixtureDistiller(),
                    pruning=self.pruning,
                    logger=ExperimentLogger(path),
                )
            self.assertEqual(MemoryStore.from_bytes(original).items()[0].item_id, "old")
            self.assertFalse(path.exists())

    def test_padded_file_cannot_evade_budget_by_normalization(self) -> None:
        original = MemoryStore((fixture_item("old", 0),))
        index = VectorIndex(original.items())
        cap = original.bytes() + index.overhead_bytes()
        encoded = original.to_bytes()
        metadata_length = RECORD_HEADER.unpack_from(encoded, STORE_HEADER.size)[0]
        padded = (
            encoded[: STORE_HEADER.size]
            + RECORD_HEADER.pack(metadata_length + 1000)
            + b" " * 1000
            + encoded[STORE_HEADER.size + RECORD_HEADER.size :]
        )

        def forbidden_factory() -> FixturePolicy:
            raise AssertionError("noncanonical input must be rejected before acting")

        with self.assertRaisesRegex(ValueError, "canonical"):
            run_generation(
                config=replace(self.config, budget_bytes=cap),
                memory_bytes=padded,
                episode_id="padded",
                generation=1,
                seed=17,
                scope=("fixture",),
                policy_factory=forbidden_factory,
                environment_factory=OneStepFixture,
                embedder=self.embedder,
                distiller=FixtureDistiller(),
                pruning=self.pruning,
                logger=NullLogger(),
            )

    def test_online_cost_excludes_agent_environment_and_audit_io(self) -> None:
        # A controlled clock attributes known costs without sleeps or wall-clock
        # thresholds. Expensive agent/environment/log operations must not appear
        # in the memory-management total used to compare retention policies.
        class Clock:
            now = 0

            def read(self) -> int:
                return self.now

            def advance(self, amount: int) -> None:
                self.now += amount

        clock = Clock()
        records: list[tuple[str, dict[str, object]]] = []

        class TimedLogger:
            def record(self, kind: str, **fields: object) -> None:
                clock.advance(500)
                records.append((kind, fields))

        class TimedPolicy(FixturePolicy):
            def choose_action(self, observation: str, memories: tuple[MemoryItem, ...]) -> str:
                clock.advance(1000)
                return super().choose_action(observation, memories)

        class TimedEnvironment(OneStepFixture):
            def reset(self, seed: int) -> str:
                clock.advance(200)
                return super().reset(seed)

            def step(self, action: str) -> StepResult:
                clock.advance(300)
                return super().step(action)

        class TimedDistiller(FixtureDistiller):
            def distill(self, trajectory: Trajectory) -> tuple[CandidateMemory, ...]:
                clock.advance(5)
                return super().distill(trajectory)

        def encoder(text: str) -> tuple[float, ...]:
            clock.advance(3)
            return (1.0, 0.0)

        result = run_generation(
            config=self.config,
            memory_bytes=b"",
            episode_id="timed",
            generation=0,
            seed=17,
            scope=("fixture",),
            policy_factory=TimedPolicy,
            environment_factory=TimedEnvironment,
            embedder=NormalizingEmbedder("fixture-v1", 2, encoder),
            distiller=TimedDistiller(),
            pruning=self.pruning,
            logger=TimedLogger(),
            clock_ns=clock.read,
        )
        self.assertEqual(result.online_cost.setup_ns, 0)
        self.assertEqual(result.online_cost.retrieval_ns, 3)
        self.assertEqual(result.online_cost.write_path_ns, 8)
        self.assertEqual(result.online_cost.total_ns, 11)
        started = next(fields for _, fields in records if fields.get("status") == "started")
        self.assertEqual(started["input_memory_sha256"], sha256(b"").hexdigest())
        self.assertEqual(started["config"], asdict(self.config))
        self.assertEqual(started["scope"], ["fixture"])
        self.assertEqual(
            started["pruning"], {"mode": "fifo", "seed": 17, "required_failure_families": []}
        )
        ready = next(fields for _, fields in records if fields.get("status") == "boundary_ready")
        self.assertEqual(ready["online_management_ns"], 11)
        self.assertIsNone(ready["offline_labeling_ns"])
        self.assertEqual(ready["output_memory_sha256"], sha256(result.memory_bytes).hexdigest())
        insert = next(
            fields
            for kind, fields in records
            if kind == "memory_event" and fields.get("action") == "insert"
        )
        self.assertEqual(insert["memory"]["provenance"][0]["episode_id"], "timed")

    def test_inherited_revision_is_checked_even_with_no_query(self) -> None:
        original = MemoryStore(
            (replace(fixture_item("old", 0), embedding_revision="wrong-v1"),)
        ).to_bytes()
        with self.assertRaisesRegex(ValueError, "inherited memory"):
            run_generation(
                config=self.config,
                memory_bytes=original,
                episode_id="revision",
                generation=1,
                seed=17,
                scope=(),
                policy_factory=FixturePolicy,
                environment_factory=OneStepFixture,
                embedder=self.embedder,
                distiller=FixtureDistiller(),
                pruning=self.pruning,
                logger=NullLogger(),
            )

    def test_failed_coverage_keeps_auditable_trajectory_without_valid_boundary(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            impossible = PruningPolicy(
                mode="fifo", seed=17, required_failure_families=frozenset({"missing"})
            )
            with self.assertRaises(BudgetInfeasible):
                run_generation(
                    config=self.config,
                    memory_bytes=b"",
                    episode_id="coverage",
                    generation=0,
                    seed=17,
                    scope=(),
                    policy_factory=FixturePolicy,
                    environment_factory=OneStepFixture,
                    embedder=self.embedder,
                    distiller=FixtureDistiller(),
                    pruning=impossible,
                    logger=ExperimentLogger(path),
                )
            text = path.read_text(encoding="utf-8")
            self.assertIn('"event_id":"coverage:1"', text)
            self.assertNotIn('"status":"boundary_ready"', text)
            records = [json.loads(line) for line in text.splitlines()]
            failed = next(record for record in records if record.get("status") == "boundary_failed")
            self.assertGreater(failed["peak_bytes"], 0)
            self.assertIsNone(failed["offline_labeling_ns"])


class PruningTests(unittest.TestCase):
    def test_fifo_counts_index_and_preserves_immutable_provenance(self) -> None:
        old, new = fixture_item("old", 0), fixture_item("new", 1)
        store = MemoryStore((old, new))
        index = VectorIndex(store.items())
        budget = MemoryStore((new,)).bytes() + VectorIndex((new,)).overhead_bytes()
        policy = PruningPolicy(mode="fifo", seed=1, required_failure_families=frozenset())
        evictions = policy.enforce(store, index, budget)
        self.assertEqual([event.item_id for event in evictions], ["old"])
        self.assertEqual(store.items(), (new,))
        self.assertEqual(store.items()[0].provenance, new.provenance)
        self.assertEqual(store.bytes() + index.overhead_bytes(), budget)

    def test_coverage_chooses_a_feasible_small_witness(self) -> None:
        small = fixture_item("small", 0, family="rare")
        large = fixture_item("large", 1, family="rare", content="long" * 100)
        store = MemoryStore((small, large))
        index = VectorIndex(store.items())
        budget = MemoryStore((small,)).bytes() + VectorIndex((small,)).overhead_bytes()
        policy = PruningPolicy(mode="fifo", seed=1, required_failure_families=frozenset({"rare"}))
        policy.enforce(store, index, budget)
        self.assertEqual(store.items(), (small,))

    def test_infeasible_coverage_leaves_store_and_index_unchanged(self) -> None:
        store = MemoryStore((fixture_item("rare", 0, family="rare"), fixture_item("extra", 1)))
        index = VectorIndex(store.items())
        before, overhead = store.to_bytes(), index.overhead_bytes()
        policy = PruningPolicy(mode="fifo", seed=1, required_failure_families=frozenset({"rare"}))
        with self.assertRaises(BudgetInfeasible):
            policy.enforce(store, index, 1)
        self.assertEqual(store.to_bytes(), before)
        self.assertEqual(index.overhead_bytes(), overhead)

    def test_random_baseline_replays_independent_of_insertion_order(self) -> None:
        items = tuple(fixture_item(str(number), number) for number in range(8))
        policy = PruningPolicy(mode="random", seed=123, required_failure_families=frozenset())
        outcomes = []
        for order in (items, tuple(reversed(items))):
            store = MemoryStore(order)
            index = VectorIndex(store.items())
            outcomes.append(policy.enforce(store, index, 2000))
        self.assertEqual(outcomes[0], outcomes[1])
