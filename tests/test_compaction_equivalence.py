"""Differential checks against the previous serialized-copy planning path."""

from dataclasses import replace
from contextlib import nullcontext
import random
import unittest
from unittest.mock import patch

from gim.memory_store import MemoryStore
from gim.pruning_policy import BudgetInfeasible, PruningPolicy
from gim.vector_index import VectorIndex
from test_generation import fixture_item


class CompactionEquivalenceTests(unittest.TestCase):
    def test_sharing_immutable_records_matches_deep_copy_for_budgets_and_coverage(self):
        rng = random.Random(90)
        for trial in range(15):
            items = tuple(replace(fixture_item(f"m-{i}", rng.randrange(4),
                                               family="rare" if i % 3 == 0 else None),
                                  content="x" * rng.randrange(1, 600)) for i in range(12))
            for mode in ("fifo", "random"):
                for budget in (0, 12, 800, 2600, 20000):
                    stores = [MemoryStore(items), MemoryStore.from_bytes(MemoryStore(items).to_bytes())]
                    indices = [VectorIndex(store.items()) for store in stores]
                    policy = PruningPolicy(mode=mode, seed=trial,
                                           required_failure_families=frozenset({"rare"}))
                    initial = stores[0].to_bytes()
                    outcomes = []
                    for number, (store, index) in enumerate(zip(stores, indices)):
                        # Reproduce the old planning allocation: every constructed
                        # planning store is round-tripped through the real decoder.
                        context = patch(
                            "gim.pruning_policy.MemoryStore",
                            side_effect=lambda records: MemoryStore.from_bytes(
                                MemoryStore(records).to_bytes()),
                        ) if number else nullcontext()
                        try:
                            with context:
                                outcomes.append(policy.enforce(store, index, budget))
                        except BudgetInfeasible:
                            outcomes.append("infeasible")
                            self.assertEqual(store.to_bytes(), initial)
                            index.assert_matches(store.items())
                    self.assertEqual(outcomes[0], outcomes[1])
                    self.assertEqual(stores[0].to_bytes(), stores[1].to_bytes())
                    self.assertEqual(indices[0].to_bytes(), indices[1].to_bytes())

    def test_failed_plan_preserves_original_container_and_nested_records(self):
        item = fixture_item("protected", 0, family="rare")
        store = MemoryStore((item,))
        index = VectorIndex(store.items())
        policy = PruningPolicy(mode="fifo", seed=0, required_failure_families=frozenset({"rare"}))
        original = store.to_bytes(), index.to_bytes()
        with self.assertRaises(BudgetInfeasible):
            policy.enforce(store, index, 12)
        self.assertEqual((store.to_bytes(), index.to_bytes()), original)
        self.assertIs(store.items()[0], item)

    def test_planning_does_not_decode_or_duplicate_authoritative_vectors(self):
        items = (fixture_item("a", 0), fixture_item("b", 1))
        store, index = MemoryStore(items), VectorIndex(items)
        policy = PruningPolicy(mode="fifo", seed=0, required_failure_families=frozenset())
        budget = MemoryStore((items[1],)).bytes() + VectorIndex((items[1],)).overhead_bytes()
        with patch.object(MemoryStore, "from_bytes", side_effect=AssertionError("redundant decode")):
            events = policy.enforce(store, index, budget)
        self.assertEqual([event.item_id for event in events], ["a"])
        self.assertIs(store.items()[0], items[1])
        self.assertIs(store.items()[0].vector, items[1].vector)


if __name__ == "__main__":
    unittest.main()
