"""Foundation checks using explicit test vectors, never experimental evidence."""

from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from gim.embedder import NormalizingEmbedder
from gim.memory_item import MemoryItem, Provenance
from gim.memory_store import FLOAT32_BYTES, RECORD_HEADER, STORE_HEADER, STORE_MAGIC, MemoryStore
from gim.retrieval_policy import RetrievalPolicy, RetrievalResult
from gim.vector_index import VectorIndex


def item(item_id="memory-a", *, outcome="success", vector=(1.0, 0.0), scope=("task-a",), **kwargs):
    """Build transparent fixture records; these are not a scientific task suite."""
    return MemoryItem(
        item_id=item_id,
        content="An explicitly supplied test statement.",
        memory_type="test-instruction",
        outcome_class=outcome,
        scope=scope,
        provenance=(Provenance("episode-1", ("event-1",), 1.0),),
        vector=vector,
        embedding_revision="test-fixed-v1",
        created_generation=0,
        **kwargs,
    )


class FixedEmbedder:
    """Known dense coordinates for tests only; not a semantic embedding model."""

    revision = "test-fixed-v1"
    dimension = 2

    def embed(self, text):
        return {"east": (1.0, 0.0), "north": (0.0, 1.0)}[text]


class MemoryItemTests(unittest.TestCase):
    def test_coordinates_are_binary32_before_first_retrieval(self):
        original = (0.6, 0.8)
        record = item(vector=original)
        expected = struct.unpack("<2f", struct.pack("<2f", *original))
        self.assertEqual(record.vector, expected)
        self.assertNotEqual(record.vector, original)
        # Rounding a unit vector is allowed; normalization is not repeatedly
        # applied, since that would move coordinates away from binary32 values.
        self.assertEqual(replace(record).vector, expected)
        with self.assertRaises(ValueError):
            item(vector=(0.600001, 0.8))

    def test_nested_containers_are_copied_and_immutable(self):
        event_ids = ["event-1"]
        scope = ["task-a"]
        vector = [1.0, 0.0]
        provenance = [Provenance("episode-1", event_ids, 1)]
        record = replace(item(), scope=scope, vector=vector, provenance=provenance)
        event_ids.append("event-2")
        scope.append("task-b")
        vector[0] = 0.0
        provenance.clear()
        self.assertEqual(record.scope, ("task-a",))
        self.assertEqual(record.vector, (1.0, 0.0))
        self.assertEqual(record.provenance[0].event_ids, ("event-1",))
        with self.assertRaises(FrozenInstanceError):
            record.content = "changed"
        with self.assertRaises(FrozenInstanceError):
            record.provenance[0].source_reward = 0

    def test_vector_validation(self):
        for vector in ((), (0.0, 0.0), (3.0, 4.0), (math.nan, 0), (math.inf, 0), (True, 0), "10"):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                item(vector=vector)

    def test_provenance_is_required(self):
        with self.assertRaises(ValueError):
            replace(item(), provenance=())
        for events in ((), ("event-1", "event-1"), "event-1"):
            with self.subTest(events=events), self.assertRaises(ValueError):
                Provenance("episode-1", events, 1)
        with self.assertRaises(ValueError):
            Provenance("episode-1", ("event-1",), float("nan"))

    def test_unknown_information_gain_stays_unknown(self):
        self.assertIsNone(item().information_gain)
        self.assertEqual(item(information_gain=0).information_gain, 0.0)
        with self.assertRaises(ValueError):
            item(information_gain=float("inf"))

    def test_outcome_and_source_reward_are_separate_fields(self):
        failure = item(outcome="failure", failure_family="family-a")
        self.assertEqual(failure.provenance[0].source_reward, 1.0)
        self.assertEqual(failure.outcome_class, "failure")
        success_warning = item(failure_family="family-a")
        self.assertEqual(success_warning.outcome_class, "success")
        self.assertEqual(success_warning.failure_family, "family-a")
        restored = MemoryStore.from_bytes(MemoryStore((success_warning,)).to_bytes())
        self.assertEqual(restored.items(), (success_warning,))
        with self.assertRaises(ValueError):
            item(outcome="unknown")

    def test_generation_is_an_integer_not_bool(self):
        for generation in (-1, 0.5, True):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                replace(item(), created_generation=generation)


class EmbedderTests(unittest.TestCase):
    def test_explicit_encoder_is_normalized(self):
        embedder = NormalizingEmbedder(
            "weights-and-preprocessing@abc123", 2, lambda text: (3.0, 4.0)
        )
        self.assertEqual(embedder.embed("example"), (0.6, 0.8))
        self.assertEqual(embedder.revision, "weights-and-preprocessing@abc123")

    def test_bad_encoder_output_is_rejected(self):
        for output in ((0, 0), (float("nan"), 1), (1, 0, 0)):
            embedder = NormalizingEmbedder("explicit-test-revision", 2, lambda text: output)
            with self.subTest(output=output), self.assertRaises(ValueError):
                embedder.embed("example")

    def test_empty_revision_and_dimension_are_rejected(self):
        with self.assertRaises(ValueError):
            NormalizingEmbedder("", 2, lambda text: (1, 0))
        with self.assertRaises(ValueError):
            NormalizingEmbedder("test-revision", True, lambda text: (1, 0))


class MemoryStoreTests(unittest.TestCase):
    def test_384_coordinates_occupy_exactly_1536_bytes_once(self):
        coordinates = (1.0 / math.sqrt(384),) * 384
        record = item(vector=coordinates)
        store = MemoryStore((record,))
        data = store.to_bytes()
        marker, count, dimension = STORE_HEADER.unpack_from(data)
        self.assertEqual((marker, count, dimension), (b"GMS2", 1, 384))
        metadata_length = RECORD_HEADER.unpack_from(data, STORE_HEADER.size)[0]
        vector_start = STORE_HEADER.size + RECORD_HEADER.size + metadata_length
        metadata = json.loads(data[STORE_HEADER.size + RECORD_HEADER.size : vector_start])
        self.assertNotIn("vector", metadata)
        self.assertEqual(len(data[vector_start:]), 384 * FLOAT32_BYTES)
        self.assertEqual(store.vector_bytes(), 1536)
        self.assertEqual(data[vector_start:], struct.pack("<384f", *record.vector))
        self.assertEqual(store.bytes(), 12 + 4 + metadata_length + 1536)
        self.assertEqual(MemoryStore.from_bytes(data).to_bytes(), data)
        # More dimensions add only their four-byte coordinates, never expanded
        # JSON floats or a second array of vector sums/representative samples.
        self.assertEqual(store.bytes() - MemoryStore((item(),)).bytes(), (384 - 2) * 4)

    def test_roundtrip_is_deterministic_and_matches_actual_disk_bytes(self):
        first = replace(item("z"), content="café and 雪", information_gain=0.125)
        second = item("a", outcome="failure", failure_family="rare")
        store = MemoryStore((first, second))
        reversed_store = MemoryStore((second, first))
        self.assertEqual(store.to_bytes(), reversed_store.to_bytes())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "memory.gms"
            store.save(path)
            self.assertEqual(store.bytes(), path.stat().st_size)
            self.assertEqual(store.to_bytes(), path.read_bytes())
            restored = MemoryStore.load(path)
            self.assertEqual(restored.items(), store.items())
            self.assertEqual(restored.to_bytes(), store.to_bytes())

    def test_snapshot_is_detached_from_writer_and_has_no_mutation_methods(self):
        store = MemoryStore((item("a"),))
        snapshot = store.snapshot()
        store.insert(item("b"))
        store.delete("a")
        self.assertEqual(tuple(record.item_id for record in snapshot.items()), ("a",))
        self.assertEqual(tuple(record.item_id for record in store.items()), ("b",))
        self.assertFalse(hasattr(snapshot, "insert"))
        self.assertFalse(hasattr(snapshot, "delete"))
        self.assertFalse(hasattr(snapshot, "_records"))
        with self.assertRaises(FrozenInstanceError):
            snapshot._items = ()

    def test_duplicate_ids_and_incompatible_embeddings_are_rejected(self):
        store = MemoryStore((item(),))
        alternatives = (
            item(),
            replace(item("other"), embedding_revision="revision-2"),
            item("other", vector=(1.0, 0.0, 0.0)),
        )
        for alternative in alternatives:
            with self.subTest(alternative=alternative), self.assertRaises(ValueError):
                store.insert(alternative)
        self.assertEqual(len(store.items()), 1)

    def test_corrupt_storage_is_rejected(self):
        data = MemoryStore((item(),)).to_bytes()
        metadata_length = RECORD_HEADER.unpack_from(data, STORE_HEADER.size)[0]
        start = STORE_HEADER.size + RECORD_HEADER.size
        metadata = json.loads(data[start : start + metadata_length])

        def encoded_metadata(value):
            encoded = json.dumps(value).encode()
            return (
                STORE_HEADER.pack(STORE_MAGIC, 1, 2)
                + RECORD_HEADER.pack(len(encoded))
                + encoded
                + struct.pack("<2f", 1, 0)
            )

        unknown_field = dict(metadata, unbudgeted_hidden_history=["secret"])
        missing_field = dict(metadata)
        del missing_field["content"]
        invalid_metadata = (
            [],
            None,
            unknown_field,
            missing_field,
            dict(metadata, vector=[1, 0]),
            dict(metadata, scope={"task-a": True}),
            dict(metadata, provenance={}),
            dict(metadata, provenance=[None]),
            dict(metadata, provenance=[dict(metadata["provenance"][0], event_ids={"e": 1})]),
            dict(metadata, provenance=[dict(metadata["provenance"][0], source_reward="1")]),
            dict(metadata, created_generation=True),
            dict(metadata, information_gain=float("nan")),
        )
        duplicate_json = b'{"content":"a","content":"b"}'
        duplicate_fields = (
            STORE_HEADER.pack(STORE_MAGIC, 1, 2)
            + RECORD_HEADER.pack(len(duplicate_json))
            + duplicate_json
            + struct.pack("<2f", 1, 0)
        )
        frame = data[STORE_HEADER.size :]
        duplicate_ids = STORE_HEADER.pack(STORE_MAGIC, 2, 2) + frame + frame
        for data in (
            b"not-a-store",
            b"\xff",
            STORE_HEADER.pack(b"GMS3", 0, 0),
            STORE_HEADER.pack(STORE_MAGIC, 0, 2),
            STORE_HEADER.pack(STORE_MAGIC, 1, 0),
            STORE_HEADER.pack(STORE_MAGIC, 0xFFFFFFFF, 0xFFFFFFFF),
            STORE_HEADER.pack(STORE_MAGIC, 1, 2) + RECORD_HEADER.pack(0xFFFFFFFF) + frame,
            data + b"trailing",
            data[:-8] + struct.pack("<2f", 0, 0),
            data[:-8] + struct.pack("<2f", float("nan"), 0),
            data[:-8] + struct.pack("<2f", float("inf"), 0),
            data[:-8] + struct.pack("<2f", 2, 0),
            data[:start] + b"\xff" + data[start + 1 :],
            duplicate_fields,
            duplicate_ids,
            *(encoded_metadata(value) for value in invalid_metadata),
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                MemoryStore.from_bytes(data)

    def test_every_truncation_is_rejected(self):
        data = MemoryStore((item("a"), item("b"))).to_bytes()
        for length in range(len(data)):
            with self.subTest(length=length), self.assertRaises(ValueError):
                MemoryStore.from_bytes(data[:length])

    def test_legacy_json_requires_explicit_migration(self):
        for data in (b'{"schema_version":1,"items":[]}', b' \n{"schema_version":1}', b"[]"):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "legacy.*migration"):
                MemoryStore.from_bytes(data)

    def test_failed_atomic_replace_keeps_previous_file_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.gms"
            old = MemoryStore((item("old"),))
            old.save(path)
            with patch("gim.memory_store.os.replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    MemoryStore((item("new"),)).save(path)
            self.assertEqual(path.read_bytes(), old.to_bytes())
            self.assertEqual(tuple(Path(directory).iterdir()), (path,))

    def test_empty_envelope_and_all_retained_fields_are_counted(self):
        empty = MemoryStore()
        self.assertEqual(empty.bytes(), len(empty.to_bytes()))
        self.assertGreater(empty.bytes(), 0)
        self.assertEqual(empty.bytes(), 12)
        self.assertEqual(empty.vector_bytes(), 0)
        short = MemoryStore((item(),))
        longer = MemoryStore((replace(item(), content=item().content + "é"),))
        self.assertEqual(longer.bytes() - short.bytes(), len("é".encode("utf-8")))
        with_provenance = replace(
            item(), provenance=item().provenance + (Provenance("episode-2", ("event-7",), 0),)
        )
        self.assertGreater(MemoryStore((with_provenance,)).bytes(), short.bytes())


class VectorIndexTests(unittest.TestCase):
    def test_exact_cosine_has_stable_ties(self):
        index = VectorIndex((item("z"), item("a"), item("north", vector=(0.0, 1.0))))
        hits = index.search((1.0, 0.0), "test-fixed-v1", k=3)
        self.assertEqual(tuple(hit.item_id for hit in hits), ("a", "z", "north"))
        self.assertEqual(tuple(hit.score for hit in hits), (1.0, 1.0, 0.0))

    def test_revision_and_dimension_checked_even_with_empty_filter_or_quota(self):
        index = VectorIndex((item(),))
        for vector, revision in (
            ((1.0, 0.0), "other-revision"),
            ((1.0, 0.0, 0.0), "test-fixed-v1"),
        ):
            with self.subTest(vector=vector, revision=revision), self.assertRaises(ValueError):
                index.search(vector, revision, k=0, eligible_ids=())

    def test_actual_mapping_contains_ids_revision_but_shares_coordinates(self):
        record = item("雪")
        index = VectorIndex((record,))
        expected = 16 + len(record.embedding_revision.encode()) + 4 + len(record.item_id.encode())
        self.assertEqual(index.overhead_bytes(), expected)
        self.assertEqual(index.overhead_bytes(), len(index.to_bytes()))
        self.assertEqual(VectorIndex().overhead_bytes(), 0)
        self.assertIn(record.item_id.encode(), index.to_bytes())
        self.assertIn(record.embedding_revision.encode(), index.to_bytes())
        self.assertIs(next(index._records())[1], record.vector)
        longer = replace(record, vector=(1.0,) + (0.0,) * 383)
        self.assertEqual(VectorIndex((longer,)).overhead_bytes(), expected)
        self.assertIs(next(VectorIndex((longer,))._records())[1], longer.vector)
        # Only dimension in the header changes; there is no coordinate payload.
        self.assertEqual(index.to_bytes()[8:], VectorIndex((longer,)).to_bytes()[8:])

    def test_rounded_vectors_use_actual_cosine_and_rank_identically_after_reload(self):
        angle = math.atan2(0.8, 0.6)
        records = (
            item("z", vector=(math.cos(angle + 1e-10), math.sin(angle + 1e-10))),
            item("a", vector=(math.cos(angle), math.sin(angle))),
            item("far", vector=(0, 1)),
        )
        store = MemoryStore(records)
        index = VectorIndex(store.items())
        query = (0.6, 0.8)
        hits = index.search(query, "test-fixed-v1", k=3)
        self.assertEqual(tuple(hit.item_id for hit in hits), ("a", "z", "far"))
        restored = MemoryStore.from_bytes(store.to_bytes())
        self.assertEqual(hits, VectorIndex(restored.items()).search(query, "test-fixed-v1", k=3))
        rounded = records[0].vector
        expected = math.fsum(a * b for a, b in zip(query, rounded)) / (
            math.hypot(*query) * math.hypot(*rounded)
        )
        self.assertEqual(hits[0].score, min(1, expected))

    def test_stale_check_compares_membership_space_and_shared_or_equal_vectors(self):
        record = item()
        index = VectorIndex((record,))
        index.assert_matches((record,))
        index.assert_matches(MemoryStore.from_bytes(MemoryStore((record,)).to_bytes()).items())
        # Content is resolved from the snapshot, so a content-only change does
        # not invalidate the coordinate mapping or require duplicating content.
        index.assert_matches((replace(record, content="Updated retained text."),))
        for changed in (
            (),
            (record, record),
            (replace(record, item_id="new"),),
            (replace(record, embedding_revision="another"),),
            (replace(record, vector=(0.0, 1.0)),),
            (replace(record, vector=(1.0, 0.0, 0.0)),),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "stale"):
                index.assert_matches(changed)
        # Retrieval must not rebuild/repack every vector to check freshness.
        with patch("gim.vector_index.struct.pack", side_effect=AssertionError("repacked")):
            index.assert_matches((record,))
            index.search((1, 0), "test-fixed-v1", k=1)

    def test_filter_precedes_top_k(self):
        index = VectorIndex((item("high"), item("low", vector=(0, 1))))
        hits = index.search((1.0, 0.0), "test-fixed-v1", k=1, eligible_ids=("low",))
        self.assertEqual(tuple(hit.item_id for hit in hits), ("low",))

    def test_rebuild_removes_stale_ids_and_rejects_mixed_space(self):
        index = VectorIndex((item("old"),))
        index.rebuild((item("new"),))
        with self.assertRaises(ValueError):
            index.rebuild((item("new"), replace(item("bad"), embedding_revision="different")))
        hits = index.search((1.0, 0.0), "test-fixed-v1", k=20)
        self.assertEqual(tuple(hit.item_id for hit in hits), ("new",))


class RetrievalTests(unittest.TestCase):
    def test_result_rejects_mutable_duck_typed_records_and_invalid_scores(self):
        class MutableRecord:
            outcome_class = "success"
            item_id = "mutable"

        with self.assertRaises(ValueError):
            RetrievalResult(positive=(MutableRecord(),), positive_scores=(1.0,))
        for score in ("1", [1.0], {}, True, float("nan"), float("inf"), 1.01, -1.01):
            with self.subTest(score=score), self.assertRaises(ValueError):
                RetrievalResult(positive=(item(),), positive_scores=(score,))

    def test_result_copies_score_containers_and_rejects_duplicate_ids(self):
        scores = [1]
        records = [item()]
        result = RetrievalResult(positive=records, positive_scores=scores)
        records.clear()
        scores[0] = 0
        self.assertEqual(result.positive_scores, (1.0,))
        self.assertEqual(len(result.positive), 1)
        self.assertIsInstance(result.positive_scores[0], float)
        with self.assertRaises(FrozenInstanceError):
            result.positive_scores = (0.0,)
        with self.assertRaises(ValueError):
            RetrievalResult(positive=(item(), item()), positive_scores=(1.0, 1.0))

    def test_polarity_quotas_do_not_compete_or_merge(self):
        store = MemoryStore(
            (
                item("success-1"),
                item("success-2"),
                item("failure", outcome="failure", vector=(0, 1)),
            )
        )
        before = store.to_bytes()
        result = RetrievalPolicy(1, 1).retrieve(
            store.snapshot(), VectorIndex(store.items()), (1.0, 0.0), "test-fixed-v1", ("task-a",)
        )
        self.assertEqual(tuple(record.item_id for record in result.positive), ("success-1",))
        self.assertEqual(tuple(record.item_id for record in result.negative), ("failure",))
        self.assertEqual(result.positive_scores, (1.0,))
        self.assertEqual(result.negative_scores, (0.0,))
        self.assertEqual(store.to_bytes(), before)

    def test_unused_quota_is_not_reallocated(self):
        store = MemoryStore((item("a"), item("b")))
        result = RetrievalPolicy(1, 8).retrieve(
            store.snapshot(), VectorIndex(store.items()), (1.0, 0.0), "test-fixed-v1", ("task-a",)
        )
        self.assertEqual(len(result.positive), 1)
        self.assertEqual(result.negative, ())

    def test_scope_and_type_filtering(self):
        store = MemoryStore(
            (
                item("a", scope=("task-b",)),
                item("b", scope=()),
                replace(item("c"), memory_type="other-type"),
                item("d"),
            )
        )
        index = VectorIndex(store.items())
        result = RetrievalPolicy(9, 9).retrieve(
            store.snapshot(), index, (1.0, 0.0), "test-fixed-v1", ("task-a",), ("test-instruction",)
        )
        self.assertEqual(tuple(record.item_id for record in result.positive), ("b", "d"))
        globally_applicable = RetrievalPolicy(9, 9).retrieve(
            store.snapshot(), index, (1.0, 0.0), "test-fixed-v1"
        )
        self.assertEqual(tuple(record.item_id for record in globally_applicable.positive), ("b",))

    def test_stale_index_is_rejected(self):
        store = MemoryStore((item("old"),))
        index = VectorIndex(store.items())
        store.delete("old")
        store.insert(item("new"))
        with self.assertRaisesRegex(ValueError, "stale"):
            RetrievalPolicy(1, 1).retrieve(
                store.snapshot(), index, (1.0, 0.0), "test-fixed-v1", ("task-a",)
            )

    def test_zero_quotas_return_nothing_without_modifying_store(self):
        store = MemoryStore((item(),))
        before = store.to_bytes()
        result = RetrievalPolicy(0, 0).retrieve(
            store.snapshot(), VectorIndex(store.items()), (1.0, 0.0), "test-fixed-v1", ("task-a",)
        )
        self.assertEqual(result.positive + result.negative, ())
        self.assertEqual(store.to_bytes(), before)

    def test_writer_cannot_be_passed_as_retrieval_snapshot(self):
        with self.assertRaises(ValueError):
            RetrievalPolicy(1, 1).retrieve(
                MemoryStore(), VectorIndex(), (1.0, 0.0), "test-fixed-v1"
            )

    def test_query_model_change_is_rejected_even_when_scope_matches_nothing(self):
        store = MemoryStore((item(),))
        with self.assertRaises(ValueError):
            RetrievalPolicy(0, 0).retrieve(
                store.snapshot(),
                VectorIndex(store.items()),
                (1.0, 0.0),
                "different-model",
                ("not-applicable",),
            )


if __name__ == "__main__":
    unittest.main()
