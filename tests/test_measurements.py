"""Independent arithmetic examples and logger failure cases."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gim.config import RunConfig
from gim.cli import main
from gim.memory_item import MemoryItem, Provenance
from gim.memory_store import MemoryStore, RECORD_HEADER, STORE_HEADER
from gim.vector_index import VectorIndex
from gim.experiment_logger import ExperimentLogger
from gim.experiment_runner import paired_difference, success_per_megabyte


class MeasurementTests(unittest.TestCase):
    def test_inspect_charges_actual_file_bytes_and_identifies_noncanonical_input(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "padded.gim"
            store = MemoryStore(
                (
                    MemoryItem(
                        "id",
                        "fixture",
                        "test",
                        "success",
                        (),
                        (Provenance("episode", ("event",), 1.0),),
                        (1.0, 0.0),
                        "fixture-v1",
                        0,
                    ),
                )
            )
            canonical = store.to_bytes()
            metadata_length = RECORD_HEADER.unpack_from(canonical, STORE_HEADER.size)[0]
            saved = (
                canonical[: STORE_HEADER.size]
                + RECORD_HEADER.pack(metadata_length + 1000)
                + b" " * 1000
                + canonical[STORE_HEADER.size + RECORD_HEADER.size :]
            )
            path.write_bytes(saved)
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["inspect", str(path)]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["store_bytes"], len(saved))
            self.assertEqual(result["canonical_store_bytes"], len(canonical))
            self.assertEqual(
                result["total_persistent_bytes"],
                len(saved) + VectorIndex(store.items()).overhead_bytes(),
            )
            self.assertEqual(result["vector_bytes"], 8)
            self.assertEqual(result["metadata_and_framing_bytes"], len(saved) - 8)
            self.assertFalse(result["canonical"])

    def test_pairing_preserves_correlation(self) -> None:
        result = paired_difference([2.0, 101.0, 201.0], [1.0, 100.0, 200.0])
        self.assertEqual(result.mean_difference, 1.0)
        self.assertEqual(result.standard_error, 0.0)
        self.assertIsNone(paired_difference([1.0], [0.0]).standard_error)
        with self.assertRaises(ValueError):
            paired_difference([], [])

    def test_eta_uses_actual_bytes_and_zero_is_undefined(self) -> None:
        self.assertAlmostEqual(success_per_megabyte(0.75, 0.5, 500_000), 0.5)
        self.assertIsNone(success_per_megabyte(0.5, 0.5, 0))
        with self.assertRaises(ValueError):
            success_per_megabyte(1.1, 0.5, 20)

    def test_config_rejects_bool_as_byte_count(self) -> None:
        with self.assertRaises(ValueError):
            RunConfig(True, 5, 1, 1, "revision", "policy-v1", "environment-v1", "distiller-v1")

    def test_logger_does_not_write_invalid_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "events.jsonl"
            logger = ExperimentLogger(path)
            logger.record("episode", episode_id="e1", terminal_reward=1.0)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                logger.record("episode", terminal_reward=float("nan"))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(before)["kind"], "episode")
