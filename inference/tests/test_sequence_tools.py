"""Software checks, not biological validation or manuscript reproduction.

Run from the repository root: python -m unittest discover -s inference/tests -v
Optional real-checkpoint checks: set SPLICEPLANT_TEST_WEIGHTS to the released .pth.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
from spliceplant_inputs import read_fasta, read_labeled_csv
from spliceplant_model import StudentModel, load_student, read_checkpoint

def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

scoring = load_module("scoring", "inference/score_sequence_per_base.py")
download = load_module("download", "data/download_assets.py")

class SequenceTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.folder = ROOT / "runs" / "software_tests" / uuid.uuid4().hex
        cls.folder.mkdir(parents=True)
        cls.weights = cls.folder / "synthetic_weights.pth"
        torch.manual_seed(42)
        torch.save(StudentModel(seq_len=21).state_dict(), cls.weights)

    def test_checkpoint_wrapper_prefix_and_window(self):
        state = read_checkpoint(self.weights)
        path = self.folder / "wrapped.pth"
        torch.save({"state_dict": {"module." + key: value for key, value in state.items()}}, path)
        model, window = load_student(path)
        self.assertEqual(window, 21)
        with torch.no_grad():
            result = model(torch.zeros((2, 21, 4)))
        self.assertEqual(tuple(result.shape), (2, 3))
        with self.assertRaises(ValueError):
            load_student(path, window=401)

    def test_checkpoint_missing_tensor_rejected(self):
        state = read_checkpoint(self.weights)
        del state["conv1.weight"]
        path = self.folder / "broken.pth"
        torch.save(state, path)
        with self.assertRaises(RuntimeError):
            load_student(path)

    def test_csv_validation(self):
        path = self.folder / "labels.csv"
        path.write_text("seq,label\nACG,0\nTNA,1\n", encoding="utf-8")
        self.assertEqual(read_labeled_csv(path), (["ACG", "TNA"], [0, 1]))
        for content in ("seq,label\n", "seq,label\nACG,3\n", "seq,label\nACG,1\nACGTA,2\n", "seq,label\nAC,1\n"):
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_labeled_csv(path)

    def test_multifasta_is_not_silently_truncated(self):
        path = self.folder / "two.fa"
        path.write_text(">first\nACG\n>second\nTGC\n", encoding="utf-8")
        self.assertEqual(len(read_fasta(path)), 2)
        with self.assertRaises(ValueError):
            scoring.read_fasta_one(path)

    def test_substitution_reference_checked(self):
        self.assertEqual(scoring.apply_single_edit("ACGT", "2:C>A")[0], "AAGT")
        with self.assertRaises(ValueError):
            scoring.apply_single_edit("ACGT", "2:G>A")

    def test_deletion_alignment_and_gap_exclusion(self):
        mutated, metadata = scoring.apply_motif_deletion("ACGTAC", "3:GT")
        self.assertEqual(mutated, "ACAC")
        raw = np.asarray([[.2, .7, .1]] * 4, dtype=np.float32)
        aligned = scoring.align_probs_after_deletion(raw, 6, metadata["start0"], metadata["del_len"])
        self.assertTrue(np.isnan(aligned[2:4]).all())
        np.testing.assert_equal(aligned[4:], raw[2:])
        calls = scoring.call_peaks(aligned, 1, 0, -1, 0, 50)
        self.assertEqual({call[0] for call in calls}, {1, 2, 5, 6})

    def test_score_cli_and_deletion_message(self):
        output = self.folder / "scoring"
        result = subprocess.run([sys.executable, str(ROOT / "inference/score_sequence_per_base.py"),
            "--fasta", str(ROOT / "data/demo/toy_dna.fa"), "--ckpt", str(self.weights),
            "--device", "cpu", "--batch-size", "8", "--del", "3:GT",
            "--out", str(output / "scores.tsv"), "--call-prefix", str(output / "sites")],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Calling peaks on mut sequence", result.stdout)
        lines = (output / "scores.tsv").read_text().splitlines()
        self.assertEqual(len(lines), 42)
        self.assertIn("nan", lines[3])

    def test_prediction_separates_records(self):
        input_dir = self.folder / "fasta"
        input_dir.mkdir()
        (input_dir / "two.records.fa").write_text(">first\n" + "ACGT" * 6 + "\n>second\n" + "TGCA" * 6 + "\n")
        output = self.folder / "predictions"
        command = [sys.executable, str(ROOT / "inference/predict_labels.py"),
            "--input_dir", str(input_dir), "--output_dir", str(output), "--weights", str(self.weights),
            "--device", "cpu", "--strand", "as-is"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = sorted(output.glob("*.txt"))
        self.assertEqual(len(outputs), 2)
        for path in outputs:
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(lines[0]), len(lines[1]))
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)

    def test_download_checksum_and_unsafe_zip(self):
        path = self.folder / "checksum.bin"
        path.write_bytes(b"abc")
        self.assertTrue(download.verified(path, 3, "900150983cd24fb0d6963f7d28e17f72"))
        self.assertFalse(download.verified(path, 4, "900150983cd24fb0d6963f7d28e17f72"))
        unsafe = self.folder / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as handle:
            handle.writestr("../outside.txt", "unsafe")
        with self.assertRaises(ValueError):
            download.extract_archive(unsafe, self.folder / "extracted")

    @unittest.skipUnless(os.getenv("SPLICEPLANT_TEST_WEIGHTS"), "Released checkpoint path not supplied")
    def test_released_checkpoint_inference(self):
        model, window = load_student(os.environ["SPLICEPLANT_TEST_WEIGHTS"])
        self.assertEqual(window, 401)
        probabilities = scoring.score_per_base("ACGTA", model, window, 2, torch.device("cpu"))
        self.assertEqual(probabilities.shape, (5, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
