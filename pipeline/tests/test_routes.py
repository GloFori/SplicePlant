"""Offline transport/executable fixtures; these tests do not claim real biological runs."""
import argparse
import copy
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import test_pipeline as fixtures
from test_pipeline import SEQ, ipr_fixture, xml_fixture
from plant_protein_trait import annotation, inference, jobs, workflow
from plant_protein_trait.compare import compare
from plant_protein_trait.core import canonical, fasta, read_json, write_json
from plant_protein_trait.net import HTTPFailure, NetworkFailure
from plant_protein_trait.settings import analysis_settings


class Clock:
    def __init__(self):
        self.value = 100000

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class RouteTests(unittest.TestCase):
    setUp = fixtures.PipelineTest.setUp
    tearDown = fixtures.PipelineTest.tearDown
    mock_call = fixtures.PipelineTest.mock_call

    def args(self, name, extra=()):
        return annotation.parser().parse_args(["--fasta", str(self.fa), "--output", str(self.root / name), *extra])

    def transport(self, url, **kwargs):
        if "form" in kwargs:
            return b"RID = TEST_RID" if "Blast.cgi" in url else b"TEST_IPR"
        if "Blast.cgi" in url:
            return xml_fixture().encode()
        if "/status/" in url:
            return b"FINISHED"
        if "/result/" in url:
            return json.dumps(ipr_fixture()).encode()
        raise AssertionError(url)

    def external(self, command, raw, label):
        output = Path(command[command.index("-out" if label == "blastp" else "-o") + 1])
        if label == "blastp":
            output.write_text(xml_fixture())
        else:
            self.assertIn("-dp", command)
            self.assertIn("-goterms", command)
            self.assertIn("-pa", command)
            output.write_text(json.dumps(ipr_fixture()))

    def clock_patches(self):
        clock = Clock()
        return clock, [patch("plant_protein_trait.jobs.time.monotonic", clock.now),
                       patch("plant_protein_trait.jobs.time.time", clock.now),
                       patch("plant_protein_trait.jobs.time.sleep", clock.sleep)]

    def test_local_online_same_evidence_prompts_and_explicit_parameters(self):
        apps = ["--interpro-applications", "Pfam,SMART"]
        local = self.args("local", ["--blast-db", "fixture_db", "--interproscan", "fixture_ipr", *apps])
        online = self.args("online", ["--online-blast", "--interpro-remote", "--email", "fixture@example.org", *apps])
        with patch.object(annotation, "external", side_effect=self.external) as executed:
            left = annotation.collect(local)
        clock, patches = self.clock_patches()
        with patches[0], patches[1], patches[2], patch.object(jobs, "request", side_effect=self.transport) as sent:
            right = annotation.collect(online)
        result = compare(left, right)
        self.assertEqual(result["status"], "matched")
        blast_command = executed.call_args_list[0].args[0]
        form = next(c.kwargs["form"] for c in sent.call_args_list if c.kwargs.get("form", {}).get("CMD") == "Put")
        for local_key, remote_key in [("-evalue", "EXPECT"), ("-max_target_seqs", "HITLIST_SIZE"),
                                      ("-word_size", "WORD_SIZE"), ("-matrix", "MATRIX"),
                                      ("-comp_based_stats", "COMPOSITION_BASED_STATISTICS")]:
            self.assertEqual(blast_command[blast_command.index(local_key) + 1], form[remote_key])
        self.assertEqual(form["DATABASE"], "nr")
        self.assertEqual(form["FILTER"], "F")
        for folder, target in [(left, "prep_local"), (right, "prep_online")]:
            inference.prepare(argparse.Namespace(fasta=str(self.fa), annotation_dir=str(folder),
                out=str(self.root / target), trait="target phenotype", forbidden=[], max_prompt_bytes=120000))
        for arm in ["blind_verify", "blind_explore", "targeted_verify", "targeted_explore"]:
            self.assertEqual((self.root / "prep_local" / arm / "messages.json").read_bytes(),
                             (self.root / "prep_online" / arm / "messages.json").read_bytes())

    def test_remote_jobs_submit_independently_before_either_is_finished(self):
        barrier = threading.Barrier(2)
        def remote(*args, **kwargs):
            barrier.wait(timeout=3)
            kind, query, raw = args[:3]
            p = raw / ("blast.xml" if kind == "blast" else "interpro.json")
            p.write_text(xml_fixture() if kind == "blast" else json.dumps(ipr_fixture()))
            return p
        args = self.args("parallel", ["--online-blast", "--interpro-remote", "--email", "fixture@example.org"])
        with patch.object(annotation, "run_job", side_effect=remote) as calls:
            annotation.collect(args)
        self.assertEqual(calls.call_count, 2)

    def test_pending_then_resume_does_not_resubmit(self):
        args = self.args("pending", ["--online-blast", "--email", "fixture@example.org", "--max-wait", "100"])
        clock, patches = self.clock_patches()
        def waiting(url, **kwargs):
            return b"RID = TEST_RID" if "form" in kwargs else b"Status=WAITING"
        with patches[0], patches[1], patches[2], patch.object(jobs, "request", side_effect=waiting):
            with self.assertRaises(jobs.PendingJob):
                annotation.collect(args)
        self.assertEqual(read_json(Path(args.output) / "run_status.json")["status"], "pending")
        args.resume = True
        resume_now = clock.value
        clock, patches = self.clock_patches()
        clock.value = resume_now
        with patches[0], patches[1], patches[2], patch.object(jobs, "request", side_effect=self.transport) as calls:
            ann = annotation.collect(args)
        self.assertTrue(all("form" not in call.kwargs for call in calls.call_args_list))
        self.assertEqual(read_json(ann / "raw/blast_job.json")["rid"], "TEST_RID")
        with patch.object(jobs, "request", side_effect=AssertionError("must reuse completed run")):
            annotation.collect(args)

    def test_transient_poll_failure_recovers_original_job(self):
        args = self.args("network")
        replies = [b"RID = TEST_RID", NetworkFailure("errno 101 Network is unreachable"),
                   b"Status=WAITING", xml_fixture().encode()]
        clock, patches = self.clock_patches()
        with patches[0], patches[1], patches[2], patch.object(jobs, "request", side_effect=replies) as calls:
            result = jobs.run_job("blast", self.query, self.root / "raw", analysis_settings(args), "fixture@example.org", 600, 10)
        self.assertTrue(result.exists())
        self.assertEqual(sum("form" in c.kwargs for c in calls.call_args_list), 1)

    def test_uncertain_post_is_not_repeated(self):
        settings = analysis_settings(self.args("unknown"))
        with patch.object(jobs, "request", side_effect=NetworkFailure("timeout")) as calls:
            with self.assertRaises(NetworkFailure):
                jobs.run_job("blast", self.query, self.root / "raw", settings, "fixture@example.org", 100, 10)
            with self.assertRaisesRegex(RuntimeError, "uncertain"):
                jobs.run_job("blast", self.query, self.root / "raw", settings, "fixture@example.org", 100, 10)
        self.assertEqual(calls.call_count, 1)

    def test_supplied_rid_never_submits_and_records_binding_limit(self):
        settings = analysis_settings(self.args("seed"))
        clock, patches = self.clock_patches()
        with patches[0], patches[1], patches[2], patch.object(jobs, "request", side_effect=self.transport) as calls:
            jobs.run_job("blast", self.query, self.root / "raw", settings, "fixture@example.org", 100, 10, "TEST_RID")
        self.assertTrue(all("form" not in call.kwargs for call in calls.call_args_list))
        self.assertIn("binding_warning", read_json(self.root / "raw/blast_job.json"))

    def test_resume_rejects_changed_search_parameters(self):
        args = self.args("ann", ["--blast-xml", str(self.blast), "--interpro-json", str(self.ipr)])
        annotation.collect(args)
        args.resume = True
        args.max_hits = 20
        with self.assertRaisesRegex(ValueError, "settings differ"):
            annotation.collect(args)

    def test_local_failure_does_not_fall_back_to_online(self):
        args = self.args("fail", ["--blast-db", "fixture_db", "--interproscan", "fixture_ipr"])
        with patch.object(annotation, "external", side_effect=RuntimeError("local fixture error")), \
             patch.object(jobs, "request", side_effect=AssertionError("silent fallback")):
            with self.assertRaisesRegex(RuntimeError, "local fixture"):
                annotation.collect(args)

    def test_one_command_replay_runs_all_four_and_resumes_without_charging(self):
        args = workflow.parser().parse_args(["--mode", "replay", "--fasta", str(self.fa), "--out", str(self.root / "run"),
                "--blast-xml", str(self.blast), "--interpro-json", str(self.ipr), "--trait", "target phenotype", "--model", "mock", "--backend", "ollama"])
        with patch.object(inference, "model_call", side_effect=self.mock_call) as calls:
            root = workflow.execute(args)
            self.assertEqual(calls.call_count, 4)
            args.resume = True
            workflow.execute(args)
            self.assertEqual(calls.call_count, 4)
        self.assertEqual(read_json(root / "workflow_status.json")["status"], "completed")
        self.assertTrue((root / "comparison/comparison.md").exists())

    def test_dna_is_not_accepted_as_protein(self):
        self.fa.write_text(">genomic_fragment\nACGTACGTACGT\n")
        with self.assertRaisesRegex(ValueError, "DNA"):
            fasta(self.fa)

    def test_prepare_only_then_configure_model_and_finish(self):
        args = workflow.parser().parse_args(["--mode", "replay", "--fasta", str(self.fa), "--out", str(self.root / "later"),
                "--blast-xml", str(self.blast), "--interpro-json", str(self.ipr), "--trait", "target phenotype", "--prepare-only"])
        with patch.object(inference, "model_call", side_effect=AssertionError("unexpected model call")):
            root = workflow.execute(args)
        self.assertEqual(read_json(root / "workflow_status.json")["status"], "prepared_not_executed")
        args.prepare_only, args.resume, args.model = False, True, "mock"
        args.backend = "ollama"
        with patch.object(inference, "model_call", side_effect=self.mock_call) as calls:
            workflow.execute(args)
        self.assertEqual(calls.call_count, 4)

    def test_partial_completed_inference_resume_does_not_repeat_success(self):
        root = fixtures.PipelineTest.prepare(self, fixtures.PipelineTest.annotate(self))
        args = fixtures.PipelineTest.run_args(self, root)
        args.analysis, args.mode = "blind", "verify"
        with patch.object(inference, "model_call", side_effect=self.mock_call) as calls:
            inference.run(args)
            args.analysis, args.mode, args.resume = "both", "both", True
            inference.run(args)
        self.assertEqual(calls.call_count, 4)
        inference.freeze(root)

    def test_cached_job_rejects_wrong_sequence(self):
        args = self.args("cache")
        clock, patches = self.clock_patches()
        with patches[0], patches[1], patches[2], patch.object(jobs, "request", side_effect=self.transport):
            jobs.run_job("blast", self.query, self.root / "raw", analysis_settings(args), "fixture@example.org", 100, 10)
        wrong = dict(self.query, sequence_sha256="different")
        with self.assertRaisesRegex(ValueError, "does not match"):
            jobs.run_job("blast", wrong, self.root / "raw", analysis_settings(args), "fixture@example.org", 100, 10)

    def test_setup_archive_cannot_escape_destination(self):
        import io, tarfile
        from plant_protein_trait.resources import extract_checked
        if not hasattr(tarfile, "data_filter"):
            self.skipTest("Safe extraction needs Python 3.12 environment")
        archive = self.root / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            member = tarfile.TarInfo("../escaped.txt")
            member.size = 3
            handle.addfile(member, io.BytesIO(b"bad"))
        with self.assertRaises(tarfile.FilterError):
            extract_checked(archive, self.root / "resources")
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_normalization_is_not_dependent_on_domain_order(self):
        a = {"domains": [{"accession": "PF002"}, {"accession": "PF001"}]}
        b = copy.deepcopy(a)
        b["domains"].reverse()
        self.assertEqual(canonical(inference.evidence_packet(self.query, a)),
                         canonical(inference.evidence_packet(self.query, b)))

    def test_different_real_observations_are_not_reported_as_equal(self):
        left = annotation.collect(self.args("left", ["--blast-xml", str(self.blast), "--interpro-json", str(self.ipr)]))
        self.blast.write_text(xml_fixture().replace("<Hsp_bit-score>80", "<Hsp_bit-score>90"))
        right = annotation.collect(self.args("right", ["--blast-xml", str(self.blast), "--interpro-json", str(self.ipr)]))
        result = compare(left, right)
        self.assertEqual(result["status"], "different")
        self.assertFalse(result["evidence_identical"])

    def test_simultaneous_resume_is_locked_before_network(self):
        from plant_protein_trait.core import run_lock
        args = self.args("locked", ["--online-blast", "--email", "fixture@example.org"])
        with run_lock(args.output), patch.object(jobs, "request", side_effect=AssertionError("duplicate task")):
            with self.assertRaisesRegex(ValueError, "locked"):
                annotation.collect(args)

    def test_retry_after_survives_restart(self):
        settings = analysis_settings(self.args("rate_limited"))
        clock, patches = self.clock_patches()
        with patches[0], patches[1], patches[2], patch.object(jobs, "request", side_effect=[b"RID = TEST_RID", HTTPFailure("ncbi", 429, 300)]):
            with self.assertRaises(jobs.PendingJob):
                jobs.run_job("blast", self.query, self.root / "raw", settings, "fixture@example.org", 100, 10)
        state = read_json(self.root / "raw/blast_job.json")
        clock, patches = self.clock_patches()
        clock.value = state["last_request_epoch"] + 10
        with patches[0], patches[1], patches[2], patch.object(jobs, "request") as calls:
            with self.assertRaises(jobs.PendingJob):
                jobs.run_job("blast", self.query, self.root / "raw", settings, "fixture@example.org", 100, 10)
        calls.assert_not_called()

    def test_missing_model_credentials_stop_before_remote_submission(self):
        args = workflow.parser().parse_args(["--mode", "online", "--fasta", str(self.fa), "--out", str(self.root / "no_key"),
               "--trait", "target phenotype", "--email", "fixture@example.org", "--model", "mock"])
        with patch.dict("os.environ", {}, clear=True), patch.object(jobs, "request") as calls:
            with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
                workflow.execute(args)
        calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
