"""Deterministic parser/protocol tests. All biological fixtures here are synthetic."""
import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plant_protein_trait import annotation, inference
from plant_protein_trait.core import canonical, digest, fasta, read_json, verify_manifest, write_json
from plant_protein_trait.parsers import load_annotation, parse_blast, parse_domtbl, parse_interpro
from plant_protein_trait.report import summarize

SEQ = "ACDEFGHIKLMNPQRSTVWY" * 4


def xml_fixture(seq=SEQ):
    return f'''<?xml version="1.0"?><BlastOutput>
<BlastOutput_version>BLASTP test-fixture</BlastOutput_version><BlastOutput_db>synthetic-test-only</BlastOutput_db>
<BlastOutput_iterations><Iteration><Iteration_query-len>{len(seq)}</Iteration_query-len>
<Iteration_hits><Hit><Hit_id>test|reference</Hit_id><Hit_accession>TEST0001</Hit_accession>
<Hit_def>Synthetic reference protein [Test organism]</Hit_def><Hit_len>{len(seq)}</Hit_len>
<Hit_hsps><Hsp><Hsp_query-from>1</Hsp_query-from><Hsp_query-to>{len(seq)}</Hsp_query-to>
<Hsp_hit-from>1</Hsp_hit-from><Hsp_hit-to>{len(seq)}</Hsp_hit-to>
<Hsp_align-len>{len(seq)+2}</Hsp_align-len><Hsp_identity>{len(seq)}</Hsp_identity>
<Hsp_evalue>1e-20</Hsp_evalue><Hsp_bit-score>80</Hsp_bit-score><Hsp_score>160</Hsp_score>
<Hsp_qseq>{seq[:10]}--{seq[10:]}</Hsp_qseq><Hsp_hseq>{seq[:10]}AA{seq[10:]}</Hsp_hseq>
</Hsp></Hit_hsps></Hit></Iteration_hits></Iteration></BlastOutput_iterations></BlastOutput>'''


def ipr_fixture():
    return {"results": [{"sequence": SEQ, "xref": [{"id": "toy"}], "matches": [{
        "signature": {"accession": "PFTEST", "name": "Synthetic domain", "signatureLibraryRelease": {
            "library": "Pfam", "version": "test-only"}, "entry": {"accession": "IPRTEST", "name": "Synthetic family",
            "goXRefs": [{"id": "GO:TEST", "name": "synthetic process", "category": {"name": "BIOLOGICAL_PROCESS"}}]}},
        "evalue": 1e-10, "score": 55, "locations": [{"start": 4, "end": 70}]}]}]}


def model_fixture(analysis, mode):
    return {"analysis": analysis, "mode": mode, "molecular_function": "Synthetic testing result, no biological inference.",
            "pathway": "unknown", "verdict": "insufficient" if analysis == "targeted" else "not_applicable",
            "top5": [{"rank": n, "biological_process": "unknown", "potential_trait": "unknown",
                      "supporting_evidence": [], "confidence": "insufficient", "evidence_grade": "speculative_connection",
                      "reasoning_chain": "Synthetic test fixture; insufficient biological evidence."} for n in range(1, 6)],
            "evidence_assessment": [], "strongest_link": "test", "weakest_link": "test",
            "contrary_evidence": [], "limitations": ["Synthetic mock, not a model run."], "proposed_tests": []}


class PipelineTest(unittest.TestCase):
    def setUp(self):
        # An explicitly selected test workspace also works on managed Windows hosts.
        self.test_parent = Path(os.getenv("PPT_TEST_TMP", tempfile.gettempdir())).resolve()
        self.test_parent.mkdir(parents=True, exist_ok=True)
        self.root = self.test_parent / ("ppt_test_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.fa = self.root / "toy.fa"
        self.fa.write_text(">toy description\n" + SEQ + "\n", encoding="utf-8")
        self.query = fasta(self.fa)
        self.blast = self.root / "blast.xml"
        self.blast.write_text(xml_fixture(), encoding="utf-8")
        self.ipr = self.root / "ipr.json"
        write_json(self.ipr, ipr_fixture())

    def tearDown(self):
        if self.root.resolve().parent != self.test_parent or self.root.is_symlink():
            raise RuntimeError("Refusing cleanup outside the temporary test workspace")
        shutil.rmtree(self.root)

    def annotate(self, name="ann"):
        args = annotation.parser().parse_args(["--fasta", str(self.fa), "--output", str(self.root / name),
                                               "--blast-xml", str(self.blast), "--interpro-json", str(self.ipr)])
        return annotation.collect(args)

    def prepare(self, directory=None, name="prepared"):
        ann = directory or self.annotate()
        args = argparse.Namespace(fasta=str(self.fa), annotation_dir=str(ann), out=str(self.root / name),
                                  trait="target phenotype", forbidden=[], max_prompt_bytes=120000)
        return inference.prepare(args)

    def run_args(self, root):
        return argparse.Namespace(prepared=str(root), backend="ollama", model="mock", base_url=None,
                                  analysis="both", mode="both", max_tokens=500, num_ctx=8192, timeout=10)

    def mock_call(self, messages, args, mode):
        analysis = "targeted" if "Targeted validation." in messages[1]["content"] else "blind"
        return json.dumps(model_fixture(analysis, mode)), {"provider": "MOCK_TEST_ONLY"}

    def test_qc_keeps_sequence_and_header(self):
        self.assertEqual(self.query["length"], 80)
        self.assertEqual(self.query["header"], "toy description")
        self.assertEqual(self.query["sequence_sha256"], digest(SEQ.encode()))

    def test_qc_rejects_second_protein(self):
        self.fa.write_text(">one\nACDE\n>two\nACDE\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            fasta(self.fa)

    def test_qc_rejects_stop_and_ambiguous_alphabet(self):
        for seq in ["ACD*E", "ACDX", "ACD*"]:
            self.fa.write_text(">one\n" + seq)
            with self.assertRaises(ValueError):
                fasta(self.fa)

    def test_blast_statistics_are_not_alignment_length_coverage(self):
        hits, meta = parse_blast(self.blast, self.query)
        self.assertAlmostEqual(hits[0]["identity_pct"], 100 * 80 / 82)
        self.assertEqual(hits[0]["query_coverage_pct"], 100)
        self.assertEqual(hits[0]["bit_score"], 80)
        self.assertEqual(hits[0]["organism"], "Test organism")
        self.assertEqual(meta["database"], "synthetic-test-only")

    def test_blast_rejects_wrong_sequence_same_length(self):
        self.blast.write_text(xml_fixture("A" * 80))
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_blast(self.blast, self.query)

    def test_blast_xml2_with_namespace(self):
        text = xml_fixture().replace("<BlastOutput>", '<BlastXML2 xmlns="urn:test"><BlastOutput2><report><Report>')
        text = text.replace("</BlastOutput>", "</Report></report></BlastOutput2></BlastXML2>")
        replacements = {"BlastOutput_version": "version", "BlastOutput_db": "db", "BlastOutput_iterations": "results",
                        "Iteration": "Search", "Iteration_query-len": "query-len", "Iteration_hits": "hits",
                        "Hit_id": "id", "Hit_accession": "accession", "Hit_def": "title", "Hit_len": "len", "Hit_hsps": "hsps"}
        for before, after in sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
            text = text.replace("<" + before + ">", "<" + after + ">").replace("</" + before + ">", "</" + after + ">")
        text = text.replace("Hsp_", "").replace("<id>", "<description><HitDescr><id>")
        text = text.replace("</title>", "</title><sciname>Test organism</sciname></HitDescr></description>")
        self.blast.write_text(text)
        hits, _ = parse_blast(self.blast, self.query)
        self.assertEqual(hits[0]["accession"], "TEST0001")

    def test_interpro_nested_go_extracted_as_computational(self):
        domains, gos = parse_interpro(self.ipr, self.query)
        self.assertEqual(domains[0]["accession"], "PFTEST")
        self.assertEqual(domains[0]["interpro_accession"], "IPRTEST")
        self.assertEqual(gos[0]["category"], "BIOLOGICAL_PROCESS")
        self.assertFalse(gos[0]["experimental_on_query"])

    def test_interpro_rejects_unbound_sequence(self):
        data = ipr_fixture()
        data["results"][0]["sequence"] = "A" * 80
        write_json(self.ipr, data)
        with self.assertRaisesRegex(ValueError, "exact target"):
            parse_interpro(self.ipr, self.query)

    def test_annotation_complete_offline_and_no_proxy(self):
        with patch("plant_protein_trait.annotation.request", side_effect=AssertionError("unexpected network")):
            ann = self.annotate()
        data = read_json(ann / "annotation_full.json")["toy"]
        self.assertIsNone(data["structure"]["avg_plddt"])
        self.assertNotIn("structure_score", data["structure"])
        verify_manifest(ann, read_json(ann / "evidence_manifest.json"))

    def test_legacy_dict_loads_and_discards_proxy(self):
        ann = self.root / "legacy"
        write_json(ann / "annotation_full.json", {"toy": {"length": 80, "domains": [],
                   "structure": {"reason": "proxy_score", "structure_score": 90, "avg_plddt": 90}}})
        rec = load_annotation(ann, self.query)
        self.assertEqual(rec["structure"]["status"], "missing")
        self.assertIn("Legacy", rec["binding_warning"])

    def test_unknown_id_does_not_select_first_candidate(self):
        ann = self.root / "legacy"
        write_json(ann / "annotation_full.json", {"other_gene": {"length": 80}})
        with self.assertRaises(ValueError):
            load_annotation(ann, self.query)

    def test_prepare_four_independent_prompt_files(self):
        root = self.prepare()
        packet = read_json(root / "protein_evidence.json")
        self.assertEqual(packet["query"]["id"], "query_001")
        for name in ["blind_verify", "blind_explore", "targeted_verify", "targeted_explore"]:
            messages = read_json(root / name / "messages.json")
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            self.assertIn(canonical(packet).decode(), messages[1]["content"])
            if name.startswith("blind"):
                self.assertNotIn("target phenotype", json.dumps(messages))
        self.assertFalse((root / "blind_verify/result.json").exists())
        self.assertFalse((root / "results_frozen.json").exists())

    def test_blind_blocks_target_and_population_context(self):
        for text in ["target phenotype", "strong GWAS support", "sQTL", "private experiment label"]:
            with self.assertRaises(ValueError):
                inference.assert_blind_clean(text, "target phenotype", ["private experiment label"])

    def test_prepare_rejects_contaminated_annotation(self):
        ann = self.annotate()
        data = read_json(ann / "annotation_full.json")
        data["toy"]["blast_hits"][0]["description"] = "target phenotype"
        write_json(ann / "annotation_full.json", data)
        with self.assertRaisesRegex(ValueError, "Frozen file changed"):
            self.prepare(ann)

    def test_manifest_detects_tampered_prompt(self):
        root = self.prepare()
        (root / "blind_verify/messages.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "Frozen file changed"):
            inference.run(self.run_args(root))

    def test_mock_four_calls_freeze_then_compare(self):
        root = self.prepare()
        with patch("plant_protein_trait.inference.model_call", side_effect=self.mock_call) as call:
            inference.run(self.run_args(root))
        self.assertEqual(call.call_count, 4)
        inference.freeze(root)
        verify_manifest(root, read_json(root / "results_frozen.json"))
        bg = self.root / "independent.txt"
        bg.write_text("TEST BACKGROUND ONLY")
        report = summarize(root, self.root / "comparison", bg)
        self.assertTrue((report / "evidence_table.tsv").is_file())
        self.assertNotIn("TEST BACKGROUND", (root / "blind_verify/messages.json").read_text())
        self.assertTrue((report / "postfreeze_background.txt").exists())

    def test_provider_failure_is_not_success(self):
        root = self.prepare()
        with patch("plant_protein_trait.inference.model_call", side_effect=RuntimeError("Unavailable")):
            with self.assertRaises(RuntimeError):
                inference.run(self.run_args(root))
        self.assertEqual(read_json(root / "blind_verify/run_status.json")["status"], "failed")
        with self.assertRaises(ValueError):
            inference.freeze(root)

    def test_empty_evidence_not_sent_to_model(self):
        args = annotation.parser().parse_args(["--fasta", str(self.fa), "--output", str(self.root / "ann")])
        ann = annotation.collect(args)
        root = self.prepare(ann)
        with patch("plant_protein_trait.inference.model_call") as call:
            with self.assertRaisesRegex(ValueError, "No objective"):
                inference.run(self.run_args(root))
        call.assert_not_called()

    def test_fabricated_evidence_id_rejected(self):
        data = model_fixture("blind", "verify")
        data["top5"][0]["supporting_evidence"] = ["NONEXISTENT"]
        with self.assertRaises(ValueError):
            inference.validate_result(data, {"evidence": []}, "blind", "verify")

    def test_output_directory_cannot_mix_old_results(self):
        self.annotate()
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.annotate()

    def test_online_submission_requires_contact(self):
        args = annotation.parser().parse_args(["--fasta", str(self.fa), "--output", str(self.root / "ann"), "--online-blast"])
        with self.assertRaisesRegex(ValueError, "email"):
            annotation.collect(args)

    def test_afdb_homolog_not_promoted_to_target(self):
        raw = self.root / "raw"
        with patch("plant_protein_trait.annotation.request", return_value=json.dumps([
                {"uniprotSequence": "A" * 80, "pdbUrl": "https://alphafold.ebi.ac.uk/example.pdb"}]).encode()) as call:
            st = annotation.afdb(self.query, "P12345", raw)
        self.assertEqual(st["status"], "reference_only")
        self.assertIsNone(st["avg_plddt"])
        self.assertEqual(call.call_count, 1)

    def test_bad_provider_schema_retains_raw_and_fails(self):
        root = self.prepare()
        with patch("plant_protein_trait.inference.model_call", return_value=("not json", {})):
            with self.assertRaises(ValueError):
                inference.run(self.run_args(root))
        self.assertEqual((root / "blind_verify/raw_response.txt").read_text(), "not json")
        self.assertFalse((root / "blind_verify/result.json").exists())

    def test_computational_evidence_cannot_become_direct_trait_evidence(self):
        data = model_fixture("targeted", "verify")
        data["top5"][0].update(supporting_evidence=["E001"], evidence_grade="direct_evidence", confidence="high")
        with self.assertRaisesRegex(ValueError, "direct experimental"):
            inference.validate_result(data, {"evidence": [{"evidence_id": "E001", "experimental_on_query": False}]}, "targeted", "verify")

    def test_neutral_id_does_not_modify_amino_acid_sequence(self):
        query = dict(self.query, id="ACDE", header="ACDE")
        packet = inference.evidence_packet(query, {"blast_hits": [], "domains": [], "go_terms": []})
        self.assertEqual(packet["query"]["sequence"], SEQ)

    def test_truncated_response_is_saved_but_cannot_freeze(self):
        root = self.prepare()
        with patch("plant_protein_trait.inference.model_call", return_value=("{partial", {"completion_complete": False})):
            with self.assertRaisesRegex(RuntimeError, "token-limited"):
                inference.run(self.run_args(root))
        self.assertEqual((root / "blind_verify/raw_response.txt").read_text(), "{partial")
        self.assertFalse((root / "blind_verify/result.json").exists())

    def test_ollama_small_context_is_rejected_before_network(self):
        args = self.run_args(self.root)
        args.num_ctx = 100
        with patch("plant_protein_trait.inference.request") as call:
            with self.assertRaisesRegex(ValueError, "context budget"):
                inference.model_call([{"role": "user", "content": "protein"}], args, "verify")
        call.assert_not_called()

    def test_pfam_independent_domain_evalue(self):
        dom = self.root / "pfam.domtbl"
        dom.write_text("Test PFTEST 80 toy - 80 1e-30 100 0 1 1 1e-10 1e-8 50 0 1 70 4 73 3 74 0.9 Test\n")
        parsed = parse_domtbl(dom, self.query)
        self.assertEqual(parsed[0]["e_value"], 1e-8)
        self.assertEqual(parsed[0]["locations"][0], {"start": 4, "end": 73})

    def test_unverified_legacy_pdb_not_counted_as_prediction(self):
        ann = self.root / "legacy"
        write_json(ann / "annotation_full.json", {"toy": {"length": 80, "structure": {
            "pdb_path": "some_model.pdb", "avg_plddt": 95}}})
        rec = load_annotation(ann, self.query)
        packet = inference.evidence_packet(self.query, rec)
        self.assertEqual(packet["structure_status"], "legacy_unverified")
        self.assertFalse(packet["evidence"])

    def test_provider_adapter_sends_only_two_messages_and_retains_model_id(self):
        args = self.run_args(self.root)
        args.backend = "dashscope"
        messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}]
        response = {"id": "mock-response", "model": "resolved-test-version", "usage": {"total_tokens": 10},
                    "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fixture-only-not-a-real-key"}):
            with patch("plant_protein_trait.inference.request", return_value=json.dumps(response).encode()) as call:
                text, metadata = inference.model_call(messages, args, "verify")
        self.assertEqual(call.call_args.kwargs["payload"]["messages"], messages)
        self.assertEqual(metadata["provider_response"]["model"], "resolved-test-version")
        self.assertNotIn("fixture-only-not-a-real-key", json.dumps(metadata))
        self.assertTrue(metadata["completion_complete"])

    def test_retrieved_incomplete_provider_envelope_is_preserved(self):
        args = self.run_args(self.root)
        args.num_ctx = 32768
        response = {"done": True, "done_reason": "length", "message": {"content": "{partial"}}
        with patch("plant_protein_trait.inference.request", return_value=json.dumps(response).encode()):
            text, metadata = inference.model_call([{"role": "user", "content": "test"}], args, "explore")
        self.assertFalse(metadata["completion_complete"])
        self.assertEqual(metadata["provider_response"], response)

    def test_negative_blast_search_is_not_a_failed_search(self):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_fixture())
        hits = root.find(".//Iteration_hits")
        hits.clear()
        self.blast.write_text(ET.tostring(root, encoding="unicode"))
        ann = self.annotate()
        rec = read_json(ann / "annotation_full.json")["toy"]
        self.assertEqual(rec["status"]["blast"], "completed_no_hits")
        self.assertEqual(rec["blast_hits"], [])

    def test_manifest_path_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid manifest path"):
            verify_manifest(self.root, {"files": {"../outside": "no"}})


if __name__ == "__main__":
    unittest.main()
