"""Synthetic structure fixtures; no real prediction, service or biological claim."""
import argparse
import importlib.util
import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import test_pipeline as fixtures
from plant_protein_trait import annotation, structure_tools as st
from plant_protein_trait.core import read_json
from plant_protein_trait.net import HTTPFailure

HAS_BIO = importlib.util.find_spec('Bio') is not None


class StructureTransportTests(unittest.TestCase):
    setUp = fixtures.PipelineTest.setUp
    tearDown = fixtures.PipelineTest.tearDown

    def args(self, name='models'):
        return argparse.Namespace(fasta=str(self.fa), output=str(self.root / name), accession=['Q9M0A4'],
            top_k=5, prefer='pdb', blast_xml=None, local_db=None, online_blast=False,
            max_wait=1800, email=None, threads=1, resume=False)

    def test_uniprot_candidates_preserve_hit_rank(self):
        hits = [{'accession': 'Q9M0A4'}, {'accession': 'not-uniprot', 'hit_id': 'sp|P12345|TEST'},
                {'accession': 'Q9M0A4'}, {'accession': 'OTHER'}]
        self.assertEqual(st.uniprot_candidates(hits), ['Q9M0A4', 'P12345'])

    def test_afdb_roles_and_cached_resume(self):
        for name, sequence, role in [('exact', fixtures.SEQ, 'exact_target'), ('homolog', 'M' * 80, 'homolog_or_unverified_reference')]:
            args = self.args(name)
            data = [{'uniprotSequence': sequence, 'uniprotStart': 1, 'uniprotEnd': 80,
                     'pdbUrl': 'https://alphafold.ebi.ac.uk/files/fixture.pdb'}]
            with patch.object(st, 'request', side_effect=[json.dumps(data).encode(), b'fixture model bytes']):
                out = st.download_models(args)
            self.assertEqual(read_json(out / 'models.json')['models'][0]['role'], role)
            args.resume = True
            with patch.object(st, 'request') as call:
                st.download_models(args)
            call.assert_not_called()

    def test_afdb_wrong_host_and_absent_model_are_distinct(self):
        with patch.object(st, 'request', return_value=json.dumps([{'pdbUrl': 'https://other.invalid/model.pdb'}]).encode()):
            with self.assertRaisesRegex(ValueError, 'host'):
                st.download_models(self.args('badhost'))
        with patch.object(st, 'request', side_effect=HTTPFailure('afdb', 404)):
            out = st.download_models(self.args('absent'))
        self.assertEqual(read_json(out / 'models.json')['status'], 'no_models')
        with patch.object(st, 'request', side_effect=HTTPFailure('afdb', 503)):
            with self.assertRaises(HTTPFailure):
                st.download_models(self.args('failed'))
        self.assertFalse((self.root / 'failed/manifest.json').exists())

    def test_zip_traversal_and_input_size_are_rejected(self):
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../outside.cif', 'x')
        raw = self.root / 'raw'
        raw.mkdir()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            st.collect_cifs(archive, raw, 100)
        big = self.root / 'big.cif'
        big.write_text('abcdef')
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            st.collect_cifs(big, raw, 2)


@unittest.skipUnless(HAS_BIO, 'Biopython 1.85 is needed for actual structure conversion tests')
class StructureBioTests(unittest.TestCase):
    setUp = fixtures.PipelineTest.setUp
    tearDown = fixtures.PipelineTest.tearDown

    def pdb(self, path, chains=None, confidence=90):
        from Bio.Data.PDBData import protein_letters_3to1
        reverse = {v: k for k, v in protein_letters_3to1.items()}
        rows, serial = [], 0
        for chain, seq in (chains or {'A': fixtures.SEQ}).items():
            for number, aa in enumerate(seq, 1):
                serial += 1
                # Widely separated CA atoms: PPBuilder would not recover this sequence.
                rows.append(f'ATOM  {serial:5d}  CA  {reverse[aa]:3s} {chain}{number:4d}    '
                            f'{number*10.:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{confidence:6.2f}          C  \n')
            rows.append('TER   \n')
        rows.append('END   \n')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(rows))
        return path

    def test_isolated_residues_and_experimental_bfactor(self):
        path = self.pdb(self.root / 'target.pdb')
        record = st.structure_evidence(path, self.query, 'alphafold3')
        self.assertEqual(record['avg_plddt'], 90)
        self.assertEqual(record['residues_scored'], 80)
        record = st.structure_evidence(path, self.query, 'experimental')
        self.assertIsNone(record['avg_plddt'])
        with self.assertRaisesRegex(ValueError, 'matching'):
            st.structure_evidence(path, dict(self.query, sequence='M' * 80), 'alphafold2')

    def test_ambiguous_chains_require_selection(self):
        path = self.pdb(self.root / 'dimer.pdb', {'A': fixtures.SEQ, 'B': fixtures.SEQ})
        with self.assertRaisesRegex(ValueError, 'one selected chain'):
            st.structure_evidence(path, self.query, 'alphafold3')
        self.assertEqual(st.structure_evidence(path, self.query, 'alphafold3', 'B')['chain'], 'B')

    def test_mapping_exports_exact_chain_only_and_fails_on_partial_match(self):
        source = self.root / 'pdbs'
        self.pdb(source / 'multi.pdb', {'A': 'M' * 20, 'B': fixtures.SEQ})
        args = argparse.Namespace(fasta=str(self.fa), pdb_dir=str(source), output=str(self.root / 'mapped'))
        out = st.map_models(args)
        row = read_json(out / 'mapping.json')['mappings'][0]
        self.assertEqual(row['chain'], 'B')
        _, _, chains = st.read_chains(out / row['file'])
        self.assertEqual(list(chains), ['B'])
        self.fa.write_text('>toy\n' + fixtures.SEQ[:-1] + '\n')
        args.output = str(self.root / 'nomatch')
        with self.assertRaisesRegex(ValueError, 'No complete chain'):
            st.map_models(args)
        self.assertEqual(read_json(Path(args.output) / 'mapping.json')['status'], 'no_exact_match')

    def test_real_cif_conversion_preserves_chain_and_avoids_filename_collisions(self):
        from Bio.PDB import MMCIFIO
        source = self.pdb(self.root / 'origin.pdb')
        model, _, _ = st.read_chains(source)
        io = MMCIFIO()
        io.set_structure(model)
        folder = self.root / 'cifs'
        for name in ['first', 'second']:
            path = folder / name / 'model.cif'
            path.parent.mkdir(parents=True)
            io.save(str(path))
        args = argparse.Namespace(input=str(folder), output=str(self.root / 'converted'), max_input_mb=2)
        out = st.convert(args)
        rows = read_json(out / 'conversion.json')['converted']
        self.assertEqual(len({r['pdb'] for r in rows}), 2)
        for row in rows:
            self.assertEqual(st.read_chains(out / row['pdb'])[2]['A'][0], fixtures.SEQ)

    def test_annotation_rejects_wrong_pdb_and_records_exact_model(self):
        path = self.pdb(self.root / 'target.pdb')
        args = annotation.parser().parse_args(['--fasta', str(self.fa), '--out', str(self.root / 'with_structure'),
                        '--structure-pdb', str(path), '--structure-origin', 'alphafold3'])
        out = annotation.collect(args)
        self.assertEqual(read_json(out / 'annotation_full.json')['toy']['structure']['avg_plddt'], 90)
        self.pdb(path, {'A': 'M' * 80})
        args.output = str(self.root / 'bad_structure')
        with self.assertRaisesRegex(ValueError, 'matching'):
            annotation.collect(args)
        self.assertFalse((Path(args.output) / 'evidence_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
