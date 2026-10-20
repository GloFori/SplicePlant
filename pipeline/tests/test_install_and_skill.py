"""Offline installation/configuration/Skill integration; external tools are mocked."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import test_pipeline as fixtures
from plant_protein_trait import annotation, configuration, resources, setup
from plant_protein_trait.core import read_json, write_json

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('skill_bridge', PACKAGE / 'skill/scripts/ppt.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class InstallTests(unittest.TestCase):
    setUp = fixtures.PipelineTest.setUp
    tearDown = fixtures.PipelineTest.tearDown

    def args(self, *extra):
        return setup.parser().parse_args(['--root', str(self.root / 'install'), '--conda', 'fixture-conda', *extra])

    def test_plan_does_not_create_root_or_run_conda(self):
        args = self.args('--with-databases')
        with patch.object(setup, 'command') as command, contextlib.redirect_stdout(io.StringIO()):
            setup.install(args)
        command.assert_not_called()
        self.assertFalse(args.root.exists())

    def test_unsupported_platform_stops_before_installation(self):
        args = self.args('--execute')
        with patch.object(setup.platform, 'system', return_value='Linux'), patch.object(setup.platform, 'machine', return_value='aarch64'):
            with self.assertRaisesRegex(ValueError, 'x86_64'):
                setup.install(args)
        self.assertFalse(args.root.exists())

    def linux(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(setup.platform, 'system', return_value='Linux'))
        stack.enter_context(patch.object(setup.platform, 'machine', return_value='x86_64'))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_install_writes_config_and_resume_skips_env_creation(self):
        args = self.args('--execute', '--with-databases')
        with self.linux(), patch.object(setup, 'command', return_value=argparse.Namespace(stdout='@EXPLICIT\nfixture')) as call:
            setup.install(args)
            first = [list(map(str, c.args[0])) for c in call.call_args_list]
            self.assertIn('create', first[0])
            self.assertTrue(any(x.endswith('0_download_databases.py') for x in first[-1]))
            self.assertIn('--execute', first[-1])
            self.assertEqual(read_json(args.root / 'installation_status.json')['status'], 'ready')
            config = read_json(args.root / 'local_config.json')
            self.assertEqual(config['resources_manifest'], str(args.root / 'resources/resources.json'))
            call.reset_mock()
            setup.install(args)
            self.assertFalse(any('create' in list(map(str, c.args[0])) for c in call.call_args_list))

    def test_foreign_directory_is_not_overwritten(self):
        args = self.args('--execute')
        args.root.mkdir()
        sentinel = args.root / 'unrelated.txt'
        sentinel.write_text('keep')
        with self.linux(), patch.object(setup, 'command') as call:
            with self.assertRaisesRegex(ValueError, 'not empty'):
                setup.install(args)
        call.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_failed_install_records_failure(self):
        args = self.args('--execute')
        with self.linux(), patch.object(setup, 'command', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                setup.install(args)
        self.assertEqual(read_json(args.root / 'installation_status.json')['status'], 'failed')
        self.assertFalse((args.root / 'environment_ready.json').exists())

    def test_interrupted_prefix_uses_update_without_unsupported_yes_flag(self):
        args = self.args('--execute')
        with self.linux(), patch.object(setup, 'command', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                setup.install(args)
        history = args.root / 'env/conda-meta/history'
        history.parent.mkdir(parents=True)
        history.write_text('fixture')
        with self.linux(), patch.object(setup, 'command', return_value=argparse.Namespace(stdout='fixture')) as call:
            setup.install(args)
        create = list(map(str, call.call_args_list[0].args[0]))
        self.assertIn('update', create)
        self.assertNotIn('--yes', create)

    def make_resources(self):
        folder = self.root / 'resources'
        folder.mkdir()
        (folder / 'swissprot.pin').write_text('fixture')
        (folder / 'interproscan.sh').write_text('fixture')
        rec = {'database': 'swissprot', 'blast_db': str(folder / 'swissprot'),
               'interproscan': str(folder / 'interproscan.sh'), 'pfam_db': None}
        write_json(folder / 'resources.json', rec)
        return folder

    def test_manifest_chooses_database_and_rejects_explicit_conflict(self):
        folder = self.make_resources()
        args = annotation.parser().parse_args(['--fasta', str(self.fa), '--out', str(self.root / 'ann'),
                                               '--resources', str(folder / 'resources.json')])
        configuration.configure_resources(args)
        self.assertEqual(args.remote_database, 'swissprot')
        self.assertEqual(args.blast_db, str(folder / 'swissprot'))
        args.remote_database = 'nr'
        with self.assertRaisesRegex(ValueError, 'differs'):
            configuration.configure_resources(args)

    def test_bad_config_and_unprepared_pfam_stop(self):
        folder = self.make_resources()
        args = annotation.parser().parse_args(['--fasta', str(self.fa), '--out', str(self.root / 'ann'), '--use-pfam'])
        with self.assertRaisesRegex(ValueError, 'requires'):
            configuration.configure_resources(args)
        args.resources = str(folder / 'resources.json')
        with self.assertRaisesRegex(ValueError, 'not prepared'):
            configuration.configure_resources(args)
        args.config = str(self.root / 'bad_config.json')
        write_json(args.config, {})
        with self.assertRaisesRegex(ValueError, 'lacks'):
            configuration.configure_resources(args)

    def test_resource_snapshot_is_archived_and_bound_to_resume(self):
        folder = self.make_resources()
        path = folder / 'resources.json'
        args = annotation.parser().parse_args(['--fasta', str(self.fa), '--out', str(self.root / 'ann'),
                                               '--resources', str(path)])
        def external(command, raw, label):
            destination = Path(command[command.index('-out' if label == 'blastp' else '-o') + 1])
            destination.write_text(fixtures.xml_fixture() if label == 'blastp' else json.dumps(fixtures.ipr_fixture()))
        with patch.object(annotation, 'external', side_effect=external):
            output = annotation.collect(args)
        self.assertEqual((output / 'raw/resources.json').read_bytes(), path.read_bytes())
        record = read_json(path)
        record['snapshot_changed'] = True
        write_json(path, record)
        args.resume = True
        with self.assertRaisesRegex(ValueError, 'sources or analysis settings differ'):
            annotation.collect(args)

    def test_blast_checkpoint_survives_later_download_failure(self):
        args = resources.parser().parse_args(['--root', str(self.root / 'db_install'), '--execute'])
        downloads = []
        def fake_command(command, cwd=None):
            if command[0] == 'update_blastdb.pl':
                downloads.append(command)
                (Path(cwd) / 'nr.pin').write_text('fixture BLAST database')
        archive_bytes = b'synthetic archive bytes'
        def fake_download(url, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if str(path).endswith('.md5'):
                name = path.name[:-4]
                path.write_text(hashlib.md5(archive_bytes).hexdigest() + '  ' + name + '\n')
            else:
                path.write_bytes(archive_bytes)
        def fake_extract(archive, target):
            ipr = target / ('interproscan-' + args.interpro_version) / 'interproscan.sh'
            ipr.parent.mkdir(parents=True)
            ipr.write_text('fixture')
        with self.linux(), patch.object(resources.shutil, 'which', side_effect=lambda x: x), \
             patch.object(resources, 'command', side_effect=fake_command), \
             patch.object(resources.subprocess, 'run', return_value=argparse.Namespace(stdout='fixture version', stderr='')):
            with patch.object(resources, 'download', side_effect=OSError('interrupted download')):
                with self.assertRaises(OSError):
                    resources.prepare(args)
            self.assertTrue((args.root / 'blast_ready.json').exists())
            with patch.object(resources, 'download', side_effect=fake_download), patch.object(resources, 'extract_checked', side_effect=fake_extract):
                resources.prepare(args)
            self.assertEqual(len(downloads), 1)
            with patch.object(resources, 'download') as download:
                resources.prepare(args)
            download.assert_not_called()
            (args.root / 'blast/nr.pin').write_text('changed size')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                resources.prepare(args)

    def test_installed_skill_binds_to_checkout_and_refuses_overwrite(self):
        dst = bridge.install_skill(PACKAGE, self.root / 'installed_skill')
        with patch.object(bridge, 'SKILL', dst), patch.dict(os.environ, {'PPT_PIPELINE_ROOT': ''}):
            self.assertEqual(bridge.locate(), PACKAGE)
        self.assertFalse((dst / 'plant_protein_trait').exists())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            bridge.install_skill(PACKAGE, dst)
        result = subprocess.run([sys.executable, str(dst / 'scripts/ppt.py'), 'stage', 'qc', '--fasta', str(self.fa)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('QC PASS', result.stdout)

    def test_skill_and_human_prepare_identical_messages(self):
        dst = bridge.install_skill(PACKAGE, self.root / 'skill')
        outputs = [self.root / 'human', self.root / 'agent']
        for prefix, out in zip(([sys.executable, str(PACKAGE / 'llm.py'), 'pipeline'],
                               [sys.executable, str(dst / 'scripts/ppt.py'), 'run']), outputs):
            result = subprocess.run([*prefix, '--mode', 'replay', '--fasta', str(self.fa),
                '--blast-xml', str(self.blast), '--interpro-json', str(self.ipr), '--trait', 'target phenotype',
                '--prepare-only', '--out', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        for slot in ['blind_verify', 'blind_explore', 'targeted_verify', 'targeted_explore']:
            self.assertEqual((outputs[0] / 'inference' / slot / 'messages.json').read_bytes(),
                             (outputs[1] / 'inference' / slot / 'messages.json').read_bytes())

    def test_skill_uses_recorded_conda_environment_for_run(self):
        config = self.root / 'config.json'
        write_json(config, {'conda': 'fixture-conda', 'environment_prefix': '/fixture/env'})
        with patch.object(bridge.subprocess, 'call', return_value=0) as call:
            with self.assertRaises(SystemExit) as raised:
                bridge.main(['run', '--config', str(config), '--mode', 'local'])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(call.call_args.args[0][:6], ['fixture-conda', 'run', '--no-capture-output', '--prefix', '/fixture/env', 'python'])
        with patch.object(bridge.subprocess, 'call', return_value=0) as call:
            with self.assertRaises(SystemExit):
                bridge.main(['run', '--config=' + str(config), '--mode', 'local'])
        self.assertEqual(call.call_args.args[0][0], 'fixture-conda')


if __name__ == '__main__':
    unittest.main()
