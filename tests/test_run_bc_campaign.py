import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'scripts' / 'run_bc_campaign.py'
SPEC = importlib.util.spec_from_file_location('run_bc_campaign', MODULE_PATH)
run_bc_campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_bc_campaign)


def make_launch_settings(tmpdir: str) -> dict:
    base = Path(tmpdir)
    return {
        'nproc_per_node': 1,
        'master_port': 29511,
        'eval_device': 'cuda:0',
        'final_val_json': base / 'val.json',
        'final_test_json': base / 'test.json',
        'campaign_summary_json': base / 'campaign.json',
        'state_file': base / 'state.pth',
        'best_state_file': base / 'best.pth',
        'train_list': base / 'train.txt',
        'val_list': base / 'val.txt',
        'test_list': base / 'test.txt',
        'path_cache': base / 'path_cache.pth',
        'step_count_summary': base / 'step_counts.json',
        'actor_filter_index': None,
    }


class RunBcCampaignMainTest(unittest.TestCase):
    def test_stage_runs_before_train_when_stage_cache_is_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            launch_settings = make_launch_settings(tmpdir)
            launch_settings['best_state_file'].touch()
            stage_and_train = []

            with mock.patch.object(
                run_bc_campaign,
                'parse_args',
                return_value=Namespace(
                    config='configs/step6_bc_large_bounded_full8dan_8192_r4.toml',
                    torchrun_bin='torchrun',
                    python_bin=sys.executable,
                ),
            ), mock.patch.object(
                run_bc_campaign,
                'launcher_payload',
                return_value=(
                    Path('/tmp/config.toml'),
                    {'bc': {'stage': {'enabled': True, 'required_splits': ['train', 'val', 'test']}}},
                    launch_settings,
                    'fingerprint',
                ),
            ), mock.patch.object(
                run_bc_campaign,
                'missing_input_paths',
                return_value={},
            ), mock.patch.object(
                run_bc_campaign,
                'ensure_output_dirs',
            ), mock.patch.object(
                run_bc_campaign,
                'subprocess_env',
                return_value={'MORTAL_CFG': '/tmp/config.toml'},
            ), mock.patch.object(
                run_bc_campaign,
                'build_stage_command',
                return_value=[sys.executable, 'scripts/stage_bc_tensor_shards.py', '--config', '/tmp/config.toml', '--split', 'train'],
            ), mock.patch.object(
                run_bc_campaign,
                'build_train_command',
                return_value=['torchrun', 'mortal/train_bc.py'],
            ), mock.patch.object(
                run_bc_campaign,
                'build_eval_command',
                side_effect=[
                    [sys.executable, 'mortal/eval_bc.py', '--split', 'val'],
                    [sys.executable, 'mortal/eval_bc.py', '--split', 'test'],
                ],
            ), mock.patch.object(
                run_bc_campaign,
                'utc_now_iso',
                side_effect=[
                    '2026-03-30T06:15:00+00:00',
                    '2026-03-30T06:20:00+00:00',
                ],
            ), mock.patch.object(
                run_bc_campaign,
                'run_command',
                side_effect=lambda command, env, stage: stage_and_train.append(stage),
            ), mock.patch.object(
                run_bc_campaign,
                'write_summary',
            ):
                run_bc_campaign.main()

            self.assertEqual(stage_and_train, ['stage', 'train', 'final_val', 'final_test'])

    def test_keyboard_interrupt_writes_interrupted_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            launch_settings = make_launch_settings(tmpdir)
            summary_calls = []

            with mock.patch.object(
                run_bc_campaign,
                'parse_args',
                return_value=Namespace(
                    config='configs/step6_bc_debug.toml',
                    torchrun_bin='torchrun',
                    python_bin=sys.executable,
                ),
            ), mock.patch.object(
                run_bc_campaign,
                'launcher_payload',
                return_value=(Path('/tmp/config.toml'), {}, launch_settings, 'fingerprint'),
            ), mock.patch.object(
                run_bc_campaign,
                'missing_input_paths',
                return_value={},
            ), mock.patch.object(
                run_bc_campaign,
                'ensure_output_dirs',
            ), mock.patch.object(
                run_bc_campaign,
                'subprocess_env',
                return_value={'MORTAL_CFG': '/tmp/config.toml'},
            ), mock.patch.object(
                run_bc_campaign,
                'build_train_command',
                return_value=['torchrun', 'mortal/train_bc.py'],
            ), mock.patch.object(
                run_bc_campaign,
                'build_eval_command',
                side_effect=[
                    [sys.executable, 'mortal/eval_bc.py', '--split', 'val'],
                    [sys.executable, 'mortal/eval_bc.py', '--split', 'test'],
                ],
            ), mock.patch.object(
                run_bc_campaign,
                'utc_now_iso',
                side_effect=[
                    '2026-03-30T06:15:00+00:00',
                    '2026-03-30T06:20:00+00:00',
                ],
            ), mock.patch.object(
                run_bc_campaign,
                'run_command',
                side_effect=KeyboardInterrupt(),
            ), mock.patch.object(
                run_bc_campaign,
                'write_summary',
                side_effect=lambda path, summary: summary_calls.append((path, summary)),
            ):
                with self.assertRaises(SystemExit) as exc:
                    run_bc_campaign.main()

            self.assertEqual(exc.exception.code, 130)
            self.assertEqual(len(summary_calls), 1)
            summary_path, summary = summary_calls[0]
            self.assertEqual(summary_path, launch_settings['campaign_summary_json'])
            self.assertEqual(summary['status'], 'interrupted')
            self.assertEqual(summary['failed_stage'], 'train')
            self.assertEqual(summary['return_code'], 130)
            self.assertEqual(summary['error'], 'KeyboardInterrupt')

    def test_runtime_error_uses_current_stage_in_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            launch_settings = make_launch_settings(tmpdir)
            launch_settings['best_state_file'].touch()
            summary_calls = []

            with mock.patch.object(
                run_bc_campaign,
                'parse_args',
                return_value=Namespace(
                    config='configs/step6_bc_debug.toml',
                    torchrun_bin='torchrun',
                    python_bin=sys.executable,
                ),
            ), mock.patch.object(
                run_bc_campaign,
                'launcher_payload',
                return_value=(Path('/tmp/config.toml'), {}, launch_settings, 'fingerprint'),
            ), mock.patch.object(
                run_bc_campaign,
                'missing_input_paths',
                return_value={},
            ), mock.patch.object(
                run_bc_campaign,
                'ensure_output_dirs',
            ), mock.patch.object(
                run_bc_campaign,
                'subprocess_env',
                return_value={'MORTAL_CFG': '/tmp/config.toml'},
            ), mock.patch.object(
                run_bc_campaign,
                'build_train_command',
                return_value=['torchrun', 'mortal/train_bc.py'],
            ), mock.patch.object(
                run_bc_campaign,
                'build_eval_command',
                side_effect=[
                    [sys.executable, 'mortal/eval_bc.py', '--split', 'val'],
                    [sys.executable, 'mortal/eval_bc.py', '--split', 'test'],
                ],
            ), mock.patch.object(
                run_bc_campaign,
                'utc_now_iso',
                side_effect=[
                    '2026-03-30T06:15:00+00:00',
                    '2026-03-30T06:20:00+00:00',
                ],
            ), mock.patch.object(
                run_bc_campaign,
                'run_command',
                side_effect=[None, RuntimeError('val exploded')],
            ), mock.patch.object(
                run_bc_campaign,
                'write_summary',
                side_effect=lambda path, summary: summary_calls.append((path, summary)),
            ):
                with self.assertRaisesRegex(RuntimeError, 'val exploded'):
                    run_bc_campaign.main()

            self.assertEqual(len(summary_calls), 1)
            _, summary = summary_calls[0]
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['failed_stage'], 'final_val')
            self.assertEqual(summary['error'], 'val exploded')


if __name__ == '__main__':
    unittest.main()
