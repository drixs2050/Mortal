import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_campaign import (  # noqa: E402
    build_eval_command,
    build_stage_command,
    build_train_command,
    make_campaign_summary,
    missing_input_paths,
    query_torch_visible_gpu_inventory,
    resolve_launch_settings,
    validate_torch_visible_launch_gpus,
)


class BcCampaignHelpersTest(unittest.TestCase):
    def test_resolve_launch_settings_requires_outputs(self):
        with self.assertRaisesRegex(ValueError, 'final_val_json'):
            resolve_launch_settings({
                'bc': {
                    'control': {
                        'device': 'cuda:0',
                        'state_file': '/tmp/state.pth',
                        'best_state_file': '/tmp/best.pth',
                    },
                    'dataset': {
                        'train_list': '/tmp/train.txt',
                        'val_list': '/tmp/val.txt',
                        'test_list': '/tmp/test.txt',
                        'path_cache': '/tmp/path_cache.pth',
                        'step_count_summary': '/tmp/step_counts.json',
                    },
                    'launch': {},
                },
            })

    def test_resolve_launch_settings_requires_nonempty_dataset_paths(self):
        with self.assertRaisesRegex(ValueError, 'bc.dataset.path_cache'):
            resolve_launch_settings({
                'bc': {
                    'control': {
                        'device': 'cuda:0',
                        'state_file': '/tmp/state.pth',
                        'best_state_file': '/tmp/best.pth',
                    },
                    'dataset': {
                        'train_list': '/tmp/train.txt',
                        'val_list': '/tmp/val.txt',
                        'test_list': '/tmp/test.txt',
                        'path_cache': '',
                        'step_count_summary': '/tmp/step_counts.json',
                    },
                    'launch': {
                        'final_val_json': '/tmp/val.json',
                        'final_test_json': '/tmp/test.json',
                        'campaign_summary_json': '/tmp/campaign.json',
                    },
                },
            })

    def test_missing_input_paths_reports_absent_artifacts(self):
        launch_settings = {
            'train_list': Path('/tmp/train.txt'),
            'val_list': Path('/tmp/val.txt'),
            'test_list': Path('/tmp/test.txt'),
            'path_cache': Path('/tmp/path_cache.pth'),
            'step_count_summary': Path('/tmp/step_counts.json'),
            'actor_filter_index': Path('/tmp/actor_filter.pth'),
        }
        missing = missing_input_paths(launch_settings)
        self.assertIn('train_list', missing)
        self.assertIn('actor_filter_index', missing)

    def test_build_train_command_uses_torchrun_contract(self):
        cmd = build_train_command(
            config_path='/tmp/config.toml',
            launch_settings={
                'nproc_per_node': 2,
                'master_port': 29517,
            },
        )
        self.assertEqual(
            cmd,
            [
                'torchrun',
                '--standalone',
                '--nproc_per_node',
                '2',
                '--master-port',
                '29517',
                'mortal/train_bc.py',
            ],
        )

    def test_build_stage_command_uses_requested_splits(self):
        cmd = build_stage_command(
            config_path='/tmp/config.toml',
            full_config={
                'bc': {
                    'stage': {
                        'enabled': True,
                        'required_splits': ['train', 'val'],
                    },
                },
            },
            python_bin='/usr/bin/python3',
            splits=['train', 'test'],
        )
        self.assertEqual(
            cmd,
            [
                '/usr/bin/python3',
                'scripts/stage_bc_tensor_shards.py',
                '--config',
                '/tmp/config.toml',
                '--split',
                'train',
                '--split',
                'test',
            ],
        )

    def test_build_stage_command_skips_when_runtime_cache_is_enabled(self):
        cmd = build_stage_command(
            config_path='/tmp/config.toml',
            full_config={
                'bc': {
                    'runtime_cache': {
                        'enabled': True,
                    },
                    'stage': {
                        'enabled': True,
                    },
                },
            },
            python_bin='/usr/bin/python3',
        )
        self.assertEqual(cmd, [])

    def test_build_eval_command_uses_full_eval_flags(self):
        cmd = build_eval_command(
            checkpoint='/tmp/best.pth',
            split='test',
            output_json='/tmp/test.json',
            eval_device='cuda:0',
            python_bin='/usr/bin/python3',
        )
        self.assertEqual(
            cmd,
            [
                '/usr/bin/python3',
                'mortal/eval_bc.py',
                '--checkpoint',
                '/tmp/best.pth',
                '--split',
                'test',
                '--device',
                'cuda:0',
                '--max-batches',
                '0',
                '--output-json',
                '/tmp/test.json',
            ],
        )

    def test_make_campaign_summary_preserves_paths_and_status(self):
        summary = make_campaign_summary(
            config_path='/tmp/config.toml',
            config_fingerprint_value='abc123',
            started_at='2026-03-29T00:00:00+00:00',
            finished_at='2026-03-29T01:00:00+00:00',
            status='completed',
            checkpoint_paths={'best_state_file': '/tmp/best.pth'},
            report_paths={'final_test_json': '/tmp/test.json'},
            commands={'train': ['torchrun']},
        )
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['config_fingerprint'], 'abc123')
        self.assertEqual(summary['checkpoint_paths']['best_state_file'], '/tmp/best.pth')
        self.assertEqual(summary['report_paths']['final_test_json'], '/tmp/test.json')

    @mock.patch('bc_campaign.subprocess.run')
    def test_query_torch_visible_gpu_inventory_returns_json_rows(self, run_mock):
        run_mock.return_value = mock.Mock(
            stdout='[{"cuda_index": 0, "name": "NVIDIA A100-SXM4-40GB", "pci_bus_id": "0000:01:00.0"}]\n',
        )
        inventory = query_torch_visible_gpu_inventory(env={'CUDA_VISIBLE_DEVICES': '0,1'})
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]['cuda_index'], 0)
        self.assertEqual(inventory[0]['name'], 'NVIDIA A100-SXM4-40GB')

    @mock.patch('bc_campaign.subprocess.run')
    def test_validate_torch_visible_launch_gpus_rejects_non_a100_selection(self, run_mock):
        run_mock.return_value = mock.Mock(
            stdout=(
                '[{"cuda_index": 0, "name": "NVIDIA A100-SXM4-40GB"}, '
                '{"cuda_index": 1, "name": "NVIDIA GeForce RTX 3070"}]\n'
            ),
        )
        with self.assertRaisesRegex(ValueError, 'non-A100'):
            validate_torch_visible_launch_gpus(
                env={'CUDA_VISIBLE_DEVICES': '0,1'},
                expected_count=2,
                required_name_substring='A100',
            )


if __name__ == '__main__':
    unittest.main()
