import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

from bc_stage import (  # noqa: E402
    StagedShardIterableDataset,
    build_stage_cache,
    load_stage_manifest,
    stage_backend_available,
    stage_manifest_path,
    stage_manifest_paths,
    stage_preload_budget_bytes,
    validate_stage_backend,
)
from dataloader import ActionChunkBuffer  # noqa: E402


def tensor_nbytes(value: torch.Tensor) -> int:
    return value.element_size() * value.numel()


class _FakeStageBuilder:
    def build_buffer_for_files(self, file_list, augmented):
        filename = Path(file_list[0]).name.split('.')[0]
        base = int(filename.lstrip('f')) * 10
        obs = torch.tensor([[base], [base + 100]], dtype=torch.int16)
        actions = torch.tensor([base, base + 1], dtype=torch.int64)
        masks = torch.ones((2, 46), dtype=torch.bool)
        return ActionChunkBuffer(
            obs=obs,
            actions=actions,
            masks=masks,
            invisible_obs=None,
            file_count=1,
            sample_count=2,
            size_bytes=tensor_nbytes(obs) + tensor_nbytes(actions) + tensor_nbytes(masks),
        )


class BcStageHelpersTest(unittest.TestCase):
    def make_config(self, tmpdir: str, *, backend: str = 'npy_shards') -> tuple[dict, dict[str, list[str]]]:
        base = Path(tmpdir)
        split_files = {
            'train': [base / 'f0.json.gz', base / 'f1.json.gz', base / 'f2.json.gz', base / 'f3.json.gz'],
            'val': [base / 'f4.json.gz', base / 'f5.json.gz'],
            'test': [base / 'f6.json.gz'],
        }
        for file_list in split_files.values():
            for filename in file_list:
                filename.write_text('', encoding='utf-8')

        list_paths = {}
        for split, file_list in split_files.items():
            list_path = base / f'{split}.txt'
            list_path.write_text(
                '\n'.join(str(filename) for filename in file_list) + '\n',
                encoding='utf-8',
            )
            list_paths[split] = list_path

        config = {
            'bc': {
                'control': {
                    'version': 4,
                },
                'dataset': {
                    'root_dir': '',
                    'train_list': str(list_paths['train']),
                    'val_list': str(list_paths['val']),
                    'test_list': str(list_paths['test']),
                    'path_cache': '',
                    'actor_filter_index': '',
                    'actor_filter_manifest': '',
                    'min_actor_dan': None,
                    'player_names_files': [],
                    'exclude_names_files': [],
                    'oracle': False,
                    'trust_seed': False,
                    'always_include_kan_select': True,
                },
                'stage': {
                    'enabled': True,
                    'backend': backend,
                    'cache_root': str(base / 'cache'),
                    'format_version': 1,
                    'target_shard_size_mib': 0.0001,
                    'max_shard_size_mib': 0.0001,
                    'preload_ram_budget_gib': 160,
                    'preload_low_watermark': 0.65,
                    'preload_high_watermark': 0.90,
                    'reuse': 'if_valid',
                    'required_splits': ['train', 'val', 'test'],
                    'allow_zarr': True,
                },
            },
        }
        normalized = {
            split: [str(filename.resolve()) for filename in file_list]
            for split, file_list in split_files.items()
        }
        return config, normalized

    @patch('bc_stage._stage_chunk_builder', return_value=_FakeStageBuilder())
    def test_build_stage_cache_and_iterate_npy_shards(self, _builder_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, _ = self.make_config(tmpdir)
            manifests = build_stage_cache(config, splits=['train', 'val'], force=True)
            manifest_paths = stage_manifest_paths(config, splits=['train', 'val'])

            self.assertEqual(manifests['train']['backend'], 'npy_shards')
            self.assertTrue(manifest_paths['train'].exists())
            payload = load_stage_manifest(manifest_paths['train'])
            self.assertEqual(payload['shard_count'], 4)
            self.assertEqual(payload['sample_count'], 8)

            dataset = StagedShardIterableDataset(
                manifest_path=manifest_paths['train'],
                batch_size=16,
                shuffle=False,
                cycle=False,
                num_epochs=1,
                preload_budget_bytes=16 * 1024 * 1024,
                preload_low_watermark=0.65,
                preload_high_watermark=0.90,
                rank=0,
                world_size=1,
            )
            all_actions = []
            for _obs, actions, _masks in dataset:
                all_actions.extend(actions.tolist())
            self.assertEqual(all_actions, [0, 1, 10, 11, 20, 21, 30, 31])

    @patch('bc_stage._stage_chunk_builder', return_value=_FakeStageBuilder())
    def test_staged_dataset_shards_train_manifest_round_robin_for_ddp(self, _builder_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, _ = self.make_config(tmpdir)
            build_stage_cache(config, splits=['train'], force=True)
            manifest_path = stage_manifest_paths(config, splits=['train'])['train']

            rank0 = StagedShardIterableDataset(
                manifest_path=manifest_path,
                batch_size=16,
                shuffle=False,
                cycle=False,
                num_epochs=1,
                preload_budget_bytes=16 * 1024 * 1024,
                preload_low_watermark=0.65,
                preload_high_watermark=0.90,
                rank=0,
                world_size=2,
            )
            rank1 = StagedShardIterableDataset(
                manifest_path=manifest_path,
                batch_size=16,
                shuffle=False,
                cycle=False,
                num_epochs=1,
                preload_budget_bytes=16 * 1024 * 1024,
                preload_low_watermark=0.65,
                preload_high_watermark=0.90,
                rank=1,
                world_size=2,
            )

            rank0_actions = []
            rank1_actions = []
            for _obs, actions, _masks in rank0:
                rank0_actions.extend(actions.tolist())
            for _obs, actions, _masks in rank1:
                rank1_actions.extend(actions.tolist())

            self.assertEqual(rank0_actions, [0, 1, 20, 21])
            self.assertEqual(rank1_actions, [10, 11, 30, 31])

    def test_stage_preload_budget_bytes_divides_node_budget_by_world_size(self):
        config = {
            'bc': {
                'stage': {
                    'enabled': True,
                    'backend': 'npy_shards',
                    'preload_ram_budget_gib': 160,
                },
            },
        }
        self.assertEqual(stage_preload_budget_bytes(full_config=config, world_size=2), 80 * (1024 ** 3))
        self.assertEqual(stage_preload_budget_bytes(full_config=config, world_size=1), 160 * (1024 ** 3))

    def test_stage_manifest_path_for_train_is_stable_across_requested_split_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, normalized = self.make_config(tmpdir)
            train_only = stage_manifest_path(
                config,
                split='train',
                split_lists={'train': normalized['train']},
            )
            all_splits = stage_manifest_paths(config, splits=['train', 'val', 'test'])['train']
            self.assertEqual(train_only, all_splits)

    def test_stage_manifest_path_changes_when_target_shard_size_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, normalized = self.make_config(tmpdir)
            manifest_a = stage_manifest_path(
                config,
                split='train',
                split_lists={'train': normalized['train']},
            )

            config['bc']['stage']['target_shard_size_mib'] = 0.0002
            config['bc']['stage']['max_shard_size_mib'] = 0.0002
            manifest_b = stage_manifest_path(
                config,
                split='train',
                split_lists={'train': normalized['train']},
            )
            self.assertNotEqual(manifest_a, manifest_b)

    def test_stage_manifest_path_changes_when_max_stage_size_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, normalized = self.make_config(tmpdir)
            manifest_a = stage_manifest_path(
                config,
                split='train',
                split_lists={'train': normalized['train']},
            )

            config['bc']['stage']['max_stage_size_gib'] = 1
            manifest_b = stage_manifest_path(
                config,
                split='train',
                split_lists={'train': normalized['train']},
            )
            self.assertNotEqual(manifest_a, manifest_b)

    @patch('bc_stage._stage_chunk_builder', return_value=_FakeStageBuilder())
    def test_build_stage_cache_can_truncate_at_max_stage_size(self, _builder_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, _ = self.make_config(tmpdir)
            config['bc']['stage']['target_shard_size_mib'] = 0.0002
            config['bc']['stage']['max_shard_size_mib'] = 0.0002
            config['bc']['stage']['max_stage_size_gib'] = 0.0000002

            manifests = build_stage_cache(config, splits=['train'], force=True)
            payload = manifests['train']

            self.assertTrue(payload['truncated'])
            self.assertEqual(payload['truncate_reason'], 'max_stage_size_gib')
            self.assertEqual(payload['source_file_count'], 4)
            self.assertLess(payload['staged_file_count'], payload['source_file_count'])
            self.assertGreater(payload['staged_file_count'], 0)

    def test_validate_stage_backend_rejects_missing_zarr_dependency(self):
        if stage_backend_available('zarr'):
            self.skipTest('zarr is installed in this environment')
        with self.assertRaisesRegex(ValueError, 'zarr package is not installed'):
            validate_stage_backend({
                'backend': 'zarr',
                'allow_zarr': True,
            })


if __name__ == '__main__':
    unittest.main()
