import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_runtime import (  # noqa: E402
    config_fingerprint,
    effective_global_batch,
    resolve_distributed_context,
    seed_everything,
    shard_file_list_round_robin,
    stored_config_fingerprint,
)
from train_bc import validate_resume_fingerprint  # noqa: E402


class BcRuntimeHelpersTest(unittest.TestCase):
    def test_effective_global_batch_multiplies_rank_and_accumulation(self):
        self.assertEqual(
            effective_global_batch(batch_size=2048, world_size=2, grad_accum_steps=2),
            8192,
        )

    def test_shard_file_list_round_robin_mixes_order(self):
        file_list = [f'f{i}' for i in range(6)]
        self.assertEqual(
            shard_file_list_round_robin(file_list, rank=0, world_size=2),
            ['f0', 'f2', 'f4'],
        )
        self.assertEqual(
            shard_file_list_round_robin(file_list, rank=1, world_size=2),
            ['f1', 'f3', 'f5'],
        )

    def test_resolve_distributed_context_uses_local_rank_cuda_device(self):
        ctx = resolve_distributed_context(
            control_device='cuda:7',
            distributed_cfg={'backend': 'nccl'},
            env={'WORLD_SIZE': '2', 'RANK': '1', 'LOCAL_RANK': '0'},
            cuda_available=True,
        )
        self.assertTrue(ctx.enabled)
        self.assertEqual(ctx.device.type, 'cuda')
        self.assertEqual(ctx.device.index, 0)
        self.assertEqual(ctx.backend, 'nccl')

    def test_seed_everything_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, 'seed must be non-negative'):
            seed_everything(-1)

    def test_config_fingerprint_round_trip(self):
        full_config = {
            'resnet': {'conv_channels': 192, 'num_blocks': 40},
            'bc': {
                'control': {'batch_size': 2048},
                'dataset': {'train_list': 'train.txt'},
            },
        }
        fingerprint = config_fingerprint(full_config)
        self.assertEqual(
            stored_config_fingerprint({
                'config_fingerprint': fingerprint,
                'config': {'ignored': True},
            }),
            fingerprint,
        )

    def test_config_fingerprint_ignores_launch_and_wandb_cosmetics(self):
        base = {
            'resnet': {'conv_channels': 192, 'num_blocks': 40},
            'bc': {
                'control': {'batch_size': 2048},
                'dataset': {'train_list': 'train.txt'},
                'launch': {'campaign_summary_json': '/tmp/a.json'},
                'wandb': {'name': 'run-a'},
            },
        }
        variant = {
            'resnet': {'conv_channels': 192, 'num_blocks': 40},
            'bc': {
                'control': {'batch_size': 2048},
                'dataset': {'train_list': 'train.txt'},
                'launch': {'campaign_summary_json': '/tmp/b.json'},
                'wandb': {'name': 'run-b'},
            },
        }
        self.assertEqual(config_fingerprint(base), config_fingerprint(variant))

    def test_validate_resume_fingerprint_accepts_matching_config(self):
        full_config = {
            'resnet': {'conv_channels': 192, 'num_blocks': 40},
            'bc': {'control': {'batch_size': 2048}},
        }
        fingerprint = config_fingerprint(full_config)
        validate_resume_fingerprint(
            state={'config_fingerprint': fingerprint},
            current_fingerprint=fingerprint,
        )

    def test_validate_resume_fingerprint_rejects_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_resume_fingerprint(
                state={'config_fingerprint': 'abc'},
                current_fingerprint='def',
            )


if __name__ == '__main__':
    unittest.main()
