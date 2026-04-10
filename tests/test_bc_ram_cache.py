import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_ram_cache import (  # noqa: E402
    resolve_runtime_cache_settings,
    runtime_cache_enabled,
    runtime_cache_split_settings,
)


class BcRuntimeCacheHelpersTest(unittest.TestCase):
    def test_runtime_cache_split_settings_divides_train_budget_across_ranks(self):
        config = {
            'bc': {
                'runtime_cache': {
                    'enabled': True,
                    'node_ram_budget_gib': 160,
                    'node_pinned_budget_gib': 8,
                    'node_inflight_budget_gib': 8,
                    'raw_lru_budget_gib': 0,
                    'target_chunk_gib': 2,
                    'max_chunk_gib': 4,
                    'decode_threads': 4,
                    'max_inflight_chunk_builders': 1,
                    'min_files_per_chunk': 16,
                    'max_files_per_chunk': 96,
                },
            },
        }
        self.assertTrue(runtime_cache_enabled(config))
        settings = runtime_cache_split_settings(config, split_name='train', world_size=2)
        self.assertEqual(settings['data_budget_bytes'], 76 * (1024 ** 3))
        self.assertEqual(settings['ready_budget_bytes'], 72 * (1024 ** 3))
        self.assertEqual(settings['inflight_budget_bytes'], 4 * (1024 ** 3))
        self.assertEqual(settings['pinned_budget_bytes'], 4 * (1024 ** 3))
        self.assertEqual(settings['target_chunk_bytes'], 2 * (1024 ** 3))

    def test_runtime_cache_split_settings_uses_eval_budget_for_eval_splits(self):
        config = {
            'bc': {
                'runtime_cache': {
                    'enabled': True,
                    'eval_node_ram_budget_gib': 16,
                    'eval_target_chunk_gib': 1,
                    'eval_decode_threads': 2,
                },
            },
        }
        settings = runtime_cache_split_settings(config, split_name='val', world_size=2)
        self.assertEqual(settings['data_budget_bytes'], 16 * (1024 ** 3))
        self.assertEqual(settings['ready_budget_bytes'], 16 * (1024 ** 3))
        self.assertEqual(settings['target_chunk_bytes'], 1 * (1024 ** 3))
        self.assertEqual(settings['decode_threads'], 2)

    def test_resolve_runtime_cache_settings_fills_defaults(self):
        settings = resolve_runtime_cache_settings({'bc': {'runtime_cache': {'enabled': True}}})
        self.assertEqual(settings['mode'], 'prepared_ram')
        self.assertEqual(settings['node_ram_budget_gib'], 160)
        self.assertEqual(settings['target_chunk_gib'], 2)

    def test_runtime_cache_split_settings_clamps_producer_threads_to_one(self):
        config = {
            'bc': {
                'runtime_cache': {
                    'enabled': True,
                    'producer_threads': 8,
                },
            },
        }
        settings = runtime_cache_split_settings(config, split_name='train', world_size=2)
        self.assertEqual(settings['producer_threads'], 1)


if __name__ == '__main__':
    unittest.main()
