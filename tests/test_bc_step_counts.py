import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_step_counts import (  # noqa: E402
    batch_count_for_steps,
    build_step_count_summary,
    expected_batches_from_summary,
    load_step_count_summary,
    save_step_count_summary,
)


class BcStepCountsTest(unittest.TestCase):
    def test_batch_count_for_steps_rounds_up(self):
        self.assertEqual(batch_count_for_steps(0, 2048), 0)
        self.assertEqual(batch_count_for_steps(1, 2048), 1)
        self.assertEqual(batch_count_for_steps(2048, 2048), 1)
        self.assertEqual(batch_count_for_steps(2049, 2048), 2)

    def test_save_and_load_step_count_summary_round_trip(self):
        payload = {
            'format': 'bc_step_counts_v1',
            'config': {'batch_size_reference': 2048},
            'splits': {
                'val': {
                    'requested_file_count': 12,
                    'step_count': 4097,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / 'step_counts_v1.json'
            save_step_count_summary(str(summary_path), payload)
            loaded = load_step_count_summary(str(summary_path))
        self.assertEqual(loaded, payload)

    def test_expected_batches_from_summary_uses_file_count_guard(self):
        payload = {
            'format': 'bc_step_counts_v1',
            'config': {},
            'splits': {
                'val': {
                    'requested_file_count': 10,
                    'step_count': 5000,
                },
            },
        }
        self.assertEqual(
            expected_batches_from_summary(
                payload,
                split='val',
                batch_size=2048,
                file_count=10,
            ),
            (5000, 3),
        )
        self.assertIsNone(
            expected_batches_from_summary(
                payload,
                split='val',
                batch_size=2048,
                file_count=9,
            )
        )
        self.assertEqual(
            expected_batches_from_summary(
                payload,
                split='val',
                batch_size=2048,
                file_count=10,
                max_batches=2,
            ),
            (5000, 2),
        )

    def test_build_step_count_summary_falls_back_to_single_process(self):
        split_lists = {'val': ['/tmp/a.json.gz', '/tmp/b.json.gz']}
        with patch('bc_step_counts._count_split_steps_parallel', side_effect=RuntimeError('boom')):
            with patch(
                'bc_step_counts._count_split_steps_serial',
                return_value={
                    'requested_file_count': 2,
                    'loaded_file_count': 2,
                    'nonempty_file_count': 2,
                    'trajectory_count': 3,
                    'step_count': 4097,
                    'skipped_file_count': 0,
                },
            ):
                summary = build_step_count_summary(
                    split_lists=split_lists,
                    version=4,
                    oracle=False,
                    file_batch_size=12,
                    player_names=None,
                    excludes=None,
                    trust_seed=False,
                    always_include_kan_select=True,
                    actor_filter_map=None,
                    batch_size_reference=2048,
                    jobs=16,
                    chunk_size=128,
                    config_summary={'jobs': 16},
                )
        self.assertTrue(summary['splits']['val']['fell_back_to_single_process'])
        self.assertEqual(summary['splits']['val']['effective_jobs'], 1)
        self.assertEqual(summary['splits']['val']['batch_count_reference'], 3)


if __name__ == '__main__':
    unittest.main()
