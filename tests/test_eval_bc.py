import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from eval_bc import make_summary, resolve_eval_file_list, split_sources  # noqa: E402


class EvalBcHelpersTest(unittest.TestCase):
    def test_split_sources_selects_expected_keys(self):
        dataset_cfg = {
            'train_list': 'train.txt',
            'val_list': 'val.txt',
            'test_list': 'test.txt',
            'train_globs': ['train/**/*.json.gz'],
            'val_globs': ['val/**/*.json.gz'],
            'test_globs': ['test/**/*.json.gz'],
        }
        self.assertEqual(split_sources(dataset_cfg, 'train'), ('train.txt', ['train/**/*.json.gz']))
        self.assertEqual(split_sources(dataset_cfg, 'val'), ('val.txt', ['val/**/*.json.gz']))
        self.assertEqual(split_sources(dataset_cfg, 'test'), ('test.txt', ['test/**/*.json.gz']))

    def test_resolve_eval_file_list_reads_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            list_file = root / 'val.txt'
            data_file = root / 'logs' / 'game.json.gz'
            data_file.parent.mkdir(parents=True)
            data_file.write_text('', encoding='utf-8')
            list_file.write_text('logs/game.json.gz\n', encoding='utf-8')

            resolved = resolve_eval_file_list(
                {'val_list': str(list_file), 'val_globs': []},
                'val',
                str(root),
            )
            self.assertEqual(resolved, [str(data_file)])

    def test_resolve_eval_file_list_requires_configured_split(self):
        with self.assertRaisesRegex(ValueError, 'has no configured test split'):
            resolve_eval_file_list({'test_list': '', 'test_globs': []}, 'test', '')

    def test_make_summary_preserves_core_fields(self):
        summary = make_summary(
            checkpoint='/tmp/best.pth',
            split='test',
            file_count=12,
            batch_count=4,
            max_batches=64,
            metrics={'accuracy': 0.5},
            state={'trainer': 'behavior_cloning', 'steps': 100, 'best_perf': {'val_accuracy': 0.5}},
        )
        self.assertEqual(summary['checkpoint'], '/tmp/best.pth')
        self.assertEqual(summary['split'], 'test')
        self.assertEqual(summary['file_count'], 12)
        self.assertEqual(summary['batch_count'], 4)
        self.assertEqual(summary['max_batches'], 64)
        self.assertTrue(summary['is_capped_eval'])
        self.assertEqual(summary['metrics']['accuracy'], 0.5)
        self.assertEqual(summary['trainer'], 'behavior_cloning')
        self.assertEqual(summary['steps'], 100)


if __name__ == '__main__':
    unittest.main()
