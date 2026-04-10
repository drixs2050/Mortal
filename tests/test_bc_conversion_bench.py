import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

from bc_conversion_bench import compare_pipeline_stages, compare_producer_vs_training, select_benchmark_files  # noqa: E402


class BenchmarkSelectionTest(unittest.TestCase):
    def test_select_benchmark_files_round_robin(self):
        items = [f'f{i}' for i in range(10)]
        self.assertEqual(
            select_benchmark_files(items, sample_size=4, sample_strategy='round_robin'),
            ['f0', 'f2', 'f5', 'f7'],
        )

    def test_select_benchmark_files_head(self):
        items = [f'f{i}' for i in range(10)]
        self.assertEqual(
            select_benchmark_files(items, sample_size=3, sample_strategy='head'),
            ['f0', 'f1', 'f2'],
        )


class BenchmarkComparisonTest(unittest.TestCase):
    def test_compare_pipeline_stages_reports_hideability(self):
        comparison = compare_pipeline_stages(
            conversion_summary={'elapsed_seconds': 8.0, 'samples_per_second': 1000.0},
            materialization_summary={'elapsed_seconds': 2.0, 'sample_count': 10_000},
            training_summary={'elapsed_seconds': 12.0, 'samples_per_second': 900.0},
        )
        self.assertAlmostEqual(comparison['conversion_only_vs_training_time_ratio'], 8.0 / 12.0)
        self.assertAlmostEqual(comparison['conversion_plus_batching_vs_training_time_ratio'], 10.0 / 12.0)
        self.assertTrue(comparison['can_hide_conversion_only'])
        self.assertTrue(comparison['can_hide_conversion_plus_batching'])
        self.assertFalse(comparison['conversion_bottleneck'])

    def test_compare_pipeline_stages_detects_bottleneck(self):
        comparison = compare_pipeline_stages(
            conversion_summary={'elapsed_seconds': 14.0, 'samples_per_second': 700.0},
            materialization_summary={'elapsed_seconds': 3.0, 'sample_count': 10_000},
            training_summary={'elapsed_seconds': 10.0, 'samples_per_second': 1000.0},
        )
        self.assertFalse(comparison['can_hide_conversion_only'])
        self.assertFalse(comparison['can_hide_conversion_plus_batching'])
        self.assertTrue(comparison['conversion_bottleneck'])

    def test_compare_producer_vs_training(self):
        comparison = compare_producer_vs_training(
            producer_summary={'elapsed_seconds': 9.0, 'samples_per_second': 1200.0},
            training_summary={'elapsed_seconds': 12.0, 'samples_per_second': 1000.0},
        )
        self.assertAlmostEqual(comparison['producer_vs_training_time_ratio'], 9.0 / 12.0)
        self.assertAlmostEqual(comparison['producer_vs_training_sps_ratio'], 1.2)
        self.assertTrue(comparison['can_hide_producer'])
        self.assertFalse(comparison['producer_bottleneck'])

    def test_compare_producer_vs_training_reports_post_warmup_reference(self):
        comparison = compare_producer_vs_training(
            producer_summary={'elapsed_seconds': 9.0, 'samples_per_second': 1200.0},
            training_summary={'elapsed_seconds': 12.0, 'samples_per_second': 1000.0, 'sample_count': 12_000},
            warmup_summary={'post_warmup_samples_per_second': 1500.0},
        )
        self.assertAlmostEqual(comparison['post_warmup_training_seconds_estimate'], 8.0)
        self.assertAlmostEqual(comparison['producer_vs_post_warmup_training_time_ratio'], 9.0 / 8.0)
        self.assertAlmostEqual(comparison['producer_vs_post_warmup_training_sps_ratio'], 1200.0 / 1500.0)
        self.assertFalse(comparison['can_hide_producer_vs_post_warmup_training'])
