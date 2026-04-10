import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from step6_experiments import (  # noqa: E402
    average_wait_fraction_after_step,
    candidate_improves,
    deterministic_round_robin_sample,
    loader_knob_overrides,
    phase4_candidate_beats_control,
    phase4_preassembled_overrides,
    phase3_worker_overrides,
    phase1_candidate_beats_control,
    phase2_candidate_beats_control,
    phase_in_window,
    phase_runtime_overrides,
    raw_source_backend_overrides,
    raw_threaded_queue_overrides,
    row_meets_loader_decision_gate,
    select_phase1_batch_count_winner,
    should_run_thread_comparison,
    summarize_phase1_queue_row,
    summarize_preflight_row,
    validate_phase_window,
)


class Step6ExperimentsHelpersTest(unittest.TestCase):
    def test_deterministic_round_robin_sample_spreads_across_full_list(self):
        items = [f'f{i}' for i in range(10)]
        sample = deterministic_round_robin_sample(items, 4)
        self.assertEqual(sample, ['f0', 'f2', 'f5', 'f7'])

    def test_validate_phase_window_rejects_reverse_range(self):
        with self.assertRaisesRegex(ValueError, 'invalid phase window'):
            validate_phase_window(start_at='phase_c', stop_after='phase_b')

    def test_phase_in_window_respects_bounds(self):
        self.assertTrue(phase_in_window(phase='phase_b', start_at='phase_a', stop_after='phase_c'))
        self.assertFalse(phase_in_window(phase='phase_d', start_at='phase_a', stop_after='phase_c'))

    def test_average_wait_fraction_after_step_uses_late_windows_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / 'metrics.jsonl'
            metrics_path.write_text(
                '\n'.join([
                    '{"event":"train_live","step":100,"loader_metrics":{"wait_fraction":0.20}}',
                    '{"event":"train_live","step":125,"loader_metrics":{"wait_fraction":0.10}}',
                    '{"event":"save_window","step":150,"loader_metrics":{"wait_fraction":0.14}}',
                ]) + '\n',
                encoding='utf-8',
            )
            self.assertAlmostEqual(
                average_wait_fraction_after_step(metrics_path, min_step=125),
                0.12,
            )

    def test_phase_runtime_overrides_isolates_file_index_and_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overrides = phase_runtime_overrides(run_root=tmpdir, experiment_name='phase_a_baseline')
            self.assertTrue(
                str(overrides['bc']['dataset']['file_index']).endswith('phase_a_baseline/file_index.pth')
            )
            self.assertTrue(
                str(overrides['bc']['preflight']['summary_json']).endswith(
                    'phase_a_baseline/preflight_summary.json'
                )
            )

    def test_loader_knob_overrides_sets_runtime_cache_and_device_prefetch_fields(self):
        overrides = loader_knob_overrides(
            knobs={
                'target_chunk_gib': 4,
                'decode_threads': 8,
                'max_inflight_chunk_builders': 2,
                'raw_lru_budget_gib': 4,
                'device_prefetch_batches': 3,
            },
            cache_root='/tmp/stage-cache',
            required_splits=['train'],
        )
        self.assertFalse(overrides['bc']['stage']['enabled'])
        self.assertTrue(overrides['bc']['runtime_cache']['enabled'])
        self.assertEqual(overrides['bc']['runtime_cache']['target_chunk_gib'], 4.0)
        self.assertEqual(overrides['bc']['runtime_cache']['decode_threads'], 8)
        self.assertEqual(overrides['bc']['runtime_cache']['max_inflight_chunk_builders'], 2)
        self.assertEqual(overrides['bc']['runtime_cache']['raw_lru_budget_gib'], 4)
        self.assertEqual(overrides['bc']['dataset']['device_prefetch_batches'], 3)

    def test_raw_threaded_queue_overrides_freeze_control_path(self):
        overrides = raw_threaded_queue_overrides(
            cpu_ready_batches=12,
            cpu_ready_bytes_gib=8.0,
        )
        self.assertEqual(overrides['bc']['dataset']['cpu_batch_pipe_backend'], 'thread')
        self.assertEqual(overrides['bc']['dataset']['cpu_ready_batches'], 12)
        self.assertAlmostEqual(overrides['bc']['dataset']['cpu_ready_bytes_gib'], 8.0)
        self.assertEqual(overrides['bc']['dataset']['raw_source_backend'], 'files')
        self.assertEqual(overrides['bc']['dataset']['loader_mode'], 'baseline')
        self.assertFalse(overrides['bc']['runtime_cache']['enabled'])
        self.assertFalse(overrides['bc']['stage']['enabled'])

    def test_raw_source_backend_overrides_sets_raw_pack_fields(self):
        overrides = raw_source_backend_overrides(
            backend='raw_pack',
            raw_pack_path='/tmp/data.raw.pack',
            raw_pack_index_path='/tmp/data.raw.index.json',
        )
        self.assertEqual(overrides['bc']['dataset']['raw_source_backend'], 'raw_pack')
        self.assertEqual(overrides['bc']['dataset']['raw_pack_path'], '/tmp/data.raw.pack')
        self.assertEqual(
            overrides['bc']['dataset']['raw_pack_index_path'],
            '/tmp/data.raw.index.json',
        )

    def test_phase3_worker_overrides_freeze_low_code_worker_branch(self):
        overrides = phase3_worker_overrides(
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
            in_order=True,
            multiprocessing_context='spawn',
        )
        dataset_cfg = overrides['bc']['dataset']
        self.assertEqual(dataset_cfg['num_workers'], 4)
        self.assertEqual(dataset_cfg['cpu_batch_pipe_backend'], 'thread')
        self.assertTrue(dataset_cfg['persistent_workers'])
        self.assertEqual(dataset_cfg['prefetch_factor'], 2)
        self.assertTrue(dataset_cfg['in_order'])
        self.assertEqual(dataset_cfg['multiprocessing_context'], 'spawn')
        self.assertEqual(dataset_cfg['raw_source_backend'], 'files')
        self.assertEqual(dataset_cfg['loader_mode'], 'baseline')
        self.assertFalse(overrides['bc']['stage']['enabled'])
        self.assertFalse(overrides['bc']['runtime_cache']['enabled'])

    def test_phase4_preassembled_overrides_freeze_single_process_rewrite_branch(self):
        overrides = phase4_preassembled_overrides(loader_block_target_samples=32768)
        dataset_cfg = overrides['bc']['dataset']
        self.assertEqual(dataset_cfg['num_workers'], 0)
        self.assertEqual(dataset_cfg['eval_num_workers'], 0)
        self.assertEqual(dataset_cfg['cpu_batch_pipe_backend'], 'thread')
        self.assertEqual(dataset_cfg['raw_source_backend'], 'files')
        self.assertEqual(dataset_cfg['loader_mode'], 'preassembled_batches')
        self.assertEqual(dataset_cfg['loader_block_target_samples'], 32768)
        self.assertFalse(overrides['bc']['stage']['enabled'])
        self.assertFalse(overrides['bc']['runtime_cache']['enabled'])

    def test_candidate_improves_can_reject_rss_growth(self):
        baseline = {
            'samples_per_second': 6000.0,
            'loader_wait_fraction': 0.10,
            'steady_gpu_ratio': 0.8,
            'startup_seconds': 10.0,
            'peak_combined_rss_gib': 100.0,
            'gate_passed': True,
        }
        candidate = {
            'samples_per_second': 6300.0,
            'loader_wait_fraction': 0.09,
            'steady_gpu_ratio': 0.85,
            'startup_seconds': 10.5,
            'peak_combined_rss_gib': 120.0,
            'gate_passed': True,
        }
        self.assertFalse(
            candidate_improves(
                baseline=baseline,
                candidate=candidate,
                min_relative_gain=0.03,
                max_rss_growth_ratio=0.05,
                max_startup_growth_seconds=2.0,
            )
        )

    def test_phase4_candidate_beats_control_accepts_large_sps_gain(self):
        control = {
            'samples_per_second': 8000.0,
            'collate_or_assemble_fraction': 0.40,
            'preflight_return_code': 0,
        }
        candidate = {
            'samples_per_second': 8405.0,
            'collate_or_assemble_fraction': 0.39,
            'preflight_return_code': 0,
        }
        self.assertTrue(phase4_candidate_beats_control(control=control, candidate=candidate))

    def test_phase4_candidate_beats_control_accepts_collate_drop_with_flat_sps(self):
        control = {
            'samples_per_second': 8000.0,
            'collate_or_assemble_fraction': 0.45,
            'preflight_return_code': 0,
        }
        candidate = {
            'samples_per_second': 7930.0,
            'collate_or_assemble_fraction': 0.30,
            'preflight_return_code': 0,
        }
        self.assertTrue(phase4_candidate_beats_control(control=control, candidate=candidate))

    def test_loader_decision_gate_allows_gpu_only_advisory_failure(self):
        row = {
            'preflight_return_code': 0,
            'samples_per_second': 6883.0,
            'loader_wait_fraction': 0.145,
            'startup_seconds': 2.54,
            'fail_reasons': ['steady_gpu_ratio_below_gate=0.000<0.700'],
        }
        self.assertTrue(row_meets_loader_decision_gate(row))
        self.assertFalse(should_run_thread_comparison(row))

    def test_loader_decision_gate_rejects_real_failures(self):
        row = {
            'preflight_return_code': 1,
            'samples_per_second': 6883.0,
            'loader_wait_fraction': 0.145,
            'startup_seconds': 2.54,
            'fail_reasons': ['train_return_code=1'],
        }
        self.assertFalse(row_meets_loader_decision_gate(row))

    def test_summarize_preflight_row_prefers_completed_window_for_reporting(self):
        row = summarize_preflight_row(
            phase='phase_a',
            name='baseline',
            knobs={'backend': 'prepared_ram', 'device_prefetch_batches': 2},
            stage_summary=None,
            preflight_summary={
                'completed_step': 200,
                'completed_window_metrics': {
                    'samples_per_second': 6400.0,
                    'steps_per_second': 0.39,
                    'wait_fraction': 0.12,
                    'cpu_pipe_wait_fraction': 0.04,
                },
                'sustained_metrics': {
                    'samples_per_second': 7100.0,
                    'wait_fraction': 0.08,
                },
                'steady_gpu': {'pass_ratio': 0.9},
                'startup': {'startup_seconds': 6.0, 'loader_snapshot': {}},
                'gate': {'passed': True, 'reasons': []},
            },
        )
        self.assertEqual(row['completed_step'], 200)
        self.assertEqual(row['measurement_source'], 'completed_window')
        self.assertAlmostEqual(row['samples_per_second'], 6400.0)
        self.assertAlmostEqual(row['loader_wait_fraction'], 0.12)
        self.assertAlmostEqual(row['gate_samples_per_second'], 7100.0)
        self.assertAlmostEqual(row['gate_loader_wait_fraction'], 0.08)
        self.assertTrue(row_meets_loader_decision_gate(row))

    def test_summarize_phase1_queue_row_reports_targets(self):
        row = summarize_phase1_queue_row(
            name='queue_b12_cap8g',
            knobs={
                'cpu_ready_batches': 12,
                'cpu_ready_bytes_gib': 8.0,
            },
            preflight_summary={
                'completed_step': 200,
                'completed_window_metrics': {
                    'samples_per_second': 8400.0,
                    'steps_per_second': 0.51,
                    'wait_fraction': 0.11,
                    'cpu_ready_bytes_gib': 6.5,
                },
                'steady_gpu': {'pass_ratio': 0.61},
                'startup': {'startup_seconds': 7.0, 'loader_snapshot': {}},
                'gate': {'passed': False, 'reasons': ['steady_gpu_ratio_below_gate=0.610<0.700']},
                'return_code': 0,
            },
        )
        self.assertEqual(row['backend'], 'raw_threaded')
        self.assertEqual(row['cpu_ready_batches_target'], 12)
        self.assertAlmostEqual(row['cpu_ready_bytes_target_gib'], 8.0)
        self.assertAlmostEqual(row['samples_per_second'], 8400.0)

    def test_phase1_candidate_beats_control_accepts_wait_drop_with_small_sps_regression(self):
        control = {
            'samples_per_second': 8086.23,
            'loader_wait_fraction': 0.1329,
            'preflight_return_code': 0,
        }
        candidate = {
            'samples_per_second': 8040.0,
            'loader_wait_fraction': 0.095,
            'preflight_return_code': 0,
        }
        self.assertTrue(
            phase1_candidate_beats_control(
                control=control,
                candidate=candidate,
            )
        )

    def test_phase2_candidate_beats_control_accepts_raw_read_drop(self):
        control = {
            'samples_per_second': 8080.0,
            'raw_read_fraction': 0.45,
            'preflight_return_code': 0,
        }
        candidate = {
            'samples_per_second': 8000.0,
            'raw_read_fraction': 0.34,
            'preflight_return_code': 0,
        }
        self.assertTrue(
            phase2_candidate_beats_control(
                control=control,
                candidate=candidate,
            )
        )

    def test_select_phase1_batch_count_winner_prefers_fast_successful_row(self):
        winner = select_phase1_batch_count_winner([
            {
                'name': 'control_b4',
                'preflight_return_code': 0,
                'samples_per_second': 8086.0,
                'loader_wait_fraction': 0.13,
                'steady_gpu_ratio': 0.39,
                'startup_seconds': 7.0,
            },
            {
                'name': 'queue_b8',
                'preflight_return_code': 0,
                'samples_per_second': 8300.0,
                'loader_wait_fraction': 0.12,
                'steady_gpu_ratio': 0.41,
                'startup_seconds': 7.5,
            },
            {
                'name': 'queue_b12',
                'preflight_return_code': 1,
                'samples_per_second': 9000.0,
                'loader_wait_fraction': 0.50,
                'steady_gpu_ratio': 0.10,
                'startup_seconds': 20.0,
            },
        ])
        self.assertEqual(winner['name'], 'queue_b8')


if __name__ == '__main__':
    unittest.main()
