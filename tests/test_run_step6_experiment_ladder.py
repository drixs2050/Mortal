import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'
MODULE_PATH = ROOT / 'scripts' / 'run_step6_experiment_ladder.py'
os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))
SPEC = importlib.util.spec_from_file_location('run_step6_experiment_ladder', MODULE_PATH)
run_step6_experiment_ladder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_step6_experiment_ladder)


class RunStep6ExperimentLadderHelpersTest(unittest.TestCase):
    def test_base_loader_knobs_match_phase_a_baseline(self):
        knobs = run_step6_experiment_ladder.base_loader_knobs()
        self.assertEqual(knobs['backend'], 'prepared_ram')
        self.assertEqual(knobs['target_chunk_gib'], 2.0)
        self.assertEqual(knobs['decode_threads'], 4)
        self.assertEqual(knobs['max_inflight_chunk_builders'], 2)
        self.assertEqual(knobs['startup_ready_chunks'], 4)
        self.assertEqual(knobs['device_prefetch_batches'], 2)
        self.assertEqual(knobs['raw_lru_budget_gib'], 0)

    def test_candidate_better_for_phase_a_prefers_lower_wait_before_throughput(self):
        baseline = {
            'preflight_return_code': 0,
            'gate_passed': False,
            'loader_wait_fraction': 0.30,
            'samples_per_second': 5000.0,
            'peak_combined_rss_gib': 80.0,
            'startup_seconds': 12.0,
        }
        candidate = {
            'preflight_return_code': 0,
            'gate_passed': False,
            'loader_wait_fraction': 0.20,
            'samples_per_second': 4900.0,
            'peak_combined_rss_gib': 90.0,
            'startup_seconds': 14.0,
        }
        self.assertTrue(
            run_step6_experiment_ladder.candidate_better_for_phase_a(
                baseline=baseline,
                candidate=candidate,
            )
        )

    def test_visible_device_count_uses_visible_a100_pair(self):
        self.assertEqual(run_step6_experiment_ladder.visible_device_count('0,1'), 2)
        self.assertEqual(run_step6_experiment_ladder.visible_device_count('0'), 1)

    def test_inflight_builder_budget_supported_requires_budget_for_requested_builders(self):
        baseline_knobs = run_step6_experiment_ladder.base_loader_knobs()
        self.assertTrue(
            run_step6_experiment_ladder.inflight_builder_budget_supported(
                knobs=baseline_knobs,
                world_size=2,
            )
        )
        inflight_four_knobs = dict(baseline_knobs)
        inflight_four_knobs['max_inflight_chunk_builders'] = 4
        self.assertFalse(
            run_step6_experiment_ladder.inflight_builder_budget_supported(
                knobs=inflight_four_knobs,
                world_size=2,
            )
        )

    def test_model_beats_baseline_requires_quality_and_throughput_gates(self):
        baseline = {
            'best_accuracy': 0.8100,
            'samples_per_second': 6000.0,
        }
        winning_candidate = {
            'status': 'completed',
            'best_accuracy': 0.8135,
            'samples_per_second': 4700.0,
        }
        losing_candidate = {
            'status': 'completed',
            'best_accuracy': 0.8120,
            'samples_per_second': 5000.0,
        }
        self.assertTrue(
            run_step6_experiment_ladder.model_beats_baseline(
                baseline_row=baseline,
                candidate_row=winning_candidate,
            )
        )
        self.assertFalse(
            run_step6_experiment_ladder.model_beats_baseline(
                baseline_row=baseline,
                candidate_row=losing_candidate,
            )
        )

    def test_persist_phase_progress_writes_state_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = {'created_at': '2026-03-31T00:00:00+00:00', 'phases': {}}
            state_path = tmp / 'state.json'
            payload = run_step6_experiment_ladder.persist_phase_progress(
                state=state,
                state_path=state_path,
                report_dir=tmp / 'reports',
                phase_name='phase_a',
                rows=[{'name': 'baseline', 'gate_passed': True}],
                experiments=[{'name': 'baseline'}],
                winner={'row': {'name': 'baseline'}, 'knobs': {'backend': 'npy_shards'}},
                columns=[('name', 'Name'), ('gate_passed', 'Gate')],
                extra_payload={'subset_metadata': {'train_file_count': 10}},
            )
            self.assertTrue(state_path.exists())
            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual(persisted['phases']['phase_a']['winner']['row']['name'], 'baseline')
            self.assertTrue(Path(payload['reports']['json_path']).exists())
            self.assertTrue(Path(payload['reports']['markdown_path']).exists())

    def test_load_phase_a_preflight_checkpoint_reuses_measured_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / 'phase_a_subset'
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / 'config.toml'
            config_path.write_text(
                "[bc.preflight]\nsummary_json = 'preflight_summary.json'\n",
                encoding='utf-8',
            )
            summary_path = run_dir / 'preflight_summary.json'
            summary_path.write_text(
                json.dumps({'metrics_event_count': 2, 'startup': {'startup_seconds': 3.2}}),
                encoding='utf-8',
            )
            phase_state = {
                'rows': [
                    {
                        'name': 'phase_a_subset',
                        'config_path': str(config_path),
                        'config_sha256': run_step6_experiment_ladder.file_sha256(config_path),
                        'preflight_return_code': 1,
                        'startup_seconds': 3.2,
                    }
                ],
                'experiments': [
                    {
                        'name': 'phase_a_subset',
                        'knobs': {
                            'backend': 'prepared_ram',
                            'target_chunk_gib': 2.0,
                            'decode_threads': 4,
                            'max_inflight_chunk_builders': 1,
                            'device_prefetch_batches': 2,
                        },
                    }
                ],
            }
            reused = run_step6_experiment_ladder.load_phase_a_preflight_checkpoint(
                phase_state=phase_state,
                experiment_name='phase_a_subset',
                config_path=config_path,
                full_config={'bc': {'preflight': {'summary_json': str(summary_path)}}},
            )
            self.assertIsNotNone(reused)
            self.assertEqual(reused['row']['name'], 'phase_a_subset')
            self.assertAlmostEqual(reused['row']['startup_seconds'], 3.2)

    def test_load_phase_a_preflight_checkpoint_skips_stale_non_measured_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / 'phase_a_subset'
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / 'config.toml'
            config_path.write_text(
                "[bc.preflight]\nsummary_json = 'preflight_summary.json'\n",
                encoding='utf-8',
            )
            summary_path = run_dir / 'preflight_summary.json'
            summary_path.write_text(
                json.dumps({'metrics_event_count': 0, 'startup': None}),
                encoding='utf-8',
            )
            phase_state = {
                'rows': [
                    {
                        'name': 'phase_a_subset',
                        'config_path': str(config_path),
                        'config_sha256': run_step6_experiment_ladder.file_sha256(config_path),
                        'preflight_return_code': 1,
                        'startup_seconds': None,
                        'samples_per_second': None,
                    }
                ],
                'experiments': [
                    {
                        'name': 'phase_a_subset',
                        'knobs': {
                            'backend': 'prepared_ram',
                            'target_chunk_gib': 2.0,
                            'decode_threads': 4,
                            'max_inflight_chunk_builders': 1,
                            'device_prefetch_batches': 2,
                        },
                    }
                ],
            }
            reused = run_step6_experiment_ladder.load_phase_a_preflight_checkpoint(
                phase_state=phase_state,
                experiment_name='phase_a_subset',
                config_path=config_path,
                full_config={'bc': {'preflight': {'summary_json': str(summary_path)}}},
            )
            self.assertIsNone(reused)

    def test_load_phase_a_preflight_checkpoint_refreshes_stale_row_from_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / 'phase_a_subset'
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / 'config.toml'
            config_path.write_text(
                "[bc.preflight]\nsummary_json = 'preflight_summary.json'\n",
                encoding='utf-8',
            )
            summary_path = run_dir / 'preflight_summary.json'
            summary_path.write_text(
                json.dumps(
                    {
                        'return_code': 0,
                        'status': 'failed',
                        'sustained_metrics': {
                            'samples_per_second': 6883.0,
                            'wait_fraction': 0.145,
                        },
                        'steady_gpu': {
                            'pass_ratio': 0.0,
                        },
                        'startup': {
                            'startup_seconds': 2.54,
                        },
                        'gate': {
                            'passed': False,
                            'reasons': ['steady_gpu_ratio_below_gate=0.000<0.700'],
                        },
                        'max_train_worker_rss_kib': 81245316,
                        'max_combined_train_worker_rss_kib': 162698684,
                    }
                ),
                encoding='utf-8',
            )
            phase_state = {
                'rows': [
                    {
                        'name': 'phase_a_subset',
                        'config_path': str(config_path),
                        'config_sha256': run_step6_experiment_ladder.file_sha256(config_path),
                        'preflight_return_code': 1,
                        'startup_seconds': None,
                        'samples_per_second': None,
                    }
                ],
                'experiments': [
                    {
                        'name': 'phase_a_subset',
                        'knobs': {
                            'backend': 'prepared_ram',
                            'target_chunk_gib': 2.0,
                            'decode_threads': 4,
                            'max_inflight_chunk_builders': 1,
                            'device_prefetch_batches': 2,
                        },
                    }
                ],
            }
            reused = run_step6_experiment_ladder.load_phase_a_preflight_checkpoint(
                phase_state=phase_state,
                experiment_name='phase_a_subset',
                config_path=config_path,
                full_config={'bc': {'preflight': {'summary_json': str(summary_path)}}},
            )
            self.assertIsNotNone(reused)
            self.assertEqual(reused['row']['preflight_return_code'], 0)
            self.assertAlmostEqual(reused['row']['samples_per_second'], 6883.0)
            self.assertAlmostEqual(reused['row']['loader_wait_fraction'], 0.145)
            self.assertAlmostEqual(reused['row']['startup_seconds'], 2.54)


if __name__ == '__main__':
    unittest.main()
