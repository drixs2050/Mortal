import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'scripts' / 'run_bc_loader_preflight.py'
SPEC = importlib.util.spec_from_file_location('run_bc_loader_preflight', MODULE_PATH)
run_bc_loader_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_bc_loader_preflight)


class RunBcLoaderPreflightHelpersTest(unittest.TestCase):
    def test_parse_visible_gpu_indices(self):
        self.assertEqual(run_bc_loader_preflight.parse_visible_gpu_indices('0,1'), [0, 1])
        self.assertEqual(run_bc_loader_preflight.parse_visible_gpu_indices(''), [])

    def test_steady_gpu_ratio_falls_back_to_sampled_gpu_rows(self):
        ratio = run_bc_loader_preflight.steady_gpu_ratio(
            samples=[
                {
                    'elapsed_seconds': 300.0,
                    'gpus': {
                        0: {'power_draw_watts': 220.0, 'utilization_gpu': 92.0},
                        2: {'power_draw_watts': 218.0, 'utilization_gpu': 89.0},
                    },
                },
            ],
            target_gpu_indices=[0, 1],
            target_pci_bus_ids=None,
            min_runtime_seconds=240.0,
            min_power_watts=150.0,
            min_utilization=60.0,
        )
        self.assertEqual(ratio['sample_count'], 1)
        self.assertEqual(ratio['matched_sample_count'], 1)
        self.assertEqual(ratio['passing_count'], 1)
        self.assertAlmostEqual(ratio['pass_ratio'], 1.0)

    def test_steady_gpu_ratio_matches_by_normalized_pci_bus_id(self):
        ratio = run_bc_loader_preflight.steady_gpu_ratio(
            samples=[
                {
                    'elapsed_seconds': 300.0,
                    'gpus': {
                        1: {'pci_bus_id': '00000000:01:00.0', 'power_draw_watts': 220.0, 'utilization_gpu': 92.0},
                        2: {'pci_bus_id': '00000000:79:00.0', 'power_draw_watts': 218.0, 'utilization_gpu': 89.0},
                    },
                },
            ],
            target_gpu_indices=[0, 1],
            target_pci_bus_ids=['1', '79'],
            min_runtime_seconds=240.0,
            min_power_watts=150.0,
            min_utilization=60.0,
        )
        self.assertEqual(ratio['matched_sample_count'], 1)
        self.assertEqual(ratio['passing_count'], 1)

    def test_summarize_preflight_passes_when_gates_are_met(self):
        summary = run_bc_loader_preflight.summarize_preflight(
            config_path='/tmp/preflight.toml',
            config_fingerprint_value='fingerprint',
            started_at='2026-03-30T10:00:00+00:00',
            finished_at='2026-03-30T10:08:00+00:00',
            command=['torchrun', 'mortal/train_bc.py'],
            return_code=0,
            metrics_events=[
                {
                    'event': 'loader_priming',
                    'split': 'train',
                    'startup_seconds': 12.5,
                    'loader_snapshot': {'loaded_shards': 3},
                },
                {
                    'event': 'train_live',
                    'step': 100,
                    'runtime_seconds_total': 300.0,
                    'runtime_metrics': {'samples_per_second': 6100.0, 'steps_per_second': 0.37},
                    'loader_metrics': {'wait_fraction': 0.11, 'queued_bytes_gib': 12.0},
                },
                {
                    'event': 'train_live',
                    'step': 125,
                    'runtime_seconds_total': 380.0,
                    'runtime_metrics': {'samples_per_second': 6200.0, 'steps_per_second': 0.38},
                    'loader_metrics': {'wait_fraction': 0.10, 'queued_bytes_gib': 12.5},
                },
                {
                    'event': 'save_window',
                    'step': 200,
                    'stop_reason': 'max_steps',
                    'runtime_seconds_total': 420.0,
                    'runtime_metrics': {'samples_per_second': 6125.0, 'steps_per_second': 0.374},
                    'loader_metrics': {'wait_fraction': 0.108, 'queued_bytes_gib': 12.5},
                },
            ],
            gpu_samples=[
                {
                    'elapsed_seconds': 310.0,
                    'gpus': {
                        0: {'pci_bus_id': '00000000:01:00.0', 'power_draw_watts': 220.0, 'utilization_gpu': 92.0},
                        1: {'pci_bus_id': '00000000:79:00.0', 'power_draw_watts': 215.0, 'utilization_gpu': 90.0},
                    },
                },
                {
                    'elapsed_seconds': 360.0,
                    'gpus': {
                        0: {'pci_bus_id': '00000000:01:00.0', 'power_draw_watts': 225.0, 'utilization_gpu': 88.0},
                        1: {'pci_bus_id': '00000000:79:00.0', 'power_draw_watts': 218.0, 'utilization_gpu': 87.0},
                    },
                },
            ],
            target_gpu_indices=[0, 1],
            target_pci_bus_ids=['1', '79'],
            preflight_cfg={
                'min_runtime_seconds': 240,
                'min_steps_before_stop': 200,
                'required_stable_windows': 2,
                'min_samples_per_second': 5900,
                'preferred_samples_per_second': 7000,
                'max_loader_wait_fraction': 0.15,
                'min_steady_gpu_watts': 150,
                'min_steady_gpu_utilization': 60,
                'min_steady_gpu_ratio': 0.70,
            },
            max_worker_rss_kib=123456,
            max_combined_worker_rss_kib=456789,
        )
        self.assertEqual(summary['status'], 'passed')
        self.assertTrue(summary['gate']['passed'])
        self.assertAlmostEqual(summary['sustained_metrics']['samples_per_second'], 6150.0)
        self.assertEqual(summary['completed_step'], 200)
        self.assertAlmostEqual(summary['completed_window_metrics']['samples_per_second'], 6125.0)
        self.assertAlmostEqual(summary['startup']['startup_seconds'], 12.5)

    def test_summarize_preflight_reports_gate_failures(self):
        summary = run_bc_loader_preflight.summarize_preflight(
            config_path='/tmp/preflight.toml',
            config_fingerprint_value='fingerprint',
            started_at='2026-03-30T10:00:00+00:00',
            finished_at='2026-03-30T10:08:00+00:00',
            command=['torchrun', 'mortal/train_bc.py'],
            return_code=0,
            metrics_events=[
                {
                    'event': 'train_live',
                    'step': 100,
                    'runtime_seconds_total': 300.0,
                    'runtime_metrics': {'samples_per_second': 5200.0, 'steps_per_second': 0.31},
                    'loader_metrics': {'wait_fraction': 0.22, 'queued_bytes_gib': 8.0},
                },
                {
                    'event': 'train_live',
                    'step': 125,
                    'runtime_seconds_total': 380.0,
                    'runtime_metrics': {'samples_per_second': 5250.0, 'steps_per_second': 0.32},
                    'loader_metrics': {'wait_fraction': 0.20, 'queued_bytes_gib': 7.5},
                },
            ],
            gpu_samples=[
                {
                    'elapsed_seconds': 300.0,
                    'gpus': {
                        0: {'pci_bus_id': '00000000:01:00.0', 'power_draw_watts': 90.0, 'utilization_gpu': 20.0},
                        1: {'pci_bus_id': '00000000:79:00.0', 'power_draw_watts': 95.0, 'utilization_gpu': 18.0},
                    },
                },
            ],
            target_gpu_indices=[0, 1],
            target_pci_bus_ids=['1', '79'],
            preflight_cfg={
                'min_runtime_seconds': 240,
                'min_steps_before_stop': 200,
                'required_stable_windows': 2,
                'min_samples_per_second': 5900,
                'preferred_samples_per_second': 7000,
                'max_loader_wait_fraction': 0.15,
                'min_steady_gpu_watts': 150,
                'min_steady_gpu_utilization': 60,
                'min_steady_gpu_ratio': 0.70,
            },
            max_worker_rss_kib=654321,
            max_combined_worker_rss_kib=765432,
        )
        self.assertEqual(summary['status'], 'failed')
        self.assertFalse(summary['gate']['passed'])
        self.assertTrue(summary['gate']['reasons'])
        self.assertIn('completed_step_below_floor=125<200', summary['gate']['reasons'])

    def test_summarize_preflight_reports_rss_guard_abort(self):
        summary = run_bc_loader_preflight.summarize_preflight(
            config_path='/tmp/preflight.toml',
            config_fingerprint_value='fingerprint',
            started_at='2026-03-30T10:00:00+00:00',
            finished_at='2026-03-30T10:04:00+00:00',
            command=['torchrun', 'mortal/train_bc.py'],
            return_code=-15,
            metrics_events=[],
            gpu_samples=[],
            target_gpu_indices=[0, 1],
            target_pci_bus_ids=['1', '79'],
            preflight_cfg={
                'min_runtime_seconds': 240,
                'min_steps_before_stop': 200,
                'required_stable_windows': 2,
                'min_samples_per_second': 5900,
                'preferred_samples_per_second': 7000,
                'max_loader_wait_fraction': 0.15,
                'min_steady_gpu_watts': 150,
                'min_steady_gpu_utilization': 60,
                'min_steady_gpu_ratio': 0.70,
            },
            max_worker_rss_kib=120 * 1024 * 1024,
            max_combined_worker_rss_kib=200 * 1024 * 1024,
            rss_guard_abort={
                'reason': 'combined_train_worker_rss_above_guard=200.0>192.0GiB',
                'rss_sum_kib': 200 * 1024 * 1024,
                'rss_max_kib': 120 * 1024 * 1024,
            },
        )
        self.assertEqual(summary['status'], 'failed')
        self.assertFalse(summary['gate']['passed'])
        self.assertTrue(
            any(
                'combined_train_worker_rss_above_guard' in reason
                for reason in summary['gate']['reasons']
            )
        )
        self.assertEqual(summary['rss_guard_abort']['rss_max_kib'], 120 * 1024 * 1024)

    def test_completed_save_window_metrics_prefers_first_window_at_min_step(self):
        completed = run_bc_loader_preflight.completed_save_window_metrics(
            events=[
                {
                    'event': 'save_window',
                    'step': 200,
                    'runtime_seconds_total': 420.0,
                    'runtime_metrics': {'samples_per_second': 6100.0, 'steps_per_second': 0.372},
                    'loader_metrics': {'wait_fraction': 0.110, 'queued_bytes_gib': 10.0},
                },
                {
                    'event': 'save_window',
                    'step': 400,
                    'runtime_seconds_total': 810.0,
                    'runtime_metrics': {'samples_per_second': 7000.0, 'steps_per_second': 0.427},
                    'loader_metrics': {'wait_fraction': 0.090, 'queued_bytes_gib': 10.0},
                },
            ],
            min_step=200,
        )
        self.assertEqual(completed['step'], 200)
        self.assertAlmostEqual(completed['samples_per_second'], 6100.0)


if __name__ == '__main__':
    unittest.main()
