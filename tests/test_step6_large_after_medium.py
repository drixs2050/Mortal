import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'scripts' / 'run_step6_large_after_medium.py'
SPEC = importlib.util.spec_from_file_location('run_step6_large_after_medium', MODULE_PATH)
run_step6_large_after_medium = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_step6_large_after_medium)


class Step6LargeAfterMediumHelpersTest(unittest.TestCase):
    def test_process_list_contains_config_matches_relative_and_absolute_lines(self):
        ps_output = '\n'.join([
            '/home/user/python scripts/run_bc_campaign.py --config configs/step6_bc_medium_full8dan.toml',
            '/home/user/python scripts/run_bc_campaign.py --config /tmp/other.toml',
        ])
        self.assertTrue(
            run_step6_large_after_medium.process_list_contains_config(
                ps_output,
                'step6_bc_medium_full8dan.toml',
            )
        )
        self.assertFalse(
            run_step6_large_after_medium.process_list_contains_config(
                ps_output,
                'step6_bc_large_bounded_full8dan.toml',
            )
        )

    def test_log_indicates_oom_detects_common_cuda_oom_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'oom.log'
            log_path.write_text('RuntimeError: CUDA out of memory while allocating tensor\n', encoding='utf-8')
            self.assertTrue(run_step6_large_after_medium.log_indicates_oom(log_path))

    def test_log_indicates_oom_ignores_non_oom_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / 'other.log'
            log_path.write_text('RuntimeError: some unrelated launch failure\n', encoding='utf-8')
            self.assertFalse(run_step6_large_after_medium.log_indicates_oom(log_path))


if __name__ == '__main__':
    unittest.main()
