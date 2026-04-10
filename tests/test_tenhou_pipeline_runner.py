from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_tenhou_pipeline import build_steps, default_archive_dir, default_dataset_id, selected_steps


def make_args(**overrides):
    base = {
        'snapshot_id': '2026-03-28_phoenix_smoke_7d',
        'dataset_id': '',
        'date': [],
        'start_date': '2026-01-01',
        'end_date': '2026-01-07',
        'archive_dir': '',
        'ruleset': [],
        'max_per_archive': 4,
        'limit': 28,
        'archive_jobs': 6,
        'stage_jobs': 10,
        'download_timeout': 45.0,
        'download_retries': 3,
        'retry_backoff_seconds': 1.5,
        'archive_publish_lag_days': 1,
        'year_archive_cache_dir': '/tmp/tenhou_scraw_cache',
        'usage_status': 'corpus-expansion-smoke-7d',
        'converter_version': 'tenhou-xml-v0',
        'converter_source': 'xml',
        'with_release_artifacts': False,
        'stop_after': 'ingest',
        'dry_run': True,
        'overwrite': True,
    }
    base.update(overrides)
    return Namespace(**base)


class TenhouPipelineRunnerTest(unittest.TestCase):
    def test_defaults_derive_expected_names(self):
        snapshot_id = '2026-03-28_phoenix_smoke_7d'
        self.assertEqual(default_dataset_id(snapshot_id), '2026-03-28_phoenix_smoke_7d_v0')
        self.assertEqual(default_archive_dir(snapshot_id), '/tmp/tenhou_2026-03-28_phoenix_smoke_7d')

    def test_build_steps_without_release_has_four_core_steps(self):
        steps = build_steps(make_args())
        self.assertEqual([step.name for step in steps], [
            'fetch',
            'select',
            'stage',
            'ingest',
        ])
        fetch_jobs_index = steps[0].cmd.index('--jobs')
        self.assertEqual(steps[0].cmd[fetch_jobs_index + 1], '6')
        publish_lag_index = steps[0].cmd.index('--publish-lag-days')
        self.assertEqual(steps[0].cmd[publish_lag_index + 1], '1')
        year_cache_index = steps[0].cmd.index('--year-archive-cache-dir')
        self.assertEqual(steps[0].cmd[year_cache_index + 1], '/tmp/tenhou_scraw_cache')
        stage_jobs_index = steps[2].cmd.index('--jobs')
        self.assertEqual(steps[2].cmd[stage_jobs_index + 1], '10')

    def test_build_steps_with_release_appends_release_steps(self):
        steps = build_steps(make_args(with_release_artifacts=True))
        self.assertEqual([step.name for step in steps], [
            'fetch',
            'select',
            'stage',
            'ingest',
            'qa',
            'split-all',
            'split-hanchan',
            'overlap',
        ])

    def test_selected_steps_release_keeps_all_release_substeps(self):
        steps = build_steps(make_args(with_release_artifacts=True))
        planned_steps = selected_steps(steps, 'release')
        self.assertEqual([step.name for step in planned_steps], [
            'fetch',
            'select',
            'stage',
            'ingest',
            'qa',
            'split-all',
            'split-hanchan',
            'overlap',
        ])


if __name__ == '__main__':
    unittest.main()
