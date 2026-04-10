from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from merge_normalized_manifests import (
    build_merge_summary,
    collect_manifest_paths,
    load_manifest_rows,
    merge_loaded_rows,
)


class MergeNormalizedManifestsTest(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write('\n')

    def test_merge_rewrites_dataset_id_and_preserves_batch_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_a = tmp / 'batch_a.jsonl'
            manifest_b = tmp / 'batch_b.jsonl'

            self.write_jsonl(manifest_a, [
                {
                    'dataset_id': 'tenhou_batch_a_v0',
                    'source': 'tenhou',
                    'source_game_id': '2026010200gm-00a9-0000-bbb',
                    'raw_snapshot_id': 'snap_a',
                    'relative_path': 'data/normalized/v1/source=tenhou/year=2026/month=01/bbb.json.gz',
                    'game_date': '2026-01-02T00:00:00',
                },
            ])
            self.write_jsonl(manifest_b, [
                {
                    'dataset_id': 'tenhou_batch_b_v0',
                    'source': 'tenhou',
                    'source_game_id': '2026010100gm-00a9-0000-aaa',
                    'raw_snapshot_id': 'snap_b',
                    'relative_path': 'data/normalized/v1/source=tenhou/year=2026/month=01/aaa.json.gz',
                    'game_date': '2026-01-01T00:00:00',
                },
            ])

            loaded = load_manifest_rows([manifest_a, manifest_b])
            merged_rows, duplicate_rows, per_manifest_summary = merge_loaded_rows(
                loaded,
                dataset_id='tenhou_phoenix_4y_v0',
                on_duplicate='error',
            )

            self.assertEqual(len(duplicate_rows), 0)
            self.assertEqual(
                [row['source_game_id'] for row in merged_rows],
                [
                    '2026010100gm-00a9-0000-aaa',
                    '2026010200gm-00a9-0000-bbb',
                ],
            )
            self.assertEqual(
                {row['dataset_id'] for row in merged_rows},
                {'tenhou_phoenix_4y_v0'},
            )
            self.assertEqual(
                {row['batch_dataset_id'] for row in merged_rows},
                {'tenhou_batch_a_v0', 'tenhou_batch_b_v0'},
            )

            summary = build_merge_summary(
                dataset_id='tenhou_phoenix_4y_v0',
                output_manifest_path=tmp / 'merged.jsonl',
                manifest_paths=[manifest_a, manifest_b],
                merged_rows=merged_rows,
                duplicate_rows=duplicate_rows,
                per_manifest_summary=per_manifest_summary,
                on_duplicate='error',
            )
            self.assertEqual(summary['output_row_count'], 2)
            self.assertEqual(summary['input_row_count'], 2)
            self.assertEqual(
                summary['date_range'],
                {
                    'min': '2026-01-01T00:00:00',
                    'max': '2026-01-02T00:00:00',
                },
            )
            self.assertEqual(
                summary['batch_dataset_ids'],
                ['tenhou_batch_a_v0', 'tenhou_batch_b_v0'],
            )

    def test_merge_raises_on_duplicate_source_game_ids_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_a = tmp / 'batch_a.jsonl'
            manifest_b = tmp / 'batch_b.jsonl'
            duplicate_row = {
                'dataset_id': 'tenhou_batch_a_v0',
                'source': 'tenhou',
                'source_game_id': '2026010100gm-00a9-0000-dup',
                'raw_snapshot_id': 'snap_a',
                'relative_path': 'data/normalized/v1/source=tenhou/year=2026/month=01/dup.json.gz',
                'game_date': '2026-01-01T00:00:00',
            }
            self.write_jsonl(manifest_a, [duplicate_row])
            self.write_jsonl(manifest_b, [
                {
                    **duplicate_row,
                    'dataset_id': 'tenhou_batch_b_v0',
                    'raw_snapshot_id': 'snap_b',
                },
            ])

            loaded = load_manifest_rows([manifest_a, manifest_b])
            with self.assertRaisesRegex(ValueError, 'duplicate source games'):
                merge_loaded_rows(
                    loaded,
                    dataset_id='tenhou_phoenix_4y_v0',
                    on_duplicate='error',
                )

    def test_merge_keep_first_mode_drops_later_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_a = tmp / 'batch_a.jsonl'
            manifest_b = tmp / 'batch_b.jsonl'
            self.write_jsonl(manifest_a, [
                {
                    'dataset_id': 'tenhou_batch_a_v0',
                    'source': 'tenhou',
                    'source_game_id': '2026010100gm-00a9-0000-dup',
                    'raw_snapshot_id': 'snap_a',
                    'relative_path': 'data/normalized/v1/source=tenhou/year=2026/month=01/dup.json.gz',
                    'game_date': '2026-01-01T00:00:00',
                },
            ])
            self.write_jsonl(manifest_b, [
                {
                    'dataset_id': 'tenhou_batch_b_v0',
                    'source': 'tenhou',
                    'source_game_id': '2026010100gm-00a9-0000-dup',
                    'raw_snapshot_id': 'snap_b',
                    'relative_path': 'data/normalized/v1/source=tenhou/year=2026/month=01/dup.json.gz',
                    'game_date': '2026-01-01T00:00:00',
                },
                {
                    'dataset_id': 'tenhou_batch_b_v0',
                    'source': 'tenhou',
                    'source_game_id': '2026010200gm-00a9-0000-unique',
                    'raw_snapshot_id': 'snap_b',
                    'relative_path': 'data/normalized/v1/source=tenhou/year=2026/month=01/unique.json.gz',
                    'game_date': '2026-01-02T00:00:00',
                },
            ])

            loaded = load_manifest_rows([manifest_a, manifest_b])
            merged_rows, duplicate_rows, _ = merge_loaded_rows(
                loaded,
                dataset_id='tenhou_phoenix_4y_v0',
                on_duplicate='keep-first',
            )

            self.assertEqual(len(merged_rows), 2)
            self.assertEqual(len(duplicate_rows), 1)
            self.assertEqual(
                [row['source_game_id'] for row in merged_rows],
                [
                    '2026010100gm-00a9-0000-dup',
                    '2026010200gm-00a9-0000-unique',
                ],
            )
            self.assertEqual(merged_rows[0]['batch_dataset_id'], 'tenhou_batch_a_v0')

    def test_collect_manifest_paths_supports_globs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_a = tmp / 'a.jsonl'
            manifest_b = tmp / 'b.jsonl'
            manifest_a.write_text('', encoding='utf-8')
            manifest_b.write_text('', encoding='utf-8')

            manifest_paths = collect_manifest_paths([], [str(tmp / '*.jsonl')])
            self.assertEqual(manifest_paths, [
                manifest_a.resolve(),
                manifest_b.resolve(),
            ])


if __name__ == '__main__':
    unittest.main()
