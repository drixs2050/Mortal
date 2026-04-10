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

from summarize_split_overlap import build_raw_player_index, summarize_split_overlap


class DatasetReleaseHelperBehaviorTest(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write('\n')

    def test_overlap_summary_reports_pairwise_and_three_way_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / 'manifest.jsonl'
            raw_manifest_path = tmp / 'raw.json'
            split_dir = tmp / 'splits'
            split_dir.mkdir()

            rows = [
                {
                    'relative_path': 'data/normalized/game_a.json.gz',
                    'source': 'tenhou',
                    'source_game_id': 'game_a',
                    'raw_snapshot_id': 'snap_a',
                },
                {
                    'relative_path': 'data/normalized/game_b.json.gz',
                    'source': 'tenhou',
                    'source_game_id': 'game_b',
                    'raw_snapshot_id': 'snap_a',
                },
                {
                    'relative_path': 'data/normalized/game_c.json.gz',
                    'source': 'tenhou',
                    'source_game_id': 'game_c',
                    'raw_snapshot_id': 'snap_a',
                },
                {
                    'relative_path': 'data/normalized/game_d.json.gz',
                    'source': 'tenhou',
                    'source_game_id': 'game_d',
                    'raw_snapshot_id': 'snap_a',
                },
            ]
            self.write_jsonl(manifest_path, rows)
            raw_manifest_path.write_text(
                json.dumps(
                    {
                        'source': 'tenhou',
                        'snapshot_id': 'snap_a',
                        'files': [
                            {
                                'source_game_id': 'game_a',
                                'player_names': ['alice', 'bob', 'carol', 'dave'],
                            },
                            {
                                'source_game_id': 'game_b',
                                'player_names': ['erin', 'frank', 'george', 'henry'],
                            },
                            {
                                'source_game_id': 'game_c',
                                'player_names': ['alice', 'ivy', 'jack', 'kate'],
                            },
                            {
                                'source_game_id': 'game_d',
                                'player_names': ['alice', 'bob', 'luke', 'mary'],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            (split_dir / 'train.txt').write_text(
                'data/normalized/game_a.json.gz\n'
                'data/normalized/game_b.json.gz\n',
                encoding='utf-8',
            )
            (split_dir / 'val.txt').write_text(
                'data/normalized/game_c.json.gz\n',
                encoding='utf-8',
            )
            (split_dir / 'test.txt').write_text(
                'data/normalized/game_d.json.gz\n',
                encoding='utf-8',
            )

            raw_player_index = build_raw_player_index([raw_manifest_path])
            summary = summarize_split_overlap(
                rows=rows,
                split_map={
                    'train': (split_dir / 'train.txt').read_text(encoding='utf-8').splitlines(),
                    'val': (split_dir / 'val.txt').read_text(encoding='utf-8').splitlines(),
                    'test': (split_dir / 'test.txt').read_text(encoding='utf-8').splitlines(),
                },
                raw_player_index=raw_player_index,
            )

            self.assertEqual(summary['splits']['train']['game_count'], 2)
            self.assertEqual(summary['splits']['train']['unique_player_hash_count'], 8)
            self.assertEqual(summary['splits']['val']['unique_player_hash_count'], 4)
            self.assertEqual(summary['splits']['test']['unique_player_hash_count'], 4)

            self.assertEqual(
                summary['pairwise_overlap']['train_val']['shared_unique_player_hash_count'],
                1,
            )
            self.assertEqual(
                summary['pairwise_overlap']['train_test']['shared_unique_player_hash_count'],
                2,
            )
            self.assertEqual(
                summary['pairwise_overlap']['val_test']['shared_unique_player_hash_count'],
                1,
            )
            self.assertEqual(
                summary['all_split_overlap']['shared_unique_player_hash_count'],
                1,
            )
            self.assertEqual(
                summary['pairwise_overlap']['train_test']['shared_source_game_count'],
                0,
            )

    def test_overlap_summary_reports_missing_raw_entries(self):
        rows = [
            {
                'relative_path': 'data/normalized/game_a.json.gz',
                'source': 'tenhou',
                'source_game_id': 'game_a',
                'raw_snapshot_id': 'snap_a',
            },
            {
                'relative_path': 'data/normalized/game_b.json.gz',
                'source': 'tenhou',
                'source_game_id': 'game_b',
                'raw_snapshot_id': 'snap_a',
            },
        ]
        summary = summarize_split_overlap(
            rows=rows,
            split_map={
                'train': ['data/normalized/game_a.json.gz'],
                'val': ['data/normalized/game_b.json.gz'],
                'test': [],
            },
            raw_player_index={
                ('tenhou', 'game_a'): ['alice', 'bob', 'carol', 'dave'],
            },
        )

        self.assertEqual(summary['splits']['train']['unresolved_source_game_id_count'], 0)
        self.assertEqual(summary['splits']['val']['unresolved_source_game_id_count'], 1)
        self.assertEqual(summary['splits']['val']['unique_player_hash_count'], 0)


if __name__ == '__main__':
    unittest.main()
