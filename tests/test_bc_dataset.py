import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from bc_dataset import (  # noqa: E402
    allowed_player_ids_for_row,
    build_actor_filter_map,
    load_path_cache,
    load_actor_filter_index,
    normalize_file_list,
    normalize_file_path,
    resolve_actor_filter_map,
    save_path_cache,
    save_actor_filter_index,
)


class BcDatasetHelpersTest(unittest.TestCase):
    def test_allowed_player_ids_for_row_respects_threshold(self):
        row = {'player_dan': [16, 17, 18, 15]}
        self.assertEqual(allowed_player_ids_for_row(row, 17), (1, 2))
        self.assertEqual(allowed_player_ids_for_row(row, 18), (2,))

    def test_normalize_file_path_handles_relative_and_absolute(self):
        rel = 'data/example.json.gz'
        abs_path = normalize_file_path(rel)
        self.assertTrue(abs_path.endswith('/data/example.json.gz'))
        self.assertEqual(normalize_file_path(abs_path), abs_path)
        self.assertEqual(normalize_file_list([rel]), [abs_path])

    def test_build_actor_filter_map_matches_requested_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / 'manifest.jsonl'
            file_a = tmp / 'a.json.gz'
            file_b = tmp / 'b.json.gz'
            file_a.write_text('', encoding='utf-8')
            file_b.write_text('', encoding='utf-8')
            manifest.write_text(
                '\n'.join([
                    '{"relative_path":"%s","player_dan":[16,17,18,16]}' % file_a.as_posix(),
                    '{"relative_path":"%s","player_dan":[16,16,16,16]}' % file_b.as_posix(),
                ]) + '\n',
                encoding='utf-8',
            )

            actor_filter_map, summary = build_actor_filter_map(
                manifest_path=str(manifest),
                file_lists=[[str(file_a), str(file_b)]],
                min_actor_dan=17,
            )
            self.assertEqual(actor_filter_map[normalize_file_path(str(file_a))], (1, 2))
            self.assertEqual(actor_filter_map[normalize_file_path(str(file_b))], ())
            self.assertEqual(summary['matched_row_count'], 2)
            self.assertEqual(summary['eligible_file_count'], 1)
            self.assertEqual(summary['filtered_out_file_count'], 1)

    def test_build_actor_filter_map_raises_on_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / 'manifest.jsonl'
            file_a = tmp / 'a.json.gz'
            file_a.write_text('', encoding='utf-8')
            manifest.write_text('', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'missing file metadata'):
                build_actor_filter_map(
                    manifest_path=str(manifest),
                    file_lists=[[str(file_a)]],
                    min_actor_dan=17,
                )

    def test_save_and_load_path_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_file = tmp / 'train.json.gz'
            val_file = tmp / 'val.json.gz'
            cache_path = tmp / 'path_cache.pth'
            train_file.write_text('', encoding='utf-8')
            val_file.write_text('', encoding='utf-8')
            split_lists = {
                'train': [normalize_file_path(str(train_file))],
                'val': [normalize_file_path(str(val_file))],
            }
            save_path_cache(
                str(cache_path),
                split_lists=split_lists,
                source_files={'train': 'train.txt', 'val': 'val.txt'},
            )
            loaded = load_path_cache(
                str(cache_path),
                expected_splits=['train', 'val'],
                expected_sources={'train': 'train.txt', 'val': 'val.txt'},
            )
            self.assertEqual(loaded, split_lists)

    def test_save_and_load_actor_filter_index_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / 'manifest.jsonl'
            file_a = tmp / 'a.json.gz'
            file_b = tmp / 'b.json.gz'
            index_path = tmp / 'actor_filter.pth'
            file_a.write_text('', encoding='utf-8')
            file_b.write_text('', encoding='utf-8')
            manifest.write_text(
                '\n'.join([
                    '{"relative_path":"%s","player_dan":[16,17,18,16]}' % file_a.as_posix(),
                    '{"relative_path":"%s","player_dan":[16,16,16,16]}' % file_b.as_posix(),
                ]) + '\n',
                encoding='utf-8',
            )
            actor_filter_map, summary = build_actor_filter_map(
                manifest_path=str(manifest),
                file_lists=[[str(file_a), str(file_b)]],
                min_actor_dan=17,
            )
            save_actor_filter_index(
                str(index_path),
                actor_filter_map=actor_filter_map,
                summary=summary,
            )
            loaded_map, loaded_summary = load_actor_filter_index(
                index_path=str(index_path),
                file_lists=[[str(file_a), str(file_b)]],
                min_actor_dan=17,
            )
            self.assertEqual(loaded_map, actor_filter_map)
            self.assertEqual(loaded_summary['source'], 'index')
            self.assertEqual(loaded_summary['matched_row_count'], 2)
            self.assertEqual(loaded_summary['eligible_file_count'], 1)

    def test_resolve_actor_filter_map_prefers_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / 'manifest.jsonl'
            file_a = tmp / 'a.json.gz'
            index_path = tmp / 'actor_filter.pth'
            file_a.write_text('', encoding='utf-8')
            manifest.write_text(
                '{"relative_path":"%s","player_dan":[17,16,16,16]}\n' % file_a.as_posix(),
                encoding='utf-8',
            )
            actor_filter_map, summary = build_actor_filter_map(
                manifest_path=str(manifest),
                file_lists=[[str(file_a)]],
                min_actor_dan=17,
            )
            save_actor_filter_index(
                str(index_path),
                actor_filter_map=actor_filter_map,
                summary=summary,
            )
            resolved_map, resolved_summary = resolve_actor_filter_map(
                file_lists=[[str(file_a)]],
                min_actor_dan=17,
                actor_filter_manifest='',
                actor_filter_index=str(index_path),
            )
            self.assertEqual(resolved_map[normalize_file_path(str(file_a))], (0,))
            self.assertEqual(resolved_summary['source'], 'index')


if __name__ == '__main__':
    unittest.main()
