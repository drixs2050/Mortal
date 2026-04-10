import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MORTAL_DIR = ROOT / 'mortal'

if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

os.environ.setdefault('MORTAL_CFG', str(MORTAL_DIR / 'config.example.toml'))

from raw_store import (  # noqa: E402
    PackedRawSource,
    build_raw_pack,
    load_raw_pack_index,
    normalize_raw_source_key,
    verify_raw_pack,
)


class RawStoreTest(unittest.TestCase):
    def test_build_raw_pack_round_trips_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / 'a.log.gz'
            file_b = root / 'b.log.gz'
            payload_a = b'raw-gz-a'
            payload_b = b'raw-gz-b'
            file_a.write_bytes(payload_a)
            file_b.write_bytes(payload_b)

            pack_path = root / 'dataset.raw.pack'
            index_path = root / 'dataset.raw.index.json'
            summary = build_raw_pack(
                [file_a, file_b],
                pack_path=pack_path,
                index_path=index_path,
            )

            self.assertEqual(summary['entry_count'], 2)
            index = load_raw_pack_index(index_path)
            self.assertEqual(index.entry_count, 2)
            self.assertIn(normalize_raw_source_key(file_a), index.entries)
            self.assertIn(normalize_raw_source_key(file_b), index.entries)

            with PackedRawSource(pack_path, index_path) as source:
                self.assertEqual(source.read(file_a), payload_a)
                self.assertEqual(source.read(file_b), payload_b)

    def test_verify_raw_pack_reports_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / 'single.log.gz'
            source_file.write_bytes(b'hello-world')
            pack_path = root / 'dataset.raw.pack'
            index_path = root / 'dataset.raw.index.json'
            build_raw_pack([source_file], pack_path=pack_path, index_path=index_path)

            with pack_path.open('r+b') as f:
                f.seek(0)
                f.write(b'X')

            summary = verify_raw_pack(
                pack_path=pack_path,
                index_path=index_path,
                file_list=[source_file],
            )
            self.assertFalse(summary['ok'])
            self.assertEqual(len(summary['mismatched_entries']), 1)

    def test_packed_raw_source_raises_for_missing_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / 'single.log.gz'
            source_file.write_bytes(b'payload')
            pack_path = root / 'dataset.raw.pack'
            index_path = root / 'dataset.raw.index.json'
            build_raw_pack([source_file], pack_path=pack_path, index_path=index_path)

            with PackedRawSource(pack_path, index_path) as source:
                with self.assertRaises(KeyError):
                    source.read(root / 'missing.log.gz')


if __name__ == '__main__':
    unittest.main()
