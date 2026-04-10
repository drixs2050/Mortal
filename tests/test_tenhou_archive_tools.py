from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
import gzip
from argparse import Namespace
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fetch_tenhou_scc_archives import (
    build_archive_filename,
    build_archive_url,
    build_year_archive_filename,
    build_year_archive_url,
    extract_archive_from_year_zip,
    is_unpublished_archive_error,
    legacy_member_name,
    iter_dates,
    latest_expected_archive_date,
)
from select_tenhou_scc_refs import collect_archive_paths
from urllib.error import HTTPError


class TenhouArchiveToolsTest(unittest.TestCase):
    def test_fetch_helpers_build_expected_daily_archive_paths(self):
        day = date(2026, 1, 3)
        self.assertEqual(build_archive_filename(day), 'scc20260103.html.gz')
        self.assertEqual(
            build_archive_url(day),
            'https://tenhou.net/sc/raw/dat/2026/scc20260103.html.gz',
        )
        self.assertEqual(build_year_archive_filename(2025), 'scraw2025.zip')
        self.assertEqual(
            build_year_archive_url(2025),
            'https://tenhou.net/sc/raw/scraw2025.zip',
        )
        self.assertEqual(
            legacy_member_name(date(2025, 3, 20)),
            '2025/scc20250320.html.gz',
        )

    def test_iter_dates_is_inclusive(self):
        days = list(iter_dates(date(2026, 1, 1), date(2026, 1, 3)))
        self.assertEqual(days, [
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 3),
        ])

    def test_latest_expected_archive_date_respects_publish_lag(self):
        self.assertEqual(
            latest_expected_archive_date(date(2026, 3, 28), lag_days=1),
            date(2026, 3, 27),
        )

    def test_same_day_404_is_treated_as_unpublished_with_default_lag(self):
        exc = HTTPError(
            url='https://tenhou.net/sc/raw/dat/2026/scc20260328.html.gz',
            code=404,
            msg='Not Found',
            hdrs=None,
            fp=None,
        )
        self.assertTrue(
            is_unpublished_archive_error(
                date(2026, 3, 28),
                exc,
                reference_date=date(2026, 3, 28),
                lag_days=1,
            )
        )

    def test_recent_trailing_404_inside_larger_lag_window_is_treated_as_unpublished(self):
        exc = HTTPError(
            url='https://tenhou.net/sc/raw/dat/2026/scc20260320.html.gz',
            code=404,
            msg='Not Found',
            hdrs=None,
            fp=None,
        )
        self.assertTrue(
            is_unpublished_archive_error(
                date(2026, 3, 20),
                exc,
                reference_date=date(2026, 3, 28),
                lag_days=14,
            )
        )

    def test_historical_404_is_still_a_real_failure(self):
        exc = HTTPError(
            url='https://tenhou.net/sc/raw/dat/2026/scc20260327.html.gz',
            code=404,
            msg='Not Found',
            hdrs=None,
            fp=None,
        )
        self.assertFalse(
            is_unpublished_archive_error(
                date(2026, 3, 27),
                exc,
                reference_date=date(2026, 3, 28),
                lag_days=1,
            )
        )

    def test_collect_archive_paths_accepts_archive_list_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_a = tmp / 'scc20260101.html.gz'
            archive_b = tmp / 'nested' / 'scc20260102.html.gz'
            archive_b.parent.mkdir()
            archive_a.write_bytes(b'')
            archive_b.write_bytes(b'')

            archive_list = tmp / 'archives.txt'
            archive_list.write_text(
                '# comment\n'
                f'{archive_a.name}\n'
                f'nested/{archive_b.name}\n',
                encoding='utf-8',
            )

            archives = collect_archive_paths([], [str(archive_list)])
            self.assertEqual(archives, [
                archive_a.resolve(),
                archive_b.resolve(),
            ])

    def test_collect_archive_paths_raises_clear_error_for_empty_archive_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_list = tmp / 'archives.txt'
            archive_list.write_text('', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'contained no archive paths'):
                collect_archive_paths([], [str(archive_list)])

    def test_extract_archive_from_year_zip_reads_daily_member(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            zip_path = tmp / 'scraw2025.zip'
            output_dir = tmp / 'out'
            member_name = '2025/scc20250320.html.gz'
            payload = gzip.compress(
                (
                    '00:00 | 36 | 四鳳南喰赤－ | '
                    '<a href="http://tenhou.net/0/?log=2025032000gm-00a9-0000-test">牌譜</a> | '
                    'a b c d<br>\n'
                ).encode('utf-8')
            )

            with zipfile.ZipFile(zip_path, 'w') as archive_zip:
                archive_zip.writestr(member_name, payload)

            with zipfile.ZipFile(zip_path) as archive_zip:
                row, present_path, status_key = extract_archive_from_year_zip(
                    date(2025, 3, 20),
                    archive_zip=archive_zip,
                    output_dir=output_dir,
                    year_archive_path=zip_path,
                    year_archive_url='https://tenhou.net/sc/raw/scraw2025.zip',
                    year_archive_meta={
                        'year_archive_filename': 'scraw2025.zip',
                        'year_archive_status': 'cached',
                        'year_archive_byte_size': zip_path.stat().st_size,
                    },
                    args=Namespace(overwrite=False),
                )

            self.assertEqual(status_key, 'downloaded')
            self.assertIsNotNone(present_path)
            self.assertEqual(Path(present_path).read_bytes(), payload)
            self.assertEqual(row['year_archive_member'], member_name)
            self.assertEqual(row['status'], 'downloaded')


if __name__ == '__main__':
    unittest.main()
