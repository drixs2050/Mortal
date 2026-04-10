#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from progress_report import ProgressReporter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = 'https://tenhou.net/sc/raw/dat'
DEFAULT_YEAR_ARCHIVE_BASE_URL = 'https://tenhou.net/sc/raw'
DEFAULT_YEAR_ARCHIVE_CACHE_DIR = '/tmp/tenhou_scraw_cache'
GZIP_MAGIC = b'\x1f\x8b'
ZIP_MAGIC = b'PK'
USER_AGENT = 'Mozilla/5.0'
DEFAULT_FETCH_JOBS = 8
DEFAULT_PUBLISH_LAG_DAYS = 14
DEFAULT_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
LEGACY_YEAR_ARCHIVE_LAST_YEAR = 2025


def parse_iso_date(raw: str) -> date:
    return date.fromisoformat(raw)


def build_archive_filename(day: date) -> str:
    return f'scc{day:%Y%m%d}.html.gz'


def build_archive_url(day: date, base_url: str = DEFAULT_BASE_URL) -> str:
    return f"{base_url}/{day:%Y}/{build_archive_filename(day)}"


def build_year_archive_filename(year: int) -> str:
    return f'scraw{year}.zip'


def build_year_archive_url(year: int, base_url: str = DEFAULT_YEAR_ARCHIVE_BASE_URL) -> str:
    return f"{base_url.rstrip('/')}/{build_year_archive_filename(year)}"


def iter_dates(start_date: date, end_date: date):
    if end_date < start_date:
        raise ValueError('end date must be on or after start date')

    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def unique_dates(days: list[date]) -> list[date]:
    return sorted(set(days))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch Tenhou scc*.html.gz raw-archive files for a date range.',
    )
    parser.add_argument(
        '--date',
        action='append',
        default=[],
        help='Specific archive date in YYYY-MM-DD format. Repeat as needed.',
    )
    parser.add_argument(
        '--start-date',
        default='',
        help='Inclusive start date in YYYY-MM-DD format.',
    )
    parser.add_argument(
        '--end-date',
        default='',
        help='Inclusive end date in YYYY-MM-DD format.',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory where downloaded scc*.html.gz files will be written.',
    )
    parser.add_argument(
        '--base-url',
        default=DEFAULT_BASE_URL,
        help='Base URL for Tenhou archive fetches.',
    )
    parser.add_argument(
        '--year-archive-base-url',
        default=DEFAULT_YEAR_ARCHIVE_BASE_URL,
        help='Base URL for old yearly Tenhou archive zip downloads.',
    )
    parser.add_argument(
        '--year-archive-cache-dir',
        default=DEFAULT_YEAR_ARCHIVE_CACHE_DIR,
        help='Directory used to cache old yearly Tenhou archive zip downloads.',
    )
    parser.add_argument(
        '--summary',
        default='',
        help='Optional JSON summary path, relative to the repo root or absolute.',
    )
    parser.add_argument(
        '--output-list',
        default='',
        help='Optional text file that will contain one downloaded archive path per line.',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Network timeout in seconds.',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=DEFAULT_FETCH_JOBS,
        help='Number of concurrent archive downloads.',
    )
    parser.add_argument(
        '--publish-lag-days',
        type=int,
        default=DEFAULT_PUBLISH_LAG_DAYS,
        help='Treat recent archive 404s inside this lag window as not-yet-published instead of failed.',
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=DEFAULT_RETRIES,
        help='Number of retry attempts for transient archive download failures.',
    )
    parser.add_argument(
        '--retry-backoff-seconds',
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help='Base exponential backoff in seconds between archive download retries.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Re-download files even if they already exist.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned fetch set without downloading files.',
    )
    return parser.parse_args()


def resolve_output_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_date_plan(args) -> list[date]:
    explicit_dates = [parse_iso_date(raw) for raw in args.date]
    has_range = bool(args.start_date or args.end_date)
    if explicit_dates and has_range:
        raise ValueError('use either repeated --date values or a --start-date/--end-date range')

    if explicit_dates:
        return unique_dates(explicit_dates)

    if bool(args.start_date) != bool(args.end_date):
        raise ValueError('--start-date and --end-date must be set together')

    if args.start_date:
        start_date = parse_iso_date(args.start_date)
        end_date = parse_iso_date(args.end_date)
        return list(iter_dates(start_date, end_date))

    raise ValueError('provide at least one --date or a --start-date/--end-date range')


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_payload(
    url: str,
    *,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> bytes:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={'User-Agent': USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, HTTPError) and exc.code == 404:
                raise
            if attempt >= retries:
                raise
            time.sleep(retry_backoff_seconds * (2 ** attempt))
    raise RuntimeError('payload download retries exhausted') from last_exc


def latest_expected_archive_date(
    reference_date: date | None = None,
    *,
    lag_days: int = DEFAULT_PUBLISH_LAG_DAYS,
) -> date:
    today = reference_date or date.today()
    return today - timedelta(days=lag_days)


def is_unpublished_archive_error(
    day: date,
    exc: Exception,
    *,
    reference_date: date | None = None,
    lag_days: int = DEFAULT_PUBLISH_LAG_DAYS,
) -> bool:
    return (
        isinstance(exc, HTTPError)
        and exc.code == 404
        and day > latest_expected_archive_date(reference_date, lag_days=lag_days)
    )


def fetch_archive(
    url: str,
    output_path: Path,
    *,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> tuple[int, str]:
    payload = download_payload(
        url,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    if not payload.startswith(GZIP_MAGIC):
        preview = payload[:120].decode('utf-8', errors='replace')
        raise ValueError(f'archive fetch did not return a gzip payload: {preview!r}')

    ensure_parent(output_path)
    output_path.write_bytes(payload)
    return len(payload), compute_sha256_bytes(payload)


def fetch_year_archive_zip(
    url: str,
    output_path: Path,
    *,
    timeout: float,
    retries: int,
    retry_backoff_seconds: float,
) -> tuple[int, str]:
    payload = download_payload(
        url,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

    if not payload.startswith(ZIP_MAGIC):
        preview = payload[:120].decode('utf-8', errors='replace')
        raise ValueError(f'year archive fetch did not return a zip payload: {preview!r}')

    ensure_parent(output_path)
    output_path.write_bytes(payload)
    return len(payload), compute_sha256_bytes(payload)


def write_lines(path: Path, lines: list[str]) -> None:
    ensure_parent(path)
    path.write_text(''.join(f'{line}\n' for line in lines), encoding='utf-8')


def format_status(*, downloaded: int, skipped_existing: int, skipped_unpublished: int, failed: int) -> str:
    return (
        f'downloaded={downloaded} '
        f'existing={skipped_existing} '
        f'unpublished={skipped_unpublished} '
        f'failed={failed}'
    )


def fetch_one_archive(day: date, *, output_dir: Path, base_url: str, args) -> tuple[dict, str | None, str]:
    filename = build_archive_filename(day)
    url = build_archive_url(day, base_url.rstrip('/'))
    output_path = output_dir / filename
    row = {
        'date': day.isoformat(),
        'filename': filename,
        'url': url,
        'output_path': str(output_path),
    }

    if output_path.exists() and not args.overwrite:
        row['status'] = 'skipped_existing'
        row['byte_size'] = output_path.stat().st_size
        return row, str(output_path), 'skipped_existing'

    try:
        byte_size, sha256 = fetch_archive(
            url,
            output_path,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
    except Exception as exc:
        row['error'] = str(exc)
        if is_unpublished_archive_error(
            day,
            exc,
            lag_days=args.publish_lag_days,
        ):
            row['status'] = 'skipped_unpublished'
            return row, None, 'skipped_unpublished'
        row['status'] = 'failed'
        return row, None, 'failed'

    row['status'] = 'downloaded'
    row['byte_size'] = byte_size
    row['sha256'] = sha256
    return row, str(output_path), 'downloaded'


def legacy_member_name(day: date) -> str:
    return f'{day:%Y}/{build_archive_filename(day)}'


def ensure_year_archive_zip(year: int, *, cache_dir: Path, args) -> tuple[Path, str, dict]:
    zip_filename = build_year_archive_filename(year)
    zip_url = build_year_archive_url(year, args.year_archive_base_url.rstrip('/'))
    zip_path = cache_dir / zip_filename
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_row = {
        'year_archive_filename': zip_filename,
        'year_archive_url': zip_url,
        'year_archive_path': str(zip_path),
    }

    if zip_path.exists():
        cache_row['year_archive_status'] = 'cached'
        cache_row['year_archive_byte_size'] = zip_path.stat().st_size
        return zip_path, zip_url, cache_row

    byte_size, sha256 = fetch_year_archive_zip(
        zip_url,
        zip_path,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    cache_row['year_archive_status'] = 'downloaded'
    cache_row['year_archive_byte_size'] = byte_size
    cache_row['year_archive_sha256'] = sha256
    return zip_path, zip_url, cache_row


def extract_archive_from_year_zip(
    day: date,
    *,
    archive_zip: zipfile.ZipFile,
    output_dir: Path,
    year_archive_path: Path,
    year_archive_url: str,
    year_archive_meta: dict,
    args,
) -> tuple[dict, str | None, str]:
    filename = build_archive_filename(day)
    output_path = output_dir / filename
    member_name = legacy_member_name(day)
    row = {
        'date': day.isoformat(),
        'filename': filename,
        'url': year_archive_url,
        'output_path': str(output_path),
        'year_archive_member': member_name,
        **year_archive_meta,
    }

    if output_path.exists() and not args.overwrite:
        row['status'] = 'skipped_existing'
        row['byte_size'] = output_path.stat().st_size
        return row, str(output_path), 'skipped_existing'

    try:
        payload = archive_zip.read(member_name)
    except KeyError as exc:
        row['error'] = f'missing archive member: {member_name}'
        row['status'] = 'failed'
        return row, None, 'failed'

    if not payload.startswith(GZIP_MAGIC):
        preview = payload[:120].decode('utf-8', errors='replace')
        row['error'] = f'archive member did not look like gzip payload: {preview!r}'
        row['status'] = 'failed'
        return row, None, 'failed'

    ensure_parent(output_path)
    output_path.write_bytes(payload)
    row['status'] = 'downloaded'
    row['byte_size'] = len(payload)
    row['sha256'] = compute_sha256_bytes(payload)
    row['year_archive_path'] = str(year_archive_path)
    return row, str(output_path), 'downloaded'


def main():
    args = parse_args()
    output_dir = resolve_output_path(args.output_dir)
    summary_path = resolve_output_path(args.summary) if args.summary else None
    output_list_path = resolve_output_path(args.output_list) if args.output_list else None
    year_archive_cache_dir = resolve_output_path(args.year_archive_cache_dir)

    planned_dates = build_date_plan(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_rows: list[dict | None] = [None] * len(planned_dates)
    ordered_paths: list[str | None] = [None] * len(planned_dates)
    day_to_index = {day: idx for idx, day in enumerate(planned_dates)}
    downloaded_count = 0
    skipped_existing_count = 0
    skipped_unpublished_count = 0
    failed_count = 0
    progress = ProgressReporter(
        total=len(planned_dates),
        desc='FETCH',
        unit='archive',
    )

    def record_result(day: date, row: dict, present_path: str | None, status_key: str) -> None:
        nonlocal downloaded_count, skipped_existing_count, skipped_unpublished_count, failed_count
        ordered_rows[day_to_index[day]] = row
        ordered_paths[day_to_index[day]] = present_path
        if status_key == 'planned':
            progress.update(status='planned')
            return
        if status_key == 'downloaded':
            downloaded_count += 1
        elif status_key == 'skipped_existing':
            skipped_existing_count += 1
        elif status_key == 'skipped_unpublished':
            skipped_unpublished_count += 1
        else:
            failed_count += 1
        progress.update(
            status=format_status(
                downloaded=downloaded_count,
                skipped_existing=skipped_existing_count,
                skipped_unpublished=skipped_unpublished_count,
                failed=failed_count,
            ),
        )

    for day in planned_dates:
        if args.dry_run:
            filename = build_archive_filename(day)
            if day.year <= LEGACY_YEAR_ARCHIVE_LAST_YEAR:
                url = build_year_archive_url(day.year, args.year_archive_base_url.rstrip('/'))
            else:
                url = build_archive_url(day, args.base_url.rstrip('/'))
            output_path = output_dir / filename
            row = {
                'date': day.isoformat(),
                'filename': filename,
                'url': url,
                'output_path': str(output_path),
            }
            if day.year <= LEGACY_YEAR_ARCHIVE_LAST_YEAR:
                row['year_archive_member'] = legacy_member_name(day)
                row['year_archive_filename'] = build_year_archive_filename(day.year)
            row['status'] = 'planned'
            record_result(day, row, None, 'planned')

    if not args.dry_run:
        legacy_days = [day for day in planned_dates if day.year <= LEGACY_YEAR_ARCHIVE_LAST_YEAR]
        direct_days = [day for day in planned_dates if day.year > LEGACY_YEAR_ARCHIVE_LAST_YEAR]

        if args.jobs <= 1:
            for day in direct_days:
                row, present_path, status_key = fetch_one_archive(
                    day,
                    output_dir=output_dir,
                    base_url=args.base_url,
                    args=args,
                )
                record_result(day, row, present_path, status_key)
        elif direct_days:
            with ThreadPoolExecutor(max_workers=min(args.jobs, len(direct_days))) as executor:
                future_to_day = {
                    executor.submit(
                        fetch_one_archive,
                        day,
                        output_dir=output_dir,
                        base_url=args.base_url,
                        args=args,
                    ): day
                    for day in direct_days
                }
                for future in as_completed(future_to_day):
                    day = future_to_day[future]
                    row, present_path, status_key = future.result()
                    record_result(day, row, present_path, status_key)

        for year in sorted({day.year for day in legacy_days}):
            year_days = [day for day in legacy_days if day.year == year]
            try:
                year_zip_path, year_zip_url, year_archive_meta = ensure_year_archive_zip(
                    year,
                    cache_dir=year_archive_cache_dir,
                    args=args,
                )
            except Exception as exc:
                for day in year_days:
                    filename = build_archive_filename(day)
                    output_path = output_dir / filename
                    row = {
                        'date': day.isoformat(),
                        'filename': filename,
                        'url': build_year_archive_url(year, args.year_archive_base_url.rstrip('/')),
                        'output_path': str(output_path),
                        'year_archive_filename': build_year_archive_filename(year),
                        'year_archive_url': build_year_archive_url(year, args.year_archive_base_url.rstrip('/')),
                        'year_archive_path': str(year_archive_cache_dir / build_year_archive_filename(year)),
                        'year_archive_member': legacy_member_name(day),
                        'status': 'failed',
                        'error': str(exc),
                    }
                    record_result(day, row, None, 'failed')
                continue

            with zipfile.ZipFile(year_zip_path) as archive_zip:
                for day in year_days:
                    row, present_path, status_key = extract_archive_from_year_zip(
                        day,
                        archive_zip=archive_zip,
                        output_dir=output_dir,
                        year_archive_path=year_zip_path,
                        year_archive_url=year_zip_url,
                        year_archive_meta=year_archive_meta,
                        args=args,
                    )
                    record_result(day, row, present_path, status_key)

    archive_rows = [row for row in ordered_rows if row is not None]
    present_paths = [path for path in ordered_paths if path is not None]

    progress.close(
        status=format_status(
            downloaded=downloaded_count,
            skipped_existing=skipped_existing_count,
            skipped_unpublished=skipped_unpublished_count,
            failed=failed_count,
        ),
    )

    if output_list_path is not None and not args.dry_run:
        write_lines(output_list_path, present_paths)

    summary = {
        'fetched_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'base_url': args.base_url,
        'year_archive_base_url': args.year_archive_base_url,
        'year_archive_cache_dir': str(year_archive_cache_dir),
        'output_dir': str(output_dir),
        'dry_run': args.dry_run,
        'date_count': len(planned_dates),
        'downloaded_count': downloaded_count,
        'skipped_existing_count': skipped_existing_count,
        'skipped_unpublished_count': skipped_unpublished_count,
        'failed_count': failed_count,
        'archives': archive_rows,
    }

    if output_list_path is not None:
        summary['output_list'] = str(output_list_path)
    if summary_path is not None:
        ensure_parent(summary_path)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    if not present_paths:
        hint = 'No archives were downloaded or found locally.'
        if failed_count > 0:
            hint = (
                'No usable archives were produced. All requested dates failed to download; '
                'this usually means the requested range is unavailable via the current daily archive path.'
            )
        elif skipped_unpublished_count > 0:
            hint = (
                'No usable archives were produced because every requested date was treated as unpublished. '
                'Try an older end date or a larger archive publish lag.'
            )

        summary_display = display_path(summary_path) if summary_path is not None else '<no summary path>'
        output_list_display = display_path(output_list_path) if output_list_path is not None else '<no output list>'
        raise SystemExit(
            f'{hint} See {summary_display} and {output_list_display} for details.'
        )


if __name__ == '__main__':
    main()
