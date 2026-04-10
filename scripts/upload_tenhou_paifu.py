#!/usr/bin/env python3
"""Upload a tenhou.net/6 JSON paifu to a public anonymous file host
and print the resulting URL.

Useful when you want a third-party tool (such as NAGA) to fetch the
paifu by URL but the embedded-JSON tenhou.net/5/?json=... URL is too
long. Hosting the file externally gives you a short permanent URL.

Default host: https://0x0.st (Anonymous file hosting, no auth)
  - files persist for 30+ days for small files
  - returns a single short URL on success
  - file extension and Content-Type are preserved

Usage:
    scripts/upload_tenhou_paifu.py <tenhou.json>
    scripts/upload_tenhou_paifu.py --input-dir <dir>     # batch
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def upload_to_0x0st(file_path: Path) -> str:
    """Upload a file to 0x0.st via curl. Returns the public URL."""
    result = subprocess.run(
        [
            'curl', '-sS', '--fail',
            '-F', f'file=@{file_path}',
            '-A', 'mortal-step8/1.0 (https://github.com/Equim-chan/Mortal)',
            'https://0x0.st',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'curl failed (exit {result.returncode}): {result.stderr.strip()}'
        )
    url = result.stdout.strip()
    if not url.startswith('http'):
        raise RuntimeError(f'unexpected response from 0x0.st: {result.stdout!r}')
    return url


HOSTS = {
    '0x0.st': upload_to_0x0st,
}


def parse_args():
    p = argparse.ArgumentParser(description='Upload a tenhou JSON paifu to a public file host')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('input', nargs='?', help='single tenhou JSON file to upload')
    g.add_argument('--input-dir', help='directory of *.tenhou.json files (batch upload)')
    p.add_argument('--host', default='0x0.st', choices=sorted(HOSTS.keys()))
    p.add_argument('--output-jsonl', help='write {file, url, size} JSONL records to this path')
    return p.parse_args()


def upload_one(file_path: Path, host: str) -> dict:
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    upload_fn = HOSTS[host]
    url = upload_fn(file_path)
    return {
        'file': str(file_path),
        'url': url,
        'size': file_path.stat().st_size,
        'host': host,
    }


def main():
    args = parse_args()

    targets: list[Path]
    if args.input:
        targets = [Path(args.input)]
    else:
        d = Path(args.input_dir)
        targets = sorted(d.glob('*.tenhou.json'))
        if not targets:
            sys.exit(f'no *.tenhou.json files in {d}')

    out_path = Path(args.output_jsonl) if args.output_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open('w', encoding='utf-8')
    else:
        out_handle = None

    success = 0
    fail = 0
    try:
        for tgt in targets:
            try:
                rec = upload_one(tgt, args.host)
            except Exception as e:
                print(f'FAIL  {tgt.name}: {e}', file=sys.stderr)
                fail += 1
                continue
            success += 1
            print(f'{rec["url"]}  {tgt.name} ({rec["size"]} bytes)')
            if out_handle:
                out_handle.write(json.dumps(rec) + '\n')
                out_handle.flush()
    finally:
        if out_handle:
            out_handle.close()

    if len(targets) > 1:
        print(f'\nuploaded {success}/{len(targets)} ({fail} failures)', file=sys.stderr)
    if fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
