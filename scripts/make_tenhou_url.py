#!/usr/bin/env python3
"""Generate tenhou.net URLs from a tenhou.net/6 JSON paifu file.

Tenhou's web viewer at http://tenhou.net/5/ accepts a `json` query
parameter that contains a complete paifu inline. This lets you display
a local game in the official Tenhou viewer without uploading it
anywhere -- and it's the same URL form NAGA accepts when you don't
have a real tenhou.net log ID.

This tool prints several candidate URLs because the exact accepted
encoding varies between tools that consume them:

  1. URL-encoded raw JSON (longest, simplest, works in most viewers)
  2. URL-encoded compact JSON (no spaces, shorter)
  3. Just the JSON content for manual paste

Usage:
    scripts/make_tenhou_url.py <tenhou.json> [--seat N]

`--seat N` (0..3) appends `&tw=N` to the URL so the viewer / NAGA
treats player N as the focus seat. NAGA in particular needs to know
which player is being analysed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Generate tenhou.net URL for a JSON paifu')
    p.add_argument('input', help='path to a tenhou.net/6 JSON paifu file')
    p.add_argument('--seat', type=int, default=0, choices=range(4),
                   help='player seat (0..3) for the viewer/NAGA focus (default: 0)')
    p.add_argument('--base', default='https://tenhou.net/5/',
                   help='tenhou viewer base URL (default https://tenhou.net/5/)')
    p.add_argument('--print-only', choices=('url', 'compact-url', 'json'),
                   help='print only one form (default prints all)')
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f'no such file: {in_path}')

    with in_path.open(encoding='utf-8') as f:
        data = json.load(f)

    # Two JSON serializations: pretty (rare in URLs) and compact (typical).
    json_pretty = json.dumps(data, ensure_ascii=False)
    json_compact = json.dumps(data, ensure_ascii=False, separators=(',', ':'))

    # URL-encode each. quote_plus is the standard for query parameters
    # (encodes spaces as '+', special chars as %xx). Tenhou's JS reads
    # location.search and decodeURIComponent's the json parameter, so
    # quote (with safe='') works. quote_plus is also OK because '+'
    # decodes to ' ' which won't appear in compact JSON.
    enc_pretty = urllib.parse.quote(json_pretty, safe='')
    enc_compact = urllib.parse.quote(json_compact, safe='')

    url_pretty = f'{args.base}?json={enc_pretty}&tw={args.seat}'
    url_compact = f'{args.base}?json={enc_compact}&tw={args.seat}'

    if args.print_only == 'json':
        print(json_compact)
        return
    if args.print_only == 'url':
        print(url_pretty)
        return
    if args.print_only == 'compact-url':
        print(url_compact)
        return

    print(f'input: {in_path}')
    print(f'raw JSON size:        {len(json_compact):>7d} bytes')
    print(f'url-encoded size:     {len(enc_compact):>7d} bytes')
    print(f'full URL length:      {len(url_compact):>7d} bytes')
    print()
    if len(url_compact) > 8000:
        print(f'WARNING: URL is {len(url_compact)} bytes — many servers reject URLs >8000 bytes.')
        print('         If NAGA rejects this, you may need to host the JSON externally')
        print('         (e.g., GitHub gist) and provide that URL instead.')
        print()
    print('=== compact URL (recommended) ===')
    print(url_compact)
    print()
    print('=== pretty URL (only if compact fails) ===')
    print(url_pretty)


if __name__ == '__main__':
    main()
