#!/usr/bin/env python3

import argparse
import gzip
import json

from tenhou_xml import official_json_to_mjai_lines


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a saved Tenhou replay JSON payload into Mortal-compatible mjai events.',
    )
    parser.add_argument(
        '--official-json',
        required=True,
        help='Path to a saved Tenhou replay JSON payload, such as mjlog2json output or an official sample-page JSON.',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output .json.gz path for the converted Mortal log.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lines = official_json_to_mjai_lines(args.official_json)
    with gzip.open(args.output, 'wt', encoding='utf-8') as f:
        for event in lines:
            f.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')))
            f.write('\n')
    print(f'wrote {len(lines)} events to {args.output}')


if __name__ == '__main__':
    main()
