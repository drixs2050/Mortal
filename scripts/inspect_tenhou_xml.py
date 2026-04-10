#!/usr/bin/env python3

import argparse
import json

from tenhou_xml import build_normalized_manifest_row, parse_tenhou_xml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect a raw Tenhou XML log and emit a summary or manifest-row skeleton.',
    )
    parser.add_argument('xml', help='Path to a Tenhou XML log.')
    parser.add_argument(
        '--official-json',
        default='',
        help='Optional saved official mjlog2json response used as a local verification oracle.',
    )
    parser.add_argument(
        '--mode',
        choices=('summary', 'manifest-row'),
        default='summary',
        help='Which JSON payload to emit.',
    )
    parser.add_argument(
        '--raw-snapshot-id',
        default='',
        help='Required for manifest-row mode.',
    )
    parser.add_argument(
        '--relative-path',
        default='',
        help='Optional normalized relative path for manifest-row mode.',
    )
    parser.add_argument(
        '--dataset-id',
        default='pending',
        help='Dataset id to place into manifest-row mode output.',
    )
    parser.add_argument(
        '--converter-version',
        default='tenhou-xml-readonly-v0',
        help='Converter version to place into manifest-row mode output.',
    )
    parser.add_argument(
        '--validation-status',
        default='raw_inspected',
        help='Validation status to place into manifest-row mode output.',
    )
    parser.add_argument(
        '--include-round-events',
        action='store_true',
        help='Include full per-round event lists instead of compact round summaries.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parsed = parse_tenhou_xml(
        args.xml,
        official_json_filename=args.official_json or None,
        include_round_events=args.include_round_events,
    )

    if args.mode == 'manifest-row':
        if not args.raw_snapshot_id:
            raise ValueError('--raw-snapshot-id is required for manifest-row mode')
        payload = build_normalized_manifest_row(
            parsed,
            raw_snapshot_id=args.raw_snapshot_id,
            relative_path=args.relative_path or None,
            dataset_id=args.dataset_id,
            converter_version=args.converter_version,
            validation_status=args.validation_status,
        )
    else:
        payload = parsed

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
