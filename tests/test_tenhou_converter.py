from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
MORTAL_DIR = ROOT / 'mortal'
FIXTURE_DIR = ROOT / 'tests' / 'fixtures' / 'tenhou'
RAW_TENHOU_DIR = ROOT / 'data' / 'raw' / 'tenhou'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from tenhou_xml import (
    build_normalized_manifest_row,
    official_json_to_mjai_lines,
    parse_tenhou_xml,
    round_arrays_to_mjai_events,
    xml_to_mjai_lines,
)

try:
    from libriichi.dataset import GameplayLoader, Grp
except Exception:
    GameplayLoader = None
    Grp = None


class TenhouConverterBehaviorTest(unittest.TestCase):
    maxDiff = None
    PHOENIX_BATCH_B_IDS = [
        '2022013100gm-00a9-0000-af91b2de',
        '2022080600gm-00a9-0000-06406b7f',
        '2022080600gm-00a9-0000-b8ad3aee',
        '2022080601gm-00a9-0000-e3595545',
        '2022080818gm-00a9-0000-6c4ec7d1',
        '2022081017gm-00e1-0000-2df24853',
        '2022081121gm-00a9-0000-372fcc17',
        '2022081318gm-00a9-0000-6c91213c',
    ]

    def convert_fixture(self, slug: str) -> list[dict]:
        fixture = FIXTURE_DIR / f'tenhou_editor_sample_{slug}.json'
        return official_json_to_mjai_lines(fixture)

    def convert_reference_fixture(self, name: str) -> list[dict]:
        fixture = FIXTURE_DIR / f'tenhou_reference_{name}.json'
        return official_json_to_mjai_lines(fixture)

    def convert_raw_xml_fixture(self, snapshot_id: str, source_game_id: str) -> list[dict]:
        xml_fixture = RAW_TENHOU_DIR / snapshot_id / f'{source_game_id}.xml'
        return xml_to_mjai_lines(xml_fixture)

    def convert_raw_json_fixture(self, snapshot_id: str, source_game_id: str) -> list[dict]:
        json_fixture = RAW_TENHOU_DIR / snapshot_id / f'{source_game_id}.mjlog2json.json'
        return official_json_to_mjai_lines(json_fixture)

    def build_minimal_round(self, result: list) -> list:
        hand0 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24]
        hand1 = [11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17]
        hand2 = [21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27]
        hand3 = [31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37]
        return [
            [0, 0, 0],
            [25000, 25000, 25000, 25000],
            [11],
            [],
            hand0, [], [],
            hand1, [], [],
            hand2, [], [],
            hand3, [], [],
            result,
        ]

    def assert_basic_structure(self, events: list[dict]) -> Counter:
        self.assertGreater(len(events), 3)
        self.assertEqual(events[0]['type'], 'start_game')
        self.assertEqual(events[-1]['type'], 'end_game')

        counts = Counter(event['type'] for event in events)
        self.assertEqual(counts['start_kyoku'], counts['end_kyoku'])
        self.assertGreaterEqual(counts['start_kyoku'], 1)
        self.assertGreaterEqual(counts['hora'] + counts['ryukyoku'], counts['end_kyoku'])
        return counts

    def write_events(self, slug: str, events: list[dict], out_dir: Path) -> Path:
        out_path = out_dir / f'{slug}.json.gz'
        with gzip.open(out_path, 'wt', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')))
                f.write('\n')
        return out_path

    def test_four_kans_sample_emits_all_supported_kan_variants(self):
        events = self.convert_fixture('four_kans')
        counts = self.assert_basic_structure(events)

        self.assertEqual(counts['daiminkan'], 1)
        self.assertEqual(counts['kakan'], 2)
        self.assertEqual(counts['ankan'], 1)
        self.assertEqual(counts['dora'], 4)
        self.assertEqual(counts['hora'], 1)

        dora_markers = [event['dora_marker'] for event in events if event['type'] == 'dora']
        self.assertEqual(dora_markers, ['S', '7m', 'S', '3m'])

    def test_rinshan_sample_ankan_is_followed_by_rinshan_draw_and_dora(self):
        events = self.convert_fixture('rinshan')
        self.assert_basic_structure(events)

        ankan_idx = next(i for i, event in enumerate(events) if event['type'] == 'ankan')
        self.assertEqual(events[ankan_idx]['actor'], 3)
        self.assertEqual(events[ankan_idx + 1], {'type': 'tsumo', 'actor': 3, 'pai': '6p'})
        self.assertEqual(events[ankan_idx + 2], {'type': 'dora', 'dora_marker': '1m'})
        self.assertEqual(events[ankan_idx + 3]['type'], 'hora')
        self.assertEqual(events[ankan_idx + 3]['actor'], 3)
        self.assertEqual(events[ankan_idx + 3]['target'], 3)

    def test_chankan_sample_kakan_is_immediately_interrupted_by_hora(self):
        events = self.convert_fixture('chankan')
        self.assert_basic_structure(events)

        kakan_idx = next(i for i, event in enumerate(events) if event['type'] == 'kakan')
        self.assertEqual(events[kakan_idx]['actor'], 3)
        self.assertEqual(events[kakan_idx + 1]['type'], 'hora')
        self.assertEqual(events[kakan_idx + 1]['actor'], 2)
        self.assertEqual(events[kakan_idx + 1]['target'], 3)

    def test_double_ron_sample_emits_two_hora_events(self):
        events = self.convert_fixture('double_ron')
        counts = self.assert_basic_structure(events)

        self.assertEqual(counts['hora'], 2)
        hora_events = [event for event in events if event['type'] == 'hora']
        self.assertEqual(hora_events[0]['actor'], 0)
        self.assertEqual(hora_events[0]['target'], 3)
        self.assertEqual(hora_events[0]['ura_markers'], ['S'])
        self.assertEqual(hora_events[1]['actor'], 2)
        self.assertEqual(hora_events[1]['target'], 3)
        self.assertNotIn('ura_markers', hora_events[1])

    def test_nagashi_mangan_sample_maps_to_ryukyoku_with_score_deltas(self):
        events = self.convert_fixture('nagashi_mangan')
        self.assert_basic_structure(events)

        ryukyoku = next(event for event in events if event['type'] == 'ryukyoku')
        self.assertEqual(ryukyoku['deltas'], [-4000, -4000, 12000, -4000])

    def test_abortive_draw_samples_map_to_zero_delta_ryukyoku(self):
        expected = {
            'triple_ron': [0, 0, 0, 0],
            'four_winds_draw': [0, 0, 0, 0],
            'four_riichi_draw': [0, 0, 0, 0],
            'nine_terminals_draw': [0, 0, 0, 0],
        }
        for slug, deltas in expected.items():
            with self.subTest(slug=slug):
                events = self.convert_fixture(slug)
                counts = self.assert_basic_structure(events)
                self.assertEqual(counts['ryukyoku'], 1)
                ryukyoku = next(event for event in events if event['type'] == 'ryukyoku')
                self.assertEqual(ryukyoku['deltas'], deltas)

    def test_four_riichi_sample_keeps_four_reach_accept_sequences(self):
        events = self.convert_fixture('four_riichi_draw')
        counts = self.assert_basic_structure(events)

        self.assertEqual(counts['reach'], 4)
        self.assertEqual(counts['reach_accepted'], 4)

    def test_lossy_reference_json_fails_clearly_when_post_kan_dora_is_missing(self):
        for name in ('abort_almost_nagashi_mangan', 'yakuman_kazoe_17'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'missing visible dora indicators for kan follow-up'):
                    self.convert_reference_fixture(name)

    def test_four_kan_abort_maps_to_zero_delta_ryukyoku(self):
        events = round_arrays_to_mjai_events(self.build_minimal_round(['四槓散了']))
        self.assertEqual(events[0]['type'], 'start_kyoku')
        self.assertEqual(events[-1]['type'], 'end_kyoku')
        self.assertEqual(events[-2], {'type': 'ryukyoku', 'deltas': [0, 0, 0, 0]})

    def test_legacy_wall_exhaust_strings_map_to_ryukyoku(self):
        for result in (['全員聴牌'], ['全員不聴']):
            with self.subTest(result=result[0]):
                events = round_arrays_to_mjai_events(self.build_minimal_round(result))
                self.assertEqual(events[-2], {'type': 'ryukyoku', 'deltas': [0, 0, 0, 0]})

    @unittest.skipUnless((RAW_TENHOU_DIR / '2026-03-28_phoenix_batch_b').exists(), 'raw Tenhou validation batch unavailable')
    def test_xml_converter_matches_oracle_output_on_phoenix_batch_b(self):
        snapshot_id = '2026-03-28_phoenix_batch_b'
        for source_game_id in self.PHOENIX_BATCH_B_IDS:
            with self.subTest(source_game_id=source_game_id):
                xml_events = self.convert_raw_xml_fixture(snapshot_id, source_game_id)
                json_events = self.convert_raw_json_fixture(snapshot_id, source_game_id)
                self.assertEqual(xml_events, json_events)

    @unittest.skipUnless((RAW_TENHOU_DIR / '2026-03-28_phoenix_batch_b').exists(), 'raw Tenhou validation batch unavailable')
    def test_xml_only_manifest_rows_keep_room_and_ruleset_metadata(self):
        snapshot_id = '2026-03-28_phoenix_batch_b'
        expected = {
            '2022013100gm-00a9-0000-af91b2de': ('鳳', '鳳南喰赤', 4),
            '2022081017gm-00e1-0000-2df24853': ('鳳', '鳳東喰赤速', 4),
        }
        for source_game_id, (room, ruleset, table_size) in expected.items():
            with self.subTest(source_game_id=source_game_id):
                xml_path = RAW_TENHOU_DIR / snapshot_id / f'{source_game_id}.xml'
                parsed = parse_tenhou_xml(xml_path, include_round_events=False)
                row = build_normalized_manifest_row(
                    parsed,
                    raw_snapshot_id=snapshot_id,
                    dataset_id='test_xml_only_manifest',
                    converter_version='tenhou-xml-v0',
                    validation_status='raw_inspected',
                )
                self.assertEqual(row['room'], room)
                self.assertEqual(row['ruleset'], ruleset)
                self.assertEqual(row['table_size'], table_size)

    @unittest.skipIf(GameplayLoader is None or Grp is None, 'libriichi dataset bindings unavailable')
    def test_supported_samples_load_through_gameplayloader_and_grp(self):
        expected_rounds = {
            'four_kans': 1,
            'rinshan': 1,
            'chankan': 1,
            'double_ron': 1,
            'nagashi_mangan': 1,
            'triple_ron': 1,
            'four_winds_draw': 1,
            'four_riichi_draw': 1,
            'nine_terminals_draw': 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            for slug, round_count in expected_rounds.items():
                events = self.convert_fixture(slug)
                log_path = self.write_events(slug, events, out_dir)

                gameplay_loader = GameplayLoader(version=1, oracle=False)
                gameplay_data = gameplay_loader.load_gz_log_files([str(log_path)])
                self.assertEqual(len(gameplay_data), 1)
                self.assertEqual(len(gameplay_data[0]), 4)
                action_counts = [len(game.take_actions()) for game in gameplay_data[0]]
                self.assertGreater(sum(action_counts), 0)

                grp_games = Grp.load_gz_log_files([str(log_path)])
                self.assertEqual(len(grp_games), 1)
                self.assertEqual(tuple(grp_games[0].take_feature().shape), (round_count, 7))


if __name__ == '__main__':
    unittest.main()
