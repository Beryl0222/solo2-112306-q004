"""检查采集资料的身份、形状和可交叉引用性。"""

import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_fixture(name):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


class FixtureContractTests(unittest.TestCase):
    def test_record_manifest(self):
        manifest = read_fixture("manifest.json")
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        self.assertEqual(manifest["fixture"], "wavelet_tile")
        self.assertEqual(manifest["records"], ["primary.json", "revision.json"])
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(revision["replaces"], primary["capture"])
        self.assertNotEqual(revision["capture"], primary["capture"])

    def test_related_input_records(self):
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        frames = {f["id"]: f for f in primary["frames"]}
        covered = {}
        for name, frame in frames.items():
            rows = set()
            for strip in frame["arrival"]:
                values = np.asarray(strip["rows"], dtype=np.int64)
                self.assertEqual(values.shape[1], frame["width"])
                self.assertTrue(((values >= 0) & (values <= 4095)).all())
                rows.update(range(strip["first_row"], strip["first_row"] + len(values)))
            covered[name] = rows
        for event in revision["events"]:
            self.assertIn(event["frame"], frames)
            self.assertTrue(covered[event["frame"]].isdisjoint(event.get("known_missing_rows", [])))
        for query in revision["reads"]:
            self.assertLessEqual(query["layer"], frames[query["frame"]]["levels"])
