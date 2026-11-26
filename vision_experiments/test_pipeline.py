import unittest
from uuid import uuid4

from PIL import Image

from vision import AnalyzedFrame, Query, VisionSession

CENTER = (39, 39, 59, 59)
ADJACENT = (51, 39, 71, 59)
EDGE = (5, 5, 25, 25)


class StubModels:
    def __init__(self):
        self.boxes = [(CENTER, 0.95), (ADJACENT, 0.8)]
        self.ids = {CENTER: "6", ADJACENT: "5", EDGE: "5"}
        self.gate_reads = []
        self.distance_reads = []

    def predict(self, image, category, box=None):
        if box is None:
            return self.boxes
        return dict(arrowDirection="left", distanceToSign="near", gateConfidence=None)

    def read_distance(self, image, category, box):
        self.distance_reads.append(box)
        return "medium"

    def read_gate(self, image, box):
        self.gate_reads.append(box)
        return self.ids.get(box), 0.99

    def frame(self, image, category):
        return AnalyzedFrame(image, category, self, self)


class VisionContractTests(unittest.TestCase):
    def setUp(self):
        self.models = StubModels()
        self.session = VisionSession(self.models)
        self.image = Image.new("RGB", (100, 100))
        self.query = Query("gate", uuid4(), 5)

    def call(self, time, query=None):
        return self.session.analyze(self.image, query or self.query, time)

    def assert_miss(self, result):
        self.assertEqual(
            set(result),
            {
                "queryID",
                "boundingBox",
                "observedGateID",
                "arrowDirection",
                "distanceToSign",
            },
        )
        self.assertTrue(all(v is None for k, v in result.items() if k != "queryID"))

    def test_fast_numbered_query_only_localizes(self):
        self.models.boxes = [(EDGE, 0.99), (CENTER, 0.98)]
        for time in (0, 0.1, 0.2, 0.8):
            result = self.call(time)
            self.assertEqual(result["boundingBox"]["x"], EDGE[0])
            self.assertIsNone(result["observedGateID"])
            self.assertIsNone(result["distanceToSign"])
            self.assertEqual(self.session.phase, "guiding")
            self.assertFalse(self.session.details_refreshed)
        self.assertEqual(self.models.gate_reads, [])
        self.assertEqual(self.models.distance_reads, [])

    def test_forced_details_only_check_selected_box(self):
        frame = self.models.frame(self.image, "gate")
        match = frame.locate(self.query)
        self.assertEqual(match.box, CENTER)
        self.assert_miss(frame.result(self.query, match, include_details=True))
        self.assertEqual(self.models.gate_reads, [CENTER])
        self.assertEqual(self.models.distance_reads, [])

    def test_wrong_gate_suppressed_and_next_candidate_centered(self):
        self.call(0)
        self.call(0.1)
        self.assert_miss(self.call(0.2))
        self.assertEqual(self.session.previous_box, CENTER)
        self.assertTrue(self.session.details_refreshed)
        self.assertEqual(self.session.phase, "rejected")
        self.assertEqual(self.models.gate_reads, [CENTER])
        moving = (40, 39, 60, 59)
        self.models.boxes = [(moving, 0.99), (ADJACENT, 0.75)]
        for time in (0.3, 0.4):
            self.assertEqual(self.call(time)["boundingBox"]["x"], ADJACENT[0])
            self.assertEqual(self.session.phase, "guiding")
            self.assertFalse(self.session.details_refreshed)
        self.assertEqual(self.call(0.5)["observedGateID"], "5")
        self.assertTrue(self.session.details_refreshed)
        self.assertEqual(self.session.phase, "verified")
        self.assertEqual(self.call(0.6)["distanceToSign"], "medium")
        self.assertFalse(self.session.details_refreshed)
        self.assertEqual(self.models.gate_reads, [CENTER, ADJACENT])
        self.assertEqual(self.models.distance_reads, [ADJACENT])

    def test_lost_rejected_track_can_reappear(self):
        self.models.boxes = [(CENTER, 0.92)]
        for time in (0, 0.1, 0.2, 0.3, 0.4):
            result = self.call(time)
        self.assert_miss(result)
        self.assertFalse(self.session.details_refreshed)
        self.assertEqual(self.session.phase, "rejected")
        self.assertEqual(self.models.gate_reads, [CENTER])
        self.models.boxes = []
        self.assert_miss(self.call(0.5))
        self.assertEqual(self.session.phase, "searching")
        self.models.boxes = [(CENTER, 0.91)]
        self.assertIsNotNone(self.call(0.6)["boundingBox"])
        self.assertEqual(self.session.phase, "guiding")

    def test_unreadable_retries_after_cooldown(self):
        self.models.ids[CENTER] = None
        self.call(0)
        self.call(0.1)
        self.assert_miss(self.call(0.2))
        self.assertTrue(self.session.details_refreshed)
        self.assertEqual(self.session.phase, "unreadable")
        self.models.ids[CENTER] = "5"
        self.assert_miss(self.call(0.35))
        self.assertFalse(self.session.details_refreshed)
        self.assertEqual(self.models.gate_reads, [CENTER])
        self.assertEqual(self.call(0.75)["observedGateID"], "5")
        self.assertEqual(self.models.gate_reads, [CENTER, CENTER])
        self.assertEqual(self.session.phase, "verified")

    def test_query_change_or_gap_resets_suppression(self):
        self.models.boxes = [(CENTER, 0.92)]
        for time in (0, 0.1, 0.2):
            self.call(time)
        self.assertIsNotNone(self.call(1.1)["boundingBox"])
        self.assertEqual(self.session.phase, "guiding")
        self.call(1.2)
        self.assert_miss(self.call(1.3))
        query = Query("gate", uuid4(), 6)
        self.assertIsNone(self.call(1.4, query)["observedGateID"])
        self.call(1.5, query)
        self.assertEqual(self.call(1.6, query)["observedGateID"], "6")

    def test_non_gate_detail_contract(self):
        query = Query("restaurant", uuid4())
        self.call(0, query)
        self.call(0.1, query)
        result = self.call(0.2, query)
        self.assertEqual(result["arrowDirection"], "left")
        self.assertIsNone(result["observedGateID"])
        self.assertNotIn("gateConfidence", result)
        self.assertEqual(self.models.gate_reads, [])


if __name__ == "__main__":
    unittest.main()
