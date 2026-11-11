import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from smart_signals import VisionEngine


ROI = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.7), (0.2, 0.7)]
CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
INSIDE = [40, 20, 60, 50, 0.95, 2]
OUTSIDE = [82, 20, 95, 50, 0.95, 2]


class DetectorBoxes:
    def __init__(self, rows):
        self.data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)

    def cpu(self):
        return self

    def numpy(self):
        return self


class Detector:
    def __init__(self, frames, names=None):
        self.names = CLASSES if names is None else names
        self.frames = iter(frames)
        self.calls = []
        self.frame_shapes = []

    def predict(self, frame, **options):
        self.calls.append(options)
        self.frame_shapes.append(frame.shape)
        return [SimpleNamespace(boxes=DetectorBoxes(next(self.frames)))]


class Tracker:
    def __init__(self, frames):
        self.frames = iter(frames)

    def update(self, detections):
        return next(self.frames)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(29).integers(35, 220, size=(100, 100, 3), dtype=np.uint8)

    def test_count_uses_current_boxes_and_preserves_unassociated_newcomers(self):
        motorcycle = [44, 22, 57, 46, 0.92, 3]
        detector = Detector([[INSIDE, motorcycle], [OUTSIDE], []])
        # The tracker's filtered box stays inside after the detector observed the exit.
        stale_box = [[40, 20, 60, 50, 17, 0.95, 2, 0]]
        tracker = Tracker([stale_box, stale_box, stale_box])
        engine = VisionEngine({"roi": ROI})

        with patch.object(engine, "_load_detector", return_value=detector), patch.object(
            engine, "_create_tracker", return_value=tracker
        ):
            first, annotated = engine.process(self.frame, 0)
            second, _ = engine.process(self.frame, 0.1)
            third, _ = engine.process(self.frame, 0.2)

        self.assertEqual(first["vehicles_present"], 2)
        self.assertEqual(first["by_class"]["motorcycle"], 1)
        self.assertEqual([d["track_id"] for d in first["detections"]], [17, None])
        self.assertEqual(second["vehicles_present"], 0)
        self.assertFalse(second["detections"][0]["inside"])
        self.assertEqual(third["vehicles_present"], 0)
        self.assertEqual(third["detections"], [])
        self.assertEqual(annotated.shape, self.frame.shape)
        self.assertFalse(np.shares_memory(annotated, self.frame))
        json.dumps(first, allow_nan=False)

    def test_low_score_requires_current_association_and_same_id_counts_once(self):
        weak = [43, 15, 55, 45, 0.18, 2]
        other_weak = [61, 30, 68, 51, 0.15, 2]
        detector = Detector([[INSIDE, weak, other_weak], [weak, other_weak]])
        tracker = Tracker([
            [[*INSIDE[:4], 12, 0.95, 2, 0], [*weak[:4], 12, 0.18, 2, 1]],
            [[*weak[:4], 12, 0.18, 2, 0]],
        ])
        engine = VisionEngine({"roi": ROI})

        with patch.object(engine, "_load_detector", return_value=detector), patch.object(
            engine, "_create_tracker", return_value=tracker
        ):
            first, _ = engine.process(self.frame, 0)
            second, _ = engine.process(self.frame, 0.1)

        self.assertEqual(first["vehicles_present"], 1)
        self.assertEqual(second["vehicles_present"], 1)
        self.assertAlmostEqual(second["detections"][0]["confidence"], 0.18)
        self.assertEqual(detector.calls[0]["conf"], 0.1)

    def test_unreadable_or_black_frames_are_null_and_empty_valid_frames_are_zero(self):
        engine = VisionEngine({"roi": ROI, "tracking": False})
        detector = Detector([[]])
        with patch.object(engine, "_load_detector", return_value=detector) as load_detector:
            for frame in (np.empty((0, 0, 3), dtype=np.uint8), np.zeros_like(self.frame)):
                observation, preview = engine.process(frame, 0)
                self.assertEqual(observation["status"], "unavailable")
                self.assertIsNone(observation["vehicles_present"])
                self.assertEqual(preview.ndim, 3)
            load_detector.assert_not_called()
            observation, _ = engine.process(self.frame, 0)
        self.assertEqual(observation["status"], "valid")
        self.assertEqual(observation["vehicles_present"], 0)

    def test_tracker_failure_preserves_the_detector_and_class_names(self):
        engine = VisionEngine({"roi": ROI, "classes": ["small vehicle"], "device": "cpu"})
        custom_box = [*INSIDE[:5], 0]
        detector = Detector([[custom_box], [custom_box]], names={0: "Small Vehicle", 1: "person"})
        with patch.object(engine, "_load_detector", return_value=detector), patch.object(
            engine, "_create_tracker", side_effect=RuntimeError("tracker test")
        ) as create_tracker:
            observation, _ = engine.process(self.frame, 0)
            again, _ = engine.process(self.frame, 0.1)
        self.assertEqual(observation["status"], "degraded")
        self.assertEqual(observation["by_class"], {"small vehicle": 1})
        self.assertIsNone(observation["detections"][0]["track_id"])
        self.assertEqual(again["vehicles_present"], 1)
        self.assertEqual(detector.calls[0]["classes"], [0])
        self.assertEqual(detector.calls[1]["conf"], 0.25)
        create_tracker.assert_called_once()

    def test_dark_image_warns_without_discarding_a_readable_frame(self):
        engine = VisionEngine({"roi": ROI, "tracking": False})
        detector = Detector([[INSIDE]])
        dark_frame = self.frame // 15
        with patch.object(engine, "_load_detector", return_value=detector):
            observation, _ = engine.process(dark_frame, 0)
        self.assertEqual(observation["status"], "degraded")
        self.assertEqual(observation["vehicles_present"], 1)
        self.assertIn("Imagen con poca luz.", observation["warnings"])

    def test_time_discontinuity_and_reset_do_not_reload_model(self):
        detector = Detector([[INSIDE], [INSIDE], [INSIDE], [INSIDE]])
        engine = VisionEngine({"roi": ROI})
        trackers = [Tracker([[]]), Tracker([[], []]), Tracker([[]])]
        with patch.object(engine, "_load_detector", return_value=detector) as load, patch.object(
            engine, "_create_tracker", side_effect=trackers
        ) as create_tracker:
            engine.process(self.frame, 0)
            engine.process(self.frame, 2)
            engine.process(self.frame, 2.2)
            engine.reset()
            engine.process(self.frame, 0)
        load.assert_called_once()
        self.assertEqual(create_tracker.call_count, 3)

    def test_large_saturated_area_warns_without_fabricating_empty_count(self):
        engine = VisionEngine({'roi': ROI, 'tracking': False})
        detector = Detector([[INSIDE]])
        glare = self.frame.copy()
        glare[:glare.shape[0] // 3, :] = 255
        with patch.object(engine, '_load_detector', return_value=detector):
            observation, _ = engine.process(glare, 0)
        self.assertEqual(observation['status'], 'degraded')
        self.assertEqual(observation['vehicles_present'], 1)
        self.assertTrue(any('saturadas' in text for text in observation['warnings']))

    def test_roi_crop_remaps_current_and_margin_boxes_before_tracking(self):
        frame = np.tile(self.frame, (4, 6, 1))
        roi = [(0.3, 0.3), (0.6, 0.3), (0.6, 0.6), (0.3, 0.6)]
        engine = VisionEngine({"roi": roi, "crop_to_roi": True, "iou": 0.7})
        detector = Detector([[[50, 44, 90, 82, 0.95, 2], [-5, 44, 8, 82, 0.95, 3]]])
        detector.predictor = SimpleNamespace(device="cuda:0")
        tracker = Mock()
        tracker.update.return_value = []
        with patch.object(engine, "_load_detector", return_value=detector), patch.object(
            engine, "_create_tracker", return_value=tracker
        ):
            observation, annotated = engine.process(frame, 0)

        self.assertEqual(detector.frame_shapes, [(184, 244, 3)])
        self.assertTrue(detector.calls[0]["agnostic_nms"])
        self.assertEqual(detector.calls[0]["iou"], 0.7)
        self.assertEqual(observation["crop_box"], [148 / 600, 88 / 400, 392 / 600, 272 / 400])
        self.assertEqual(observation["effective_device"], "cuda:0")
        tracker_input = tracker.update.call_args.args[0]
        np.testing.assert_allclose(tracker_input.xywh, [[218, 151, 40, 38], [152, 151, 8, 38]])
        self.assertEqual(observation["vehicles_present"], 1)
        self.assertEqual(observation["detections"][0]["box"], [198 / 600, 132 / 400, 238 / 600, 170 / 400])
        self.assertEqual(observation["detections"][1]["box"], [148 / 600, 132 / 400, 156 / 600, 170 / 400])
        self.assertFalse(observation["detections"][1]["inside"])
        self.assertEqual(annotated.shape, frame.shape)

    def test_roi_crop_is_clipped_to_frame_edges(self):
        frame = np.tile(self.frame, (4, 6, 1))
        roi = [(0.7, 0.5), (1.0, 0.5), (1.0, 1.0), (0.7, 1.0)]
        engine = VisionEngine({"roi": roi, "crop_to_roi": True, "tracking": False})
        detector = Detector([[[190, 190, 500, 500, 0.95, 2]]])
        with patch.object(engine, "_load_detector", return_value=detector):
            observation, _ = engine.process(frame, 0)
        self.assertEqual(detector.frame_shapes, [(232, 212, 3)])
        self.assertEqual(observation["crop_box"], [388 / 600, 168 / 400, 1.0, 1.0])
        self.assertEqual(observation["detections"][0]["box"], [578 / 600, 358 / 400, 1.0, 1.0])
        self.assertEqual(observation["vehicles_present"], 1)

    def test_near_full_crop_and_default_keep_full_frame(self):
        frame = np.tile(self.frame, (10, 10, 1))
        cases = [
            {"roi": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], "crop_to_roi": True},
            {"roi": ROI},
        ]
        for config in cases:
            with self.subTest(config=config):
                engine = VisionEngine(dict(config, tracking=False))
                detector = Detector([[]])
                with patch.object(engine, "_load_detector", return_value=detector):
                    observation, _ = engine.process(frame, 0)
                self.assertEqual(detector.frame_shapes, [frame.shape])
                self.assertIsNone(observation["crop_box"])
                self.assertIsNone(observation["effective_device"])

    def test_missing_requested_classes_are_unavailable(self):
        engine = VisionEngine({"roi": ROI})
        detector = Detector([], names={0: "person"})
        with patch.object(engine, "_load_detector", return_value=detector):
            observation, _ = engine.process(self.frame, 0)
        self.assertEqual(observation["status"], "unavailable")
        self.assertIsNone(observation["vehicles_present"])
        self.assertEqual(detector.calls, [])


if __name__ == "__main__":
    unittest.main()
