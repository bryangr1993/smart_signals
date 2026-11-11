import unittest

from smart_signals.geometry import box_anchor, contains_point, validate_roi


class GeometryTests(unittest.TestCase):
    def test_concave_polygon_and_its_boundary(self):
        points = [(0, 0), (1, 0), (1, 1), (0.5, 0.5), (0, 1)]
        roi = validate_roi(points + [points[0]])
        self.assertEqual(roi, tuple(points))
        self.assertTrue(contains_point(roi, (0.5, 0.25)))
        self.assertTrue(contains_point(roi, (0.25, 0.75)))
        self.assertTrue(contains_point(roi, (1, 1)))
        self.assertFalse(contains_point(roi, (0.5, 0.75)))
        self.assertFalse(contains_point(roi, (1.01, 0.5)))

    def test_malformed_polygons(self):
        invalid = [
            None,
            [(0, 0), (1, 0)],
            [(0, 0), (0.5, 0), (1, 0)],
            [(0, 0), (1, 0), (1, 1), (1, 0), (0, 1)],
            [(0, 0), (1, 1), (0, 1), (1, 0)],
            [(0, 0), (1, 0), (0.5, 0), (1, 1), (0, 1)],
            [(0, 0), (1.1, 0), (1, 1)],
            [(0, 0), (float("nan"), 0), (1, 1)],
        ]
        for roi in invalid:
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                validate_roi(roi)

    def test_configured_anchor_changes_membership(self):
        roi = validate_roi([(0, 0.5), (1, 0.5), (1, 1), (0, 1)])
        box = (0.1, 0.1, 0.4, 0.6)
        self.assertTrue(contains_point(roi, box_anchor(box, "bottom_center")))
        self.assertFalse(contains_point(roi, box_anchor(box, "center")))


if __name__ == "__main__":
    unittest.main()
