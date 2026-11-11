import unittest

from smart_signals.evaluation import evaluate_samples


def sample(manual, predicted, condition="day"):
    return {"manual_count": manual, "predicted_count": predicted, "condition": condition}


class PresenceEvaluationTests(unittest.TestCase):
    def test_zero_truth_counts_are_valid(self):
        result = evaluate_samples([sample(0, 0), sample(0, 2)])

        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["exact_agreement"], 0.5)
        self.assertEqual(result["mae"], 1)
        self.assertEqual(result["false_occupied_rate"], 0.5)
        self.assertIsNone(result["false_empty_rate"])
        self.assertEqual(result["by_count_band"]["0"]["samples"], 2)
        self.assertIsNone(result["by_count_band"]["1-2"]["exact_agreement"])

    def test_unobservable_truth_and_missing_predictions_have_different_effects(self):
        result = evaluate_samples([
            sample(None, 7), sample(None, None), sample(0, None),
            sample(1, 0), sample(2, 2), sample(3, None),
        ])

        self.assertEqual(result["total_samples"], 6)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["unavailable"], 2)
        self.assertAlmostEqual(result["coverage"], 4 / 6)
        self.assertEqual(result["prediction_coverage"], 0.5)
        self.assertEqual(result["exact_agreement"], 0.25)
        self.assertEqual(result["mae"], 0.5)
        self.assertEqual(result["false_empty_rate"], 0.5)
        self.assertIsNone(result["false_occupied_rate"])

    def test_signed_bias_can_cancel_without_canceling_absolute_error(self):
        result = evaluate_samples([sample(3, 5), sample(3, 1), sample(1, 2)])

        self.assertAlmostEqual(result["bias"], 1 / 3)
        self.assertAlmostEqual(result["mae"], 5 / 3)
        self.assertEqual(result["exact_agreement"], 0)

    def test_condition_and_manual_bands_have_their_own_denominators(self):
        result = evaluate_samples([
            sample(0, 0), sample(2, 0), sample(4, 4, "rain"),
            sample(0, 2, "rain"), sample(None, 1, "rain"),
            sample(None, None, "night"),
        ])

        conditions = result["by_condition"]
        self.assertEqual(conditions["day"]["false_empty_rate"], 1)
        self.assertEqual(conditions["rain"]["false_occupied_rate"], 1)
        self.assertAlmostEqual(conditions["rain"]["coverage"], 2 / 3)
        self.assertEqual(conditions["night"]["coverage"], 0)
        self.assertIsNone(conditions["night"]["exact_agreement"])
        bands = result["by_count_band"]
        self.assertEqual(bands["0"]["samples"], 2)
        self.assertEqual(bands["1-2"]["false_empty_rate"], 1)
        self.assertEqual(bands["3+"]["exact_agreement"], 1)

    def test_empty_input_and_invalid_count(self):
        result = evaluate_samples([])
        self.assertEqual(result["samples"], 0)
        self.assertIsNone(result["coverage"])
        self.assertIsNone(result["mae"])
        with self.assertRaisesRegex(ValueError, "predicted_count"):
            evaluate_samples([sample(1, -1)])
        with self.assertRaisesRegex(ValueError, "manual_count"):
            evaluate_samples([sample(True, 0)])


if __name__ == "__main__":
    unittest.main()
