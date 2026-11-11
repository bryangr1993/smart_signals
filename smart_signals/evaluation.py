"""Metrics for synchronized manual and automatic presence counts."""

from collections import defaultdict


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(samples: list[dict]) -> dict:
    labeled = matched = paired = 0
    absolute_error = signed_error = 0
    occupied = empty = false_empty = false_occupied = 0

    for sample in samples:
        manual = sample["manual_count"]
        predicted = sample["predicted_count"]
        if manual is None:
            continue
        labeled += 1
        if predicted is None:
            continue

        paired += 1
        error = predicted - manual
        matched += error == 0
        absolute_error += abs(error)
        signed_error += error
        if manual == 0:
            empty += 1
            false_occupied += predicted > 0
        else:
            occupied += 1
            false_empty += predicted == 0

    return {
        "total_samples": len(samples),
        "samples": labeled,
        "matched": matched,
        "unavailable": labeled - paired,
        "predicted_samples": paired,
        "exact_agreement": _rate(matched, labeled),
        "mae": _rate(absolute_error, paired),
        "bias": _rate(signed_error, paired),
        "false_empty_rate": _rate(false_empty, occupied),
        "false_occupied_rate": _rate(false_occupied, empty),
        "occupied_pairs": occupied,
        "empty_pairs": empty,
        "coverage": _rate(labeled, len(samples)),
        "prediction_coverage": _rate(paired, labeled),
    }


def evaluate_samples(samples: list[dict]) -> dict:
    """Evaluate paired snapshots, grouped by condition and manual count.

    Each sample supplies ``condition``, ``manual_count`` and ``predicted_count``.
    Counts are non-negative integers or None. A missing manual count lowers
    reference coverage. A missing prediction against known truth lowers exact
    agreement and prediction coverage, while error rates use comparable pairs.
    """
    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_count_band: dict[str, list[dict]] = {"0": [], "1-2": [], "3+": []}

    for index, sample in enumerate(samples):
        condition = sample["condition"]
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(f"Sample {index}: condition must be a non-empty string")
        for field in ("manual_count", "predicted_count"):
            value = sample[field]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Sample {index}: {field} must be a non-negative integer or None")

        by_condition[condition.strip()].append(sample)
        manual = sample["manual_count"]
        if manual is None:
            continue
        if manual == 0:
            band = "0"
        elif manual <= 2:
            band = "1-2"
        else:
            band = "3+"
        by_count_band[band].append(sample)

    result = _summarize(samples)
    result["by_condition"] = {
        condition: _summarize(group) for condition, group in sorted(by_condition.items())
    }
    result["by_count_band"] = {band: _summarize(group) for band, group in by_count_band.items()}
    return result
