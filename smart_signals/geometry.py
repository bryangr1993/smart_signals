"""Geometry in normalized image coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

Point = tuple[float, float]
Polygon = tuple[Point, ...]
_EPSILON = 1e-12


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, point: Point) -> bool:
    return (
        abs(_cross(a, b, point)) <= _EPSILON
        and min(a[0], b[0]) - _EPSILON <= point[0] <= max(a[0], b[0]) + _EPSILON
        and min(a[1], b[1]) - _EPSILON <= point[1] <= max(a[1], b[1]) + _EPSILON
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def opposite(x: float, y: float) -> bool:
        return min(x, y) < -_EPSILON and max(x, y) > _EPSILON

    return (
        (opposite(_cross(a, b, c), _cross(a, b, d))
         and opposite(_cross(c, d, a), _cross(c, d, b)))
        or _on_segment(a, b, c)
        or _on_segment(a, b, d)
        or _on_segment(c, d, a)
        or _on_segment(c, d, b)
    )


def validate_roi(roi: Sequence[Sequence[float]]) -> Polygon:
    """Return a simple polygon with coordinates in [0, 1]."""
    if not isinstance(roi, (list, tuple)):
        raise ValueError("Dibuja un polígono para la zona de conteo.")

    points: list[Point] = []
    for vertex in roi:
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise ValueError("Cada vértice del polígono debe tener dos coordenadas.")
        if not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and 0 <= value <= 1
            for value in vertex
        ):
            raise ValueError("Las coordenadas del polígono deben estar entre 0 y 1.")
        points.append((float(vertex[0]), float(vertex[1])))

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("El polígono necesita al menos tres vértices distintos y sin repeticiones.")

    edges = list(zip(points, points[1:] + points[:1]))
    for index, (a, b) in enumerate(edges):
        c = points[(index + 2) % len(points)]
        if _on_segment(a, b, c) or _on_segment(b, c, a):
            raise ValueError("Los lados del polígono no pueden superponerse.")
        for other_index in range(index + 2, len(edges)):
            if index == 0 and other_index == len(edges) - 1:
                continue
            if _segments_intersect(a, b, *edges[other_index]):
                raise ValueError("Los lados del polígono no pueden cruzarse ni tocarse.")

    double_area = sum(a[0] * b[1] - a[1] * b[0] for a, b in edges)
    if abs(double_area) <= _EPSILON:
        raise ValueError("El polígono debe delimitar una superficie.")
    return tuple(points)


def contains_point(roi: Polygon, point: Point) -> bool:
    """Test a point against a validated polygon. The boundary belongs to the zone."""
    inside = False
    x, y = point
    previous = roi[-1]
    for vertex in roi:
        if _on_segment(previous, vertex, point):
            return True
        x1, y1 = previous
        x2, y2 = vertex
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
        previous = vertex
    return inside


def box_anchor(box: Sequence[float], anchor: str) -> Point:
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2
    if anchor == "bottom_center":
        return x, y2
    if anchor == "center":
        return x, (y1 + y2) / 2
    raise ValueError("El punto de referencia debe ser bottom_center o center.")
