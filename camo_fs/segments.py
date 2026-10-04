"""Validated COCO polygon conversion for Ultralytics segmentation labels."""

from __future__ import annotations

import math
from typing import Any


class DataIntegrityError(ValueError):
    """Raised when a COCO annotation cannot safely become a YOLO segment."""


def annotation_to_yolo(
    annotation: dict[str, Any],
    image: dict[str, Any],
    category_to_index: dict[int, int],
) -> tuple[str, bool]:
    """Convert one polygon instance; normalize CAMO-FS's exact -0.5 sentinel.

    Only the known left/top half-pixel boundary becomes zero. Other out-of-
    bounds geometry fails, and source annotation/image dictionaries stay intact.
    """
    width, height = _image_dimensions(image)
    annotation_image_id = annotation.get("image_id")
    image_id = image.get("id")
    if (
        not isinstance(annotation_image_id, int)
        or isinstance(annotation_image_id, bool)
        or not isinstance(image_id, int)
        or isinstance(image_id, bool)
        or annotation_image_id != image_id
    ):
        raise DataIntegrityError("Annotation and image must have matching integer image IDs")
    category_id = annotation.get("category_id")
    if not isinstance(category_id, int) or category_id not in category_to_index:
        raise DataIntegrityError("Annotation category is absent from the canonical taxonomy")
    _validate_bbox(annotation.get("bbox"), width, height)
    components = _validate_polygons(annotation.get("segmentation"), width, height)

    merged = _merge_components(components)
    values = [coordinate for point in merged for coordinate in (point[0] / width, point[1] / height)]
    return " ".join([str(category_to_index[category_id]), *(_format(value) for value in values)]), len(components) > 1


def _image_dimensions(image: dict[str, Any]) -> tuple[float, float]:
    width = image.get("width")
    height = image.get("height")
    if not _is_finite_number(width) or not _is_finite_number(height) or width <= 0 or height <= 0:
        raise DataIntegrityError("Image width and height must be positive finite numbers")
    return float(width), float(height)


def _validate_bbox(bbox: object, width: float, height: float) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_finite_number(value) for value in bbox):
        raise DataIntegrityError("Bounding box must contain four finite numbers")
    x, y, box_width, box_height = (float(value) for value in bbox)
    # Official CAMO-FS uses exactly -0.5 at the left/top pixel boundary.
    # Keep bbox extents in source coordinates: right/bottom checks stay strict.
    if ((x < 0 and x != -0.5) or (y < 0 and y != -0.5)
            or box_width <= 0 or box_height <= 0 or x + box_width > width or y + box_height > height):
        raise DataIntegrityError("Bounding box is outside image bounds")


def _validate_polygons(segmentation: object, width: float, height: float) -> list[list[tuple[float, float]]]:
    if isinstance(segmentation, dict):
        raise DataIntegrityError("RLE segmentations are unsupported; polygon data is required")
    if not isinstance(segmentation, list) or not segmentation:
        raise DataIntegrityError("Segmentation must be a non-empty list of polygons")

    components: list[list[tuple[float, float]]] = []
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise DataIntegrityError("Each polygon needs at least three x/y point pairs")
        if not all(_is_finite_number(value) for value in polygon):
            raise DataIntegrityError("Polygon coordinates must be finite numbers")
        # Work on new points, then validate topology after normalization: two
        # distinct source vertices may collapse onto the same image boundary.
        normalized = [0.0 if value == -0.5 else float(value) for value in polygon]
        points = list(zip(normalized[::2], normalized[1::2]))
        if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
            raise DataIntegrityError("Polygon coordinates are outside image bounds")
        if len(set(points)) < 3 or _polygon_area(points) <= 1e-9:
            raise DataIntegrityError("Polygon must have at least three distinct non-collinear points")
        components.append(points)
    return components


def _merge_components(components: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    if len(components) == 1:
        return components[0]

    # YOLO accepts one contour per instance, so disconnected components require
    # a topology compromise. Deterministic nearest-boundary out-and-back bridges
    # preserve every component while avoiding a large enclosed gap between them.
    segments = [component.copy() for component in components]
    connection_indices: list[list[int]] = [[] for _ in segments]
    for index in range(1, len(segments)):
        left_index, right_index = _nearest_indices(segments[index - 1], segments[index])
        connection_indices[index - 1].append(left_index)
        connection_indices[index].append(right_index)

    merged: list[tuple[float, float]] = []
    for index, indices in enumerate(connection_indices):
        if len(indices) == 2 and indices[0] > indices[1]:
            indices = indices[::-1]
            segments[index] = segments[index][::-1]
        segments[index] = _rotate(segments[index], indices[0])
        segments[index] = [*segments[index], segments[index][0]]
        if index in {0, len(segments) - 1}:
            merged.extend(segments[index])
        else:
            merged.extend(segments[index][: indices[1] - indices[0] + 1])

    for index in range(len(segments) - 2, 0, -1):
        first_index, second_index = connection_indices[index]
        merged.extend(segments[index][abs(second_index - first_index) :])
    return merged


def _rotate(points: list[tuple[float, float]], index: int) -> list[tuple[float, float]]:
    return points[index:] + points[:index]


def _squared_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _nearest_indices(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> tuple[int, int]:
    return min(
        (
            (left_index, right_index)
            for left_index in range(len(left))
            for right_index in range(len(right))
        ),
        key=lambda indices: (
            _squared_distance(left[indices[0]], right[indices[1]]),
            indices[0],
            indices[1],
        ),
    )


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            point[0] * next_point[1] - next_point[0] * point[1]
            for point, next_point in zip(points, [*points[1:], points[0]], strict=True)
        )
    ) / 2


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _format(value: float) -> str:
    return format(value, ".12g")
