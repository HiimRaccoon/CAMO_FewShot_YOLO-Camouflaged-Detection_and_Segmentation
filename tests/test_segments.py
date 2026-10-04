import math

import pytest

from camo_fs.segments import DataIntegrityError, annotation_to_yolo


IMAGE = {"id": 1, "file_name": "fixture.png", "width": 100, "height": 50}
CATEGORY_TO_INDEX = {5: 0, 20: 1}


def _annotation(
    segmentation: object = None,
    *,
    category_id: int = 20,
    bbox: object = None,
) -> dict:
    return {
        "id": 7,
        "image_id": 1,
        "category_id": category_id,
        "bbox": [0, 0, 50, 25] if bbox is None else bbox,
        "segmentation": [[0, 0, 50, 0, 50, 25]] if segmentation is None else segmentation,
    }


def _coordinates(line: str) -> list[float]:
    return [float(value) for value in line.split()[1:]]


def _rasterize_even_odd(points: list[tuple[float, float]], width: int, height: int) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for row in range(height):
        for column in range(width):
            x, y = column + 0.5, row + 0.5
            inside = False
            previous = points[-1]
            for current in points:
                if (current[1] > y) != (previous[1] > y):
                    crossing_x = (previous[0] - current[0]) * (y - current[1]) / (previous[1] - current[1]) + current[0]
                    if x < crossing_x:
                        inside = not inside
                previous = current
            if inside:
                pixels.add((column, row))
    return pixels


def test_annotation_to_yolo_normalizes_simple_polygon_with_contiguous_class_index() -> None:
    line, multi_polygon = annotation_to_yolo(_annotation(), IMAGE, CATEGORY_TO_INDEX)

    assert line == "1 0 0 0.5 0 0.5 0.5"
    assert multi_polygon is False


@pytest.mark.parametrize("x,y", [(-0.5, 0), (0, -0.5), (-0.5, -0.5)])
def test_bbox_accepts_exact_camo_half_pixel_boundary_without_mutating_input(x, y):
    from copy import deepcopy

    annotation = _annotation(bbox=[x, y, 50, 25])
    original = deepcopy(annotation)
    line, _ = annotation_to_yolo(annotation, IMAGE, CATEGORY_TO_INDEX)
    assert line == "1 0 0 0.5 0 0.5 0.5"
    assert annotation == original


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("value", [-1, -0.5001, -0.4999, -0.1])
def test_bbox_rejects_every_negative_coordinate_except_exact_sentinel(axis, value):
    bbox = [0, 0, 50, 25]
    bbox[axis] = value
    with pytest.raises(DataIntegrityError, match="bounds"):
        annotation_to_yolo(_annotation(bbox=bbox), IMAGE, CATEGORY_TO_INDEX)


@pytest.mark.parametrize("bbox", [[-0.5, 0, 100.5001, 1], [0, -0.5, 1, 50.5001],
                                  [-0.5, 0, 0, 1], [0, -0.5, 1, 0]])
def test_bbox_sentinel_does_not_relax_extent_or_positive_size(bbox):
    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(_annotation(bbox=bbox), IMAGE, CATEGORY_TO_INDEX)


@pytest.mark.parametrize("x,y", [(-0.5, 0), (0, -0.5), (-0.5, -0.5)])
def test_polygon_normalizes_only_exact_half_pixel_sentinel_without_mutation(x, y):
    from copy import deepcopy

    annotation = _annotation([[x, y, 100, y, 100, 50, x, 50]], bbox=[x, y, 100 - x, 50 - y])
    original, image, mapping = deepcopy(annotation), deepcopy(IMAGE), deepcopy(CATEGORY_TO_INDEX)
    line, multi = annotation_to_yolo(annotation, IMAGE, CATEGORY_TO_INDEX)
    assert line == "1 0 0 1 0 1 1 0 1"
    assert all(0 <= value <= 1 for value in _coordinates(line))
    assert not multi
    assert annotation == original and IMAGE == image and CATEGORY_TO_INDEX == mapping


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("value", [-1, -0.5001, -0.4999, -0.1])
def test_polygon_rejects_every_negative_coordinate_except_exact_sentinel(axis, value):
    polygon = [0, 0, 100, 0, 100, 50, 0, 50]
    polygon[axis] = value
    with pytest.raises(DataIntegrityError, match="bounds"):
        annotation_to_yolo(_annotation([polygon]), IMAGE, CATEGORY_TO_INDEX)


@pytest.mark.parametrize("polygon", [[-0.5, 0, 100.0001, 0, 50, 25],
                                     [0, -0.5, 100, 0, 50, 50.0001]])
def test_polygon_sentinel_does_not_relax_right_bottom_bounds(polygon):
    with pytest.raises(DataIntegrityError, match="bounds"):
        annotation_to_yolo(_annotation([polygon]), IMAGE, CATEGORY_TO_INDEX)


@pytest.mark.parametrize("polygon", [[-0.5, 0, 0, 0, 0, 1], [-0.5, 0, 0, 1, 0, 2]])
def test_polygon_requires_distinct_noncollinear_points_after_boundary_normalization(polygon):
    with pytest.raises(DataIntegrityError, match="distinct non-collinear"):
        annotation_to_yolo(_annotation([polygon]), IMAGE, CATEGORY_TO_INDEX)


def test_annotation_to_yolo_keeps_vertices_from_every_disconnected_polygon() -> None:
    annotation = _annotation(
        [
            [0, 0, 10, 0, 10, 10],
            [80, 30, 90, 30, 90, 40],
        ],
        bbox=[0, 0, 90, 40],
    )

    line, multi_polygon = annotation_to_yolo(annotation, IMAGE, CATEGORY_TO_INDEX)

    coordinates = _coordinates(line)
    assert multi_polygon is True
    assert len(coordinates) == 16
    assert {0.0, 0.1, 0.2, 0.6, 0.8, 0.9} <= set(coordinates)


def test_multi_polygon_yolo_output_contains_all_source_vertices() -> None:
    line, _ = annotation_to_yolo(
        _annotation(
            [
                [0, 0, 10, 0, 10, 10],
                [80, 30, 90, 30, 90, 40],
            ],
            bbox=[0, 0, 90, 40],
        ),
        IMAGE,
        CATEGORY_TO_INDEX,
    )

    coordinates = _coordinates(line)
    rendered_points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
    assert {(0.0, 0.0), (0.1, 0.0), (0.1, 0.2)} <= set(rendered_points)
    assert {(0.8, 0.6), (0.9, 0.6), (0.9, 0.8)} <= set(rendered_points)


def test_multi_polygon_merge_does_not_fill_large_gap_between_components() -> None:
    image = {**IMAGE, "width": 100, "height": 100}
    components = [
        [5, 5, 15, 5, 15, 15, 5, 15],
        [85, 85, 95, 85, 95, 95, 85, 95],
    ]
    line, _ = annotation_to_yolo(
        _annotation(components, bbox=[5, 5, 90, 90]),
        image,
        CATEGORY_TO_INDEX,
    )

    coordinates = _coordinates(line)
    merged_points = [
        (round(x * 100, 8), round(y * 100, 8))
        for x, y in zip(coordinates[::2], coordinates[1::2], strict=True)
    ]
    merged_mask = _rasterize_even_odd(merged_points, 100, 100)
    source_mask = set()
    for component in components:
        source_points = list(zip(component[::2], component[1::2], strict=True))
        source_mask |= _rasterize_even_odd(source_points, 100, 100)

    assert len(merged_mask) <= len(source_mask) + 4


def test_three_component_merge_preserves_middle_return_path() -> None:
    image = {**IMAGE, "width": 100, "height": 100}
    components = [
        [5, 5, 15, 5, 15, 15, 5, 15],
        [45, 5, 55, 5, 55, 15, 45, 15],
        [85, 5, 95, 5, 95, 15, 85, 15],
    ]
    annotation = _annotation(components, bbox=[5, 5, 90, 10])

    first_line, multi_polygon = annotation_to_yolo(annotation, image, CATEGORY_TO_INDEX)
    second_line, _ = annotation_to_yolo(annotation, image, CATEGORY_TO_INDEX)

    coordinates = _coordinates(first_line)
    merged_points = [
        (round(x * 100, 8), round(y * 100, 8))
        for x, y in zip(coordinates[::2], coordinates[1::2], strict=True)
    ]
    source_mask = set()
    for component in components:
        source_points = list(zip(component[::2], component[1::2], strict=True))
        source_mask |= _rasterize_even_odd(source_points, 100, 100)

    assert multi_polygon is True
    assert first_line == second_line
    assert {(5, 5), (15, 5), (45, 5), (55, 5), (85, 5), (95, 5)} <= set(merged_points)
    assert len(_rasterize_even_odd(merged_points, 100, 100)) <= len(source_mask) + 8


@pytest.mark.parametrize(
    ("annotation", "image", "mapping"),
    [
        (_annotation(segmentation={"counts": "abc", "size": [50, 100]}), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, 10]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, 10, 0]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, math.nan, 0, 10, 10]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(segmentation=[[0, 0, 101, 0, 10, 10]]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(bbox=[0, 0, 101, 1]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(bbox=[0, 0, -1, 1]), IMAGE, CATEGORY_TO_INDEX),
        (_annotation(), {**IMAGE, "width": 0}, CATEGORY_TO_INDEX),
        (_annotation(category_id=999), IMAGE, CATEGORY_TO_INDEX),
    ],
)
def test_annotation_to_yolo_rejects_malformed_input(
    annotation: dict,
    image: dict,
    mapping: dict[int, int],
) -> None:
    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(annotation, image, mapping)


def test_annotation_to_yolo_rejects_nonfinite_bbox() -> None:
    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(_annotation(bbox=[0, 0, math.inf, 1]), IMAGE, CATEGORY_TO_INDEX)


def test_annotation_to_yolo_rejects_polygon_with_fewer_than_three_distinct_points() -> None:
    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(
            _annotation([[10, 10, 10, 10, 10, 10]], bbox=[10, 10, 1, 1]),
            IMAGE,
            CATEGORY_TO_INDEX,
        )


def test_annotation_to_yolo_rejects_zero_area_polygon() -> None:
    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(
            _annotation([[0, 0, 10, 0, 20, 0]], bbox=[0, 0, 20, 1]),
            IMAGE,
            CATEGORY_TO_INDEX,
        )


def test_annotation_to_yolo_rejects_mismatched_annotation_and_image_ids() -> None:
    annotation = _annotation()
    annotation["image_id"] = 999

    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(annotation, IMAGE, CATEGORY_TO_INDEX)


@pytest.mark.parametrize("invalid_id", [None, "1", 1.0, True])
def test_annotation_to_yolo_rejects_missing_or_noninteger_image_id(invalid_id: object) -> None:
    annotation = _annotation()
    image = {**IMAGE}
    annotation["image_id"] = invalid_id
    image["id"] = invalid_id

    with pytest.raises(DataIntegrityError):
        annotation_to_yolo(annotation, image, CATEGORY_TO_INDEX)
