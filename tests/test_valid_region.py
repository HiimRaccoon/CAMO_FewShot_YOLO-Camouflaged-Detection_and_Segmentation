"""T07 public geometry seams, using literal synthetic pixel footprints."""

import importlib

import pytest
import torch


def _api():
    return importlib.import_module("camo_fs.valid_region")


def test_square_image_has_no_letterbox_padding():
    # Break caught: a fully occupied network input loses valid image pixels.
    mask = _api().valid_letterbox_mask((4, 4), (8, 8))

    assert mask.dtype == torch.bool
    assert mask.shape == (8, 8)
    assert mask.tolist() == [[True] * 8] * 8


def test_letterbox_output_stays_on_cpu_under_a_non_cpu_device_context():
    # Break caught: implicit allocation inherits a training device context.
    with torch.device("meta"):
        mask = _api().valid_letterbox_mask((4, 4), (8, 8))

    assert mask.device.type == "cpu"
    assert mask.tolist() == [[True] * 8] * 8


@pytest.mark.parametrize(
    "source_hw,input_hw,expected",
    [
        ((4, 2), (4, 4), [[False, True, True, False]] * 4),
        ((2, 4), (4, 4), [[False] * 4, [True] * 4, [True] * 4, [False] * 4]),
        ((3, 2), (6, 7), [[False, True, True, True, True, False, False]] * 6),
        ((2, 3), (7, 6), [[False] * 6] + [[True] * 6] * 4 + [[False] * 6] * 2),
        ((3, 2), (4, 4), [[True, True, True, False]] * 4),
        ((6, 4), (3, 3), [[True, True, False]] * 3),
    ],
)
def test_centered_fit_excludes_tall_wide_and_odd_padding(source_hw, input_hw, expected):
    # Break caught: swapped H/W, incorrect resize, or odd padding on the wrong side.
    assert _api().valid_letterbox_mask(source_hw, input_hw).tolist() == expected


@pytest.mark.parametrize(
    "source_hw,input_hw,ratio_pad,expected",
    [
        ((2, 2), (4, 6), ((1.5, 1.0), (2, 1)),
         [[False] * 6] + [[False, False, True, True, True, False]] * 2 + [[False] * 6]),
        ((2, 3), (4, 6), ((1.0, 1.0), (0, 0)),
         [[True, True, True, False, False, False]] * 2 + [[False] * 6] * 2),
        ((3, 2), (4, 4), ((1.5, 4 / 3), (1, 0)), [[False, True, True, True]] * 4),
    ],
)
def test_explicit_geometry_controls_resize_and_leading_padding(source_hw, input_hw, ratio_pad, expected):
    # Break caught: ignoring adapter-provided dimensions, anisotropic ratios, or offset.
    mask = _api().valid_letterbox_mask(source_hw, input_hw, ratio_pad)
    assert mask.tolist() == expected


def test_feature_cells_require_all_eight_pixels_to_be_valid():
    # Break caught: nearest/any-valid admits 3 real pixels plus 5 padding pixels.
    pixels = torch.tensor([[True] * 11 + [False] * 5], dtype=torch.bool)

    reduced = _api().reduce_valid_mask(pixels, (1, 2))

    assert reduced.dtype == torch.bool
    assert reduced.tolist() == [[True, False]]


@pytest.mark.parametrize(
    "pixels,feature_hw,expected",
    [
        ([[True, True, False, True, True]], (1, 2), [[False, False]]),
        ([[True], [True], [False], [True], [True]], (2, 1), [[False], [False]]),
        ([[True, True, True, False, False]], (1, 2), [[True, False]]),
        ([[True] * 5] * 5, (2, 2), [[True, True], [True, True]]),
    ],
)
def test_nondivisible_grid_excludes_every_intersecting_padding_pixel(pixels, feature_hw, expected):
    # Break caught: rounding a fractional cell boundary discards a padding pixel.
    reduced = _api().reduce_valid_mask(torch.tensor(pixels), feature_hw)
    assert reduced.tolist() == expected


def test_horizontal_flip_moves_odd_padding_with_image_content():
    # Break caught: reconstructing centered validity after a flip leaves padding stale.
    original = _api().valid_letterbox_mask((8, 3), (8, 8))
    flipped = original.flip(-1)

    assert original.tolist() == [[False, False, True, True, True, False, False, False]] * 8
    assert flipped.tolist() == [[False, False, False, True, True, True, False, False]] * 8
    assert _api().reduce_valid_mask(original, (1, 4)).tolist() == [[False, True, False, False]]
    assert _api().reduce_valid_mask(flipped, (1, 4)).tolist() == [[False, False, True, False]]


def test_two_dimensional_cells_reject_a_single_padding_pixel():
    # Break caught: sampling only cell centers or requiring padding on a whole row.
    pixels = torch.ones((16, 16), dtype=torch.bool)
    pixels[0, 0] = False
    pixels[15, 15] = False
    assert _api().reduce_valid_mask(pixels, (2, 2)).tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize(
    "source_hw,input_hw,expected",
    [
        ((16, 16), (16, 16), [[True, True], [True, True]]),
        ((16, 8), (16, 16), [[False, False], [False, False]]),
        ((8, 16), (16, 16), [[False, False], [False, False]]),
        ((24, 16), (24, 24), [[False, True, False]] * 3),
    ],
)
def test_letterbox_to_stride_eight_grid_accepts_only_complete_image_cells(source_hw, input_hw, expected):
    pixels = _api().valid_letterbox_mask(source_hw, input_hw)
    assert _api().reduce_valid_mask(pixels, (input_hw[0] // 8, input_hw[1] // 8)).tolist() == expected


def test_stride_eight_cell_with_three_content_columns_and_five_padding_columns_is_invalid():
    pixels = _api().valid_letterbox_mask((8, 11), (8, 16), ((1, 1), (0, 0)))
    assert _api().reduce_valid_mask(pixels, (1, 2)).tolist() == [[True, False]]


def test_empty_batch_keeps_feature_shape_without_inventing_valid_cells():
    reduced = _api().reduce_valid_mask(torch.empty((0, 8, 16), dtype=torch.bool), (1, 2))
    assert reduced.shape == (0, 1, 2)
    assert reduced.dtype == torch.bool


def test_reduction_preserves_each_image_and_does_not_mutate_input():
    # Break caught: mixing batch validity or modifying the caller's current masks.
    pixels = torch.tensor([[[True, True, False, True]], [[False, False, True, True]]])
    before = pixels.clone()
    reduced = _api().reduce_valid_mask(pixels, (1, 2))
    assert reduced.tolist() == [[[True, False]], [[False, True]]]
    assert reduced.device == pixels.device
    assert torch.equal(pixels, before)
    assert _api().reduce_valid_mask(pixels, (1, 4)).tolist() == pixels.tolist()


@pytest.mark.parametrize("bad_hw", [(0, 4), (-1, 4), (2.5, 4), (True, 4), (4,), "44", None])
@pytest.mark.parametrize("parameter", ["source_hw", "input_hw"])
def test_letterbox_rejects_invalid_dimensions(parameter, bad_hw):
    # Break caught: malformed dimensions are guessed or yield an empty valid region.
    arguments = {"source_hw": (4, 4), "input_hw": (8, 8), parameter: bad_hw}
    with pytest.raises(ValueError, match=parameter):
        _api().valid_letterbox_mask(**arguments)


@pytest.mark.parametrize(
    "ratio_pad",
    [
        ((0, 1), (0, 0)), ((-1, 1), (0, 0)), ((float("nan"), 1), (0, 0)),
        ((float("inf"), 1), (0, 0)), ((True, 1), (0, 0)), (("1", 1), (0, 0)),
        ((1e308, 1), (0, 0)), ((1, 1), (-1, 0)), ((1, 1), (0, -1)),
        ((1, 1), (0.5, 0)), ((1, 1), (0, float("nan"))),
        ((1, 1), (True, 0)), ((1, 1), (7, 0)), ((1, 1), (0, 7)),
        ((3, 1), (0, 0)), ((1, 3), (0, 0)),
        ((1.1, 1), (0, 0)), ((1, 1.1), (0, 0)),
        (), (1, 0), ((1,), (0, 0)), ((1, 1), (0,)),
    ],
)
def test_explicit_geometry_rejects_invalid_or_ambiguous_pixel_placement(ratio_pad):
    # Break caught: clipping out-of-bounds geometry or guessing fractional rounding.
    with pytest.raises(ValueError, match="ratio_pad"):
        _api().valid_letterbox_mask((3, 3), (8, 8), ratio_pad)


@pytest.mark.parametrize(
    "valid,feature_hw",
    [
        (torch.ones((4, 4)), (2, 2)),
        (torch.ones((4, 4), dtype=torch.int64), (2, 2)),
        (torch.tensor(True), (1, 1)),
        (torch.ones(4, dtype=torch.bool), (1, 1)),
        (torch.empty((0, 4), dtype=torch.bool), (1, 1)),
        (torch.ones((4, 4), dtype=torch.bool), (0, 2)),
        (torch.ones((4, 4), dtype=torch.bool), (2.5, 2)),
        (torch.ones((4, 4), dtype=torch.bool), (True, 2)),
        (torch.ones((4, 4), dtype=torch.bool), (5, 2)),
        (torch.ones((4, 4), dtype=torch.bool), (2, 5)),
        (torch.ones((4, 4), dtype=torch.bool), (2,)),
        ([[True, False]], (1, 1)),
    ],
)
def test_reduction_rejects_non_boolean_masks_and_invalid_feature_grids(valid, feature_hw):
    # Break caught: implicit numeric mask conversion or upsampling used as reduction.
    with pytest.raises(ValueError, match="valid|feature_hw"):
        _api().reduce_valid_mask(valid, feature_hw)
