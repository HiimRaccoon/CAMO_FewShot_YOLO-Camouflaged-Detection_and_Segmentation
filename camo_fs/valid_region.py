"""Version-independent input validity and conservative feature-grid geometry.

T09 must translate verified loader geometry into this module's explicit inputs.
No Ultralytics metadata or augmentation policy is inferred here.
"""

import math
from numbers import Real

import torch


def valid_letterbox_mask(
    source_hw: tuple[int, int],
    input_hw: tuple[int, int],
    ratio_pad: tuple | None = None,
) -> torch.Tensor:
    """Return one CPU bool mask of shape ``input_hw`` (height, width).

    ``ratio_pad=((scale_x, scale_y), (left, top))`` describes the *actual*
    resized content: scales must recover integer dimensions from ``source_hw``
    within 1e-6 pixels, and padding is the integer leading offset, not half the
    total padding. This avoids guessing a loader's rounding. Content must fit
    inside the input; cropping and other geometry require their own adapter.

    With no explicit geometry, use a project-owned centered aspect fit:
    round sizes to nearest integer (ties to even), keep each at least one
    pixel, and place any odd extra padding on the bottom/right. This default
    is a synthetic geometry convention, not a verified Ultralytics contract.
    Flip this mask along with the transformed image and GT masks when needed.
    """
    source_h, source_w = _hw(source_hw, "source_hw")
    input_h, input_w = _hw(input_hw, "input_hw")
    if ratio_pad is None:
        gain = min(input_h / source_h, input_w / source_w)
        resized_h, resized_w = max(1, round(source_h * gain)), max(1, round(source_w * gain))
        top, left = (input_h - resized_h) // 2, (input_w - resized_w) // 2
    else:
        if (not isinstance(ratio_pad, (tuple, list)) or len(ratio_pad) != 2
                or any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in ratio_pad)):
            raise ValueError("ratio_pad must be ((scale_x, scale_y), (left, top))")
        (scale_x, scale_y), (left, top) = ratio_pad
        if any(not _finite_real(scale) or scale <= 0 for scale in (scale_x, scale_y)):
            raise ValueError("ratio_pad scales must be positive finite numbers")
        resized_h = _pixel_extent(source_h * scale_y, minimum=1)
        resized_w = _pixel_extent(source_w * scale_x, minimum=1)
        left, top = _pixel_extent(left, minimum=0), _pixel_extent(top, minimum=0)
        if left + resized_w > input_w or top + resized_h > input_h:
            raise ValueError("ratio_pad image content must fit entirely inside input_hw")
    mask = torch.zeros(input_hw, dtype=torch.bool, device="cpu")
    mask[top:top + resized_h, left:left + resized_w] = True
    return mask


def reduce_valid_mask(valid: torch.Tensor, feature_hw: tuple[int, int]) -> torch.Tensor:
    """Reduce a bool ``[..., H, W]`` mask to bool ``[..., rows, columns]``.

    Each cell covers an equal spatial fraction of the input; all intersecting
    pixels must be valid. Fractional boundaries conservatively include pixels
    on both sides. At stride 8, a cell is valid only if all 8 x 8 pixels are
    valid. Preserve leading axes/device and do not mutate the input. Upsampling
    is rejected; nearest-neighbor/any-valid rules cannot enforce this contract.
    """
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.ndim < 2 or any(size <= 0 for size in valid.shape[-2:])):
        raise ValueError("valid must be a bool tensor with nonempty trailing spatial dimensions")
    height, width = valid.shape[-2:]
    rows, columns = _hw(feature_hw, "feature_hw")
    if rows > height or columns > width:
        raise ValueError("feature_hw must not exceed the input validity dimensions")
    # Integer floor/ceil bounds include every pixel intersecting a feature cell.
    y = torch.arange(rows, device=valid.device)
    x = torch.arange(columns, device=valid.device)
    top, bottom = y * height // rows, ((y + 1) * height + rows - 1) // rows
    left, right = x * width // columns, ((x + 1) * width + columns - 1) // columns
    table = torch.zeros((*valid.shape[:-2], height + 1, width + 1),
                        dtype=torch.int64, device=valid.device)
    table[..., 1:, 1:] = (~valid).to(torch.int64).cumsum(-2).cumsum(-1)
    invalid_count = (
        table[..., bottom[:, None], right[None, :]]
        - table[..., top[:, None], right[None, :]]
        - table[..., bottom[:, None], left[None, :]]
        + table[..., top[:, None], left[None, :]]
    )
    return invalid_count == 0


def _hw(value: object, name: str) -> tuple[int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(size) is not int or size <= 0 for size in value)):
        raise ValueError(name + " must contain two positive integer dimensions (height, width)")
    return value[0], value[1]


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _pixel_extent(value: Real, *, minimum: int) -> int:
    if (not _finite_real(value) or value < minimum
            or not math.isclose(value, round(value), rel_tol=0, abs_tol=1e-6)):
        raise ValueError("ratio_pad must describe integer pixel extents and padding; provide actual resize ratios")
    return round(value)
