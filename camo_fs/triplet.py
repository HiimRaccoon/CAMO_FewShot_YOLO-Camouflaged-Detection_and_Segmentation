"""Bounded FG/BG triplets from current-batch transformed instance masks."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

import torch
from torch.nn import functional as F

from camo_fs.valid_region import reduce_valid_mask


@dataclass(frozen=True)
class TripletResult:
    """Raw unweighted mean loss, sampling counts, and feature-grid coordinates.

    ``sampled_positions`` is int64 ``[T, 8]`` on the feature device, with rows
    ``[instance, image, ay, ax, py, px, ny, nx]``. T09 owns weighting and logging.
    """

    loss: torch.Tensor
    sampled_triplets: int
    skipped_instances: int
    sampled_positions: torch.Tensor


def sample_and_loss(
    features: torch.Tensor,
    masks: torch.Tensor,
    batch_idx: torch.Tensor,
    valid: torch.Tensor,
    count: int,
    margin: float,
    generator: torch.Generator,
    *,
    context: Mapping[str, object] | None = None,
) -> TripletResult:
    """Sample from transformed GT masks without inspecting YOLO or COCO inputs.

    Features are floating ``[B, C, Hf, Wf]``; masks are binary per-instance
    ``[N, Hm, Wm]``; integer ``batch_idx[N]`` associates masks with images.
    Bool ``valid[B, Hv, Wv]`` comes from the *same* image transforms and is
    all-valid reduced to actual feature dimensions before moving to the
    feature device. Foreground projects with nearest-neighbor. Background
    conservatively excludes every source-mask pixel intersecting its cell,
    including objects that disappear from nearest foreground projection.

    Draw exactly ``count`` triplets per usable instance, with replacement,
    but distinct anchor/positive positions. Instances with fewer than two
    valid FG positions or no same-image BG are skipped and counted. All RNG
    uses the explicit generator (which may be on CPU with CUDA features).
    Sampling never mutates inputs or detaches features. Empty sampling gives
    differentiable zero. Half/bfloat16 loss arithmetic promotes to float32;
    other feature precision is preserved. The normalization denominator is
    clamped at 1e-4 for float16 inputs (safe cast-back gradients near zero),
    and 1e-12 otherwise. Euclidean margin loss uses L2
    normalized vectors, mean reduction, and no distance epsilon or swapping.

    Optional ``context`` contains shot/method/batch identifiers for non-finite
    diagnostics. The CLI/config supplies engineering defaults of 16 and 0.3;
    this pure seam requires count, margin, and generator explicitly.
    """
    _validate(features, masks, batch_idx, valid, count, margin, generator, context)
    context = context or {}
    diagnostic = (f"method={context.get('method', 'fgbg-triplet')}, "
                  f"shot={context.get('shot', 'unspecified')}, batch={context.get('batch', 'unspecified')}, "
                  f"feature_shape={tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise FloatingPointError("non-finite triplet features: " + diagnostic)
    loss_dtype = torch.float32 if features.dtype in (torch.float16, torch.bfloat16) else features.dtype
    normalization_eps = 1e-4 if features.dtype == torch.float16 else 1e-12
    feature_hw = features.shape[-2:]
    source_masks = masks.to(device=features.device, dtype=torch.bool)
    masks = F.interpolate(masks[:, None].to(device=features.device, dtype=torch.float32),
                          size=feature_hw, mode="nearest")[:, 0].bool()
    valid = reduce_valid_mask(valid, feature_hw).to(features.device)
    image_indices = batch_idx.to(device=features.device, dtype=torch.int64)
    backgrounds: dict[int, torch.Tensor] = {}
    positions = []
    skipped = 0

    def draw(size: int) -> torch.Tensor:
        return torch.randint(size, (count,), generator=generator,
                             device=generator.device).to(features.device)

    for instance, image in enumerate(batch_idx.tolist()):
        foreground = torch.nonzero(masks[instance] & valid[image])
        if image not in backgrounds:
            union = source_masks[image_indices == image].any(dim=0)
            # Max occupancy over original floor/ceil footprints, even for
            # fractional upsampling or mixed axes; any GT pixel rules out BG.
            occupied = F.adaptive_max_pool2d(union[None, None].float(), feature_hw)[0, 0].bool()
            backgrounds[image] = torch.nonzero(~occupied & valid[image])
        background = backgrounds[image]
        if len(foreground) < 2 or len(background) == 0:
            skipped += 1
            continue

        anchor = draw(len(foreground))
        positive = draw(len(foreground) - 1)
        positive += positive >= anchor
        negative = draw(len(background))
        identities = torch.tensor([instance, image], device=features.device).expand(count, 2)
        positions.append(torch.cat((identities, foreground[anchor], foreground[positive], background[negative]), dim=1))
    if not positions:
        zero = features[..., :0].sum(dtype=loss_dtype)
        return TripletResult(zero, 0, skipped, torch.empty((0, 8), dtype=torch.int64, device=features.device))
    sampled = torch.cat(positions)
    image, ay, ax, py, px, ny, nx = sampled[:, 1:].unbind(dim=1)
    anchor = _normalize(features[image, :, ay, ax].to(loss_dtype), normalization_eps)
    positive = _normalize(features[image, :, py, px].to(loss_dtype), normalization_eps)
    negative = _normalize(features[image, :, ny, nx].to(loss_dtype), normalization_eps)
    loss = F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2, eps=0.0)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite auxiliary triplet loss: " + diagnostic)
    return TripletResult(loss, len(sampled), skipped, sampled)


def _normalize(vectors: torch.Tensor, eps: float) -> torch.Tensor:
    # Scaling large finite vectors first prevents overflow of the L2 norm.
    scale = vectors.abs().amax(dim=1, keepdim=True).clamp_min(1)
    return F.normalize(vectors / scale, p=2, dim=1, eps=eps)


def _validate(features, masks, batch_idx, valid, count, margin, generator, context) -> None:
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if (not isinstance(margin, Real) or isinstance(margin, bool)
            or not math.isfinite(margin) or margin <= 0):
        raise ValueError("margin must be a positive finite number")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be an explicit seeded torch.Generator")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("context must be a mapping of experiment diagnostic fields")
    if (not isinstance(features, torch.Tensor) or features.ndim != 4
            or features.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or any(size <= 0 for size in features.shape[1:])):
        raise ValueError("features must be a floating [B, C, H, W] tensor with nonempty C/H/W")
    if (not isinstance(masks, torch.Tensor) or masks.ndim != 3
            or any(size <= 0 for size in masks.shape[-2:]) or masks.is_complex()
            or not ((masks == 0) | (masks == 1)).all()):
        raise ValueError("masks must be binary transformed per-instance [N, H, W] masks")
    if (not isinstance(batch_idx, torch.Tensor) or batch_idx.ndim != 1
            or batch_idx.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or len(batch_idx) != len(masks)
            or ((batch_idx < 0) | (batch_idx >= features.shape[0])).any()):
        raise ValueError("batch_idx must contain one valid integer image index per instance mask")
    if (not isinstance(valid, torch.Tensor) or valid.ndim != 3 or valid.dtype != torch.bool
            or valid.shape[0] != features.shape[0]
            or any(size < target for size, target in zip(valid.shape[-2:], features.shape[-2:]))):
        raise ValueError("valid must be bool [B, H, W] validity at input or feature resolution")
