"""Guarded enhancement seams; native version bindings load only on demand."""

from contextlib import contextmanager
import math
from numbers import Real
from importlib import metadata
from copy import copy

import torch


COMPATIBLE_VERSION = "8.3.228"


def require_compatible_version(version=None):
    """Fail closed on unverified native interfaces; this is not a release pin."""
    if version is None:
        try:
            version = metadata.version("ultralytics")
        except metadata.PackageNotFoundError as error:
            raise RuntimeError("Ultralytics 8.3.228 compatibility candidate is required") from error
    if version != COMPATIBLE_VERSION:
        raise RuntimeError(f"Unverified Ultralytics {version}; extension supports candidate {COMPATIBLE_VERSION} only")
    return version


class LetterBoxTrace:
    """Observe native placement using its metadata and an independent all-one probe.

    Native preprocessing runs unchanged on the real image and instances. The
    same native LetterBox resizes a one-channel probe with zero padding, so no
    project code guesses upstream resize rounding. T07 receives integer extents
    measured from that probe, normalized against the original source dimensions.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, labels):
        import numpy as np
        from camo_fs.valid_region import valid_letterbox_mask

        try:
            source = tuple(labels["ori_shape"])
            pre = tuple(labels["img"].shape[:2])
            gains = tuple(labels["ratio_pad"])
            if (len(source) != 2 or any(type(size) is not int or size <= 0 for size in source)
                    or tuple(labels["resized_shape"]) != pre
                    or gains != tuple(size / original for size, original in zip(pre, source))):
                raise ValueError("contradictory source/resize metadata")
            target = labels.get("rect_shape", self.transform.new_shape)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing or contradictory native letterbox geometry") from error
        output = self.transform(labels)
        shadow = copy(self.transform)
        shadow.new_shape, shadow.padding_value = target, 0
        probe = shadow(image=np.ones((*pre, 1), dtype=np.uint8))
        valid = torch.from_numpy(np.asarray(probe[..., 0] == 1))
        coords = valid.nonzero()
        if not len(coords) or tuple(valid.shape) != tuple(output["img"].shape[:2]):
            raise ValueError("Native letterbox probe geometry is incompatible")
        top, left = coords.amin(0).tolist()
        bottom, right = (coords.amax(0) + 1).tolist()
        try:
            actual_gains, actual_pad = output["ratio_pad"]
            if tuple(actual_gains) != gains or tuple(actual_pad) != (left, top):
                raise ValueError("contradictory native padding metadata")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing or contradictory native padding geometry") from error
        resized = (bottom - top, right - left)
        ratio_pad = ((resized[1] / source[1], resized[0] / source[0]), (left, top))
        explicit = valid_letterbox_mask(source, tuple(valid.shape), ratio_pad)
        if not torch.equal(valid, explicit):
            raise ValueError("Native placement is not an integer letterbox geometry")
        output["camo_valid"] = explicit
        output["camo_geometry"] = {"source_hw": source, "input_hw": tuple(valid.shape),
                                   "ratio_pad": ratio_pad, "flipped": False}
        return output


class _FlipInstances:
    """Forward native instance operations while observing its actual flip call."""

    def __init__(self, instances):
        self.instances, self.flipped = instances, False

    def __getattr__(self, name):
        return getattr(self.instances, name)

    def fliplr(self, width):
        self.flipped = True
        return self.instances.fliplr(width)


class HorizontalFlipTrace:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, labels):
        proxy = _FlipInstances(labels["instances"])
        labels["instances"] = proxy
        try:
            output = self.transform(labels)
        finally:
            labels["instances"] = proxy.instances
        if proxy.flipped:
            output["camo_valid"] = output["camo_valid"].flip(-1)
            output["camo_geometry"]["flipped"] = not output["camo_geometry"]["flipped"]
        return output


def adapt_batch_geometry(batch):
    """Validate native collation of per-image trace/mask, then stack validity."""
    from camo_fs.valid_region import valid_letterbox_mask

    try:
        images, traces, masks = batch["img"], batch["camo_geometry"], batch["camo_valid"]
        if len(traces) != len(images) or len(masks) != len(images) or len(batch["ori_shape"]) != len(images):
            raise ValueError("trace/image count mismatch")
        for index, (trace, mask, source) in enumerate(zip(traces, masks, batch["ori_shape"])):
            if tuple(source) != tuple(trace["source_hw"]) or tuple(images.shape[-2:]) != tuple(trace["input_hw"]):
                raise ValueError("source/input shape mismatch")
            expected = valid_letterbox_mask(trace["source_hw"], trace["input_hw"], trace["ratio_pad"])
            if trace["flipped"]:
                expected = expected.flip(-1)
            if not torch.equal(mask.cpu(), expected):
                raise ValueError(f"placement mismatch for batch image {index}")
        batch["camo_valid"] = torch.stack(list(masks))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("Missing or contradictory native batch geometry: " + str(error)) from error
    return batch


def __getattr__(name):
    if name in {"FGSegmentationTrainer", "FGSegmentationModel", "instrument_dataset"}:
        require_compatible_version()
        from camo_fs import _ultralytics_83228
        return getattr(_ultralytics_83228, name)
    raise AttributeError(name)


class P3Capture:
    """One scoped, undetached first input of a semantically discovered Segment."""

    def __init__(self, model, segment_type, *, expected_channels):
        heads = [module for module in model.modules() if isinstance(module, segment_type)]
        if len(heads) != 1:
            raise ValueError("Expected exactly one semantic Segment head")
        self.model, self.head = model, heads[0]
        self.expected_channels = expected_channels
        self._feature = None
        self._active = False

    @property
    def feature(self):
        if not self._active:
            raise RuntimeError("P3 capture requires an active scope")
        if self._feature is None:
            raise RuntimeError("Missing current-forward P3 capture")
        return self._feature

    @contextmanager
    def scope(self, images, *, direct_predict=False):
        if self._active:
            raise RuntimeError("P3 capture scope is already active")
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or any(size <= 0 for size in images.shape)
                or any(size % 32 for size in images.shape[-2:])):
            raise ValueError("P3 requires BCHW images with spatial dimensions divisible by 32")
        self._active = True
        self._feature = None
        forwards = captures = 0

        def begin_forward(module, args):
            nonlocal forwards
            forwards += 1
            if forwards != 1:
                raise RuntimeError("P3 scope received multiple model forwards")
            if not args or args[0] is not images:
                raise RuntimeError("P3 scope received a different image tensor")

        def capture_head(module, args):
            nonlocal captures
            captures += 1
            if captures != 1:
                raise RuntimeError("P3 scope received multiple Segment captures")
            if len(args) != 1 or not isinstance(args[0], (list, tuple)) or len(args[0]) != 3:
                raise ValueError("Segment requires three spatial head inputs")
            for feature, stride in zip(args[0], (8, 16, 32)):
                if (not isinstance(feature, torch.Tensor) or feature.ndim != 4
                        or feature.shape[0] != images.shape[0]
                        or tuple(feature.shape[-2:]) != tuple(size // stride for size in images.shape[-2:])):
                    raise ValueError("P3/P4/P5 head batch/stride shape is incompatible")
            feature = args[0][0]
            if feature.shape[1] != self.expected_channels:
                raise ValueError("P3 channel count is incompatible")
            if torch.is_grad_enabled() and not feature.requires_grad:
                raise RuntimeError("P3 input has no gradient path")
            self._feature = feature

        root_handle = self.model.register_forward_pre_hook(begin_forward)
        handle = self.head.register_forward_pre_hook(capture_head)
        try:
            yield self
            _ = self.feature
            if forwards != 1 and not direct_predict:
                raise RuntimeError("Missing current model forward for P3")
        finally:
            handle.remove()
            root_handle.remove()
            self._feature = None
            self._active = False


class FGModelLossMixin:
    """Native 8.3.228 loss vector plus one weighted mean auxiliary objective.

    The trainer sums the four-element native vector. Add the auxiliary once to
    its first entry; detached native loss-items stay identical. No logging keys
    enter the native return contract. Evaluation delegates unchanged.
    """

    def configure_triplet(self, *, weight, margin, count, seed):
        for name, value, minimum in (("weight", weight, 0), ("margin", margin, 0)):
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    or value < minimum or (name == "margin" and value == 0)):
                raise ValueError(name + " must be finite and in its permitted range")
        for name, value, minimum in (("count", count, 1), ("seed", seed, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(name + " must be an integer in its permitted range")
        self._fg_settings = dict(weight=weight, margin=margin, count=count)
        self._fg_generator = torch.Generator().manual_seed(seed)
        self._fg_pending = None
        self.triplet_metrics = {}

    def predict(self, images, *args, **kwargs):
        self._fg_pending = None
        if not self.training or not hasattr(self, "_fg_settings"):
            return super().predict(images, *args, **kwargs)
        capture = P3Capture(self, self.segment_type, expected_channels=self.p3_channels())
        with capture.scope(images, direct_predict=True):
            prediction = super().predict(images, *args, **kwargs)
            feature = capture.feature
        self._fg_pending = (images, prediction, feature)
        return prediction

    def loss(self, batch, preds=None):
        if not self.training or not hasattr(self, "_fg_settings"):
            self._fg_pending = None
            return super().loss(batch, preds)
        self.triplet_metrics = {}
        try:
            native_result = super().loss(batch, preds)
            if type(native_result) is not tuple or len(native_result) != 2:
                raise RuntimeError("Incompatible native segmentation loss tuple contract")
            native, items = native_result
            pending = self._fg_pending
            if pending is None or pending[0] is not batch["img"] or (preds is not None and pending[1] is not preds):
                raise RuntimeError("Missing or stale current-forward P3 predictions")
            if (not isinstance(native, torch.Tensor) or native.shape != (4,)
                    or not isinstance(items, torch.Tensor) or items.shape != (4,)
                    or items.requires_grad or not native.is_floating_point()
                    or native.dtype != items.dtype or native.device != items.device):
                raise RuntimeError("Incompatible native segmentation loss return contract (expected vector[4], items[4])")
            if not torch.isfinite(items).all():
                raise FloatingPointError("Non-finite native loss items")
            from camo_fs.triplet import sample_and_loss

            settings = self._fg_settings
            valid = batch.get("camo_valid")
            if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                    or tuple(valid.shape) != (batch["img"].shape[0], *batch["img"].shape[-2:])):
                raise ValueError("Missing or contradictory batch camo_valid geometry")
            indices = batch["batch_idx"]
            if indices.is_floating_point():
                if not torch.isfinite(indices).all() or not torch.equal(indices, indices.round()):
                    raise ValueError("Native batch_idx must contain integral finite image indices")
                indices = indices.to(torch.int64)
            result = sample_and_loss(pending[2], batch["masks"], indices, batch["camo_valid"],
                                     settings["count"], settings["margin"], self._fg_generator)
            weighted = result.loss * settings["weight"]
            loss = native.clone()
            loss[0] = loss[0] + weighted
            if not torch.isfinite(weighted).all() or not torch.isfinite(loss).all():
                raise FloatingPointError("Non-finite native/weighted triplet loss")
            self.triplet_metrics = {
                "raw_triplet": result.loss.detach().item(), "weighted_triplet": weighted.detach().item(),
                "sampled_triplets": result.sampled_triplets, "skipped_instances": result.skipped_instances,
            }
            return loss, items
        finally:
            self._fg_pending = None
