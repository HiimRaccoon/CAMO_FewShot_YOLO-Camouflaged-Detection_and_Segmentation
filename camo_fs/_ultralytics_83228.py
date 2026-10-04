"""Concrete provisional native bindings. Import only through ultralytics_ext."""

import numpy as np

from camo_fs.ultralytics_ext import (FGModelLossMixin, HorizontalFlipTrace, LetterBoxTrace,
                                    adapt_batch_geometry, require_compatible_version)

require_compatible_version()

from ultralytics.data.augment import (Albumentations, Compose, CopyPaste, CutMix, Format,
                                    LetterBox, MixUp, Mosaic, RandomFlip, RandomHSV, RandomPerspective)
from ultralytics.nn.modules import Segment
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import DEFAULT_CFG, RANK


class FGSegmentationModel(FGModelLossMixin, SegmentationModel):
    segment_type = Segment

    def p3_channels(self):
        heads = [module for module in self.modules() if isinstance(module, Segment)]
        if len(heads) != 1 or tuple(heads[0].stride.tolist()) != (8, 16, 32):
            raise ValueError("Expected one Segment head with P3/P4/P5 strides 8/16/32")
        return heads[0].cv2[0][0].conv.in_channels


class FGSegmentationTrainer(SegmentationTrainer):
    """Native trainer with model/loss and traced train-loader seams only.

    Pass triplet settings as constructor keywords, not native config overrides.
    CLI dispatch and the real training/checkpoint smoke gate belong to T10.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, *, triplet_weight=0.1,
                 triplet_margin=0.3, triplets_per_instance=16):
        self.triplet_settings = dict(weight=triplet_weight, margin=triplet_margin, count=triplets_per_instance)
        super().__init__(cfg, dict(overrides or {}), _callbacks)
        if self.args.compile:
            raise ValueError("Enhanced capture requires compile=False")

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Mirror the inspected 8.3.228 factory, constructing our subclass so
        # its initialization runs normally instead of changing a live class.
        model = FGSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        model.p3_channels()  # semantic stride/channel preflight before training
        model.configure_triplet(**self.triplet_settings, seed=self.args.seed)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        return instrument_dataset(dataset, self.args) if mode == "train" else dataset

    def preprocess_batch(self, batch):
        return super().preprocess_batch(adapt_batch_geometry(batch))


class CheckedPerspective(RandomPerspective):
    def affine_transform(self, image, border):
        output, matrix, scale = super().affine_transform(image, border)
        if not np.array_equal(matrix, np.eye(3)) or scale != 1 or output.shape != image.shape:
            raise ValueError("Enhanced geometry requires identity RandomPerspective after letterbox")
        return output, matrix, scale


def instrument_dataset(dataset, hyp):
    """Keep the native pipeline; observe letterbox/flip and forbid other geometry."""
    for name in ("degrees", "translate", "scale", "shear", "perspective", "flipud", "mosaic",
                 "mixup", "copy_paste", "cutmix", "multi_scale", "close_mosaic"):
        if getattr(hyp, name, 0) != 0:
            raise ValueError("Unsupported enhanced geometry option: " + name)
    if hyp.overlap_mask or getattr(hyp, "augmentations", None) or dataset.rect:
        raise ValueError("Enhanced loader requires per-instance masks and the project geometry preset")
    seen = set()

    def visit(transform):
        if id(transform) in seen:
            return transform
        seen.add(id(transform))
        if isinstance(transform, Compose):
            transform.transforms = [visit(item) for item in transform.transforms]
        elif isinstance(transform, LetterBox):
            return LetterBoxTrace(transform)
        elif isinstance(transform, RandomPerspective):
            transform.__class__ = CheckedPerspective
            transform.pre_transform = visit(transform.pre_transform)
        elif isinstance(transform, RandomFlip):
            if transform.direction == "horizontal":
                return HorizontalFlipTrace(transform)
            if transform.p != 0:
                raise ValueError("Vertical flips are unsupported")
        elif isinstance(transform, (Mosaic, MixUp, CutMix, CopyPaste)):
            if transform.p != 0:
                raise ValueError("Mixed-image geometry is unsupported")
        elif isinstance(transform, Albumentations):
            if getattr(transform, "contains_spatial", False):
                raise ValueError("Spatial Albumentations is unsupported")
        elif isinstance(transform, Format):
            if not transform.return_mask or transform.mask_overlap:
                raise ValueError("Loader must supply non-overlapping per-instance mask format")
        elif not isinstance(transform, RandomHSV):
            raise ValueError("Unverified native transform: " + type(transform).__name__)
        return transform

    dataset.transforms = visit(dataset.transforms)
    return dataset
