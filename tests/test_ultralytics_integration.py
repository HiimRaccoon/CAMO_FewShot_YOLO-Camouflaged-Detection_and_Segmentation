"""Native CPU synthetic proof plus a distinct attached-checkpoint Kaggle gate."""

from copy import deepcopy
from pathlib import Path
import json
import os

import pytest
import torch


pytestmark = pytest.mark.integration


@pytest.fixture
def native(tmp_path, monkeypatch):
    # Directory must exist before import: native otherwise falls back to cwd.
    settings = tmp_path / "settings"
    settings.mkdir()
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(settings))
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    pytest.importorskip("ultralytics")
    from camo_fs.ultralytics_ext import require_compatible_version
    require_compatible_version()
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.instance import Instances
    import numpy as np
    return get_cfg, YOLODataset, LetterBox, Instances, np


@pytest.mark.parametrize("shape", [(19, 32), (32, 19), (13, 21)])
def test_traced_letterbox_matches_actual_native_image_and_padding(native, shape):
    from camo_fs.ultralytics_ext import LetterBoxTrace

    _, _, LetterBox, Instances, np = native
    labels = {"img": np.full((*shape, 3), 7, dtype=np.uint8), "ori_shape": shape,
              "resized_shape": shape, "ratio_pad": (1.0, 1.0),
              "instances": Instances(np.empty((0, 4), dtype=np.float32),
                                     np.empty((0, 1000, 2), dtype=np.float32), normalized=True)}
    transform = LetterBox((32, 32), scaleup=False)
    expected = transform(deepcopy(labels))
    actual = LetterBoxTrace(transform)(deepcopy(labels))
    assert np.array_equal(actual["img"], expected["img"])
    # Image pixels supply an independent placement oracle (padding is 114).
    assert torch.equal(actual["camo_valid"], torch.from_numpy(actual["img"][..., 0] == 7))
    assert actual["camo_geometry"]["source_hw"] == shape


@pytest.mark.parametrize("bad", ["missing", "source", "gain"])
def test_geometry_metadata_is_required_and_verified(native, bad):
    from camo_fs.ultralytics_ext import LetterBoxTrace

    _, _, LetterBox, Instances, np = native
    labels = {"img": np.ones((19, 32, 3), dtype=np.uint8), "ori_shape": (19, 32),
              "resized_shape": (19, 32), "ratio_pad": (1.0, 1.0),
              "instances": Instances(np.empty((0, 4), dtype=np.float32), normalized=True)}
    if bad == "missing":
        labels.pop("ratio_pad")
    elif bad == "source":
        labels["ori_shape"] = (0, 32)
    else:
        labels["ratio_pad"] = (2.0, 1.0)
    with pytest.raises(ValueError, match="geometry"):
        LetterBoxTrace(LetterBox((32, 32)))(labels)


@pytest.fixture
def prepared(tmp_path, native):
    from PIL import Image
    from camo_fs.paths import DatasetPaths
    from camo_fs.prepare import prepare_selected

    paths = DatasetPaths.from_root(tmp_path / "input", tmp_path / "work")
    paths.images_dir.mkdir(parents=True)
    paths.few_shot_dir.mkdir(parents=True)
    for name in ("support.png", "test.png"):
        Image.new("RGB", (64, 93), (7, 7, 7)).save(paths.images_dir / name)
    def document(identifier, name):
        return {"categories": [{"id": 5, "name": "Bat"}],
                "images": [{"id": identifier, "file_name": name, "width": 64, "height": 93}],
                "annotations": [{"id": identifier, "image_id": identifier, "category_id": 5,
                    "bbox": [8, 8, 46, 72], "segmentation": [[8, 8, 28, 8, 28, 32, 8, 32],
                                                           [32, 48, 54, 48, 54, 80, 32, 80]]}]}
    paths.test_json.write_text(json.dumps(document(99, "test.png")))
    (paths.few_shot_dir / "camo5_Bat_1shot_split1.json").write_text(json.dumps(document(1, "support.png")))
    outcome = prepare_selected([1], paths, False, False, val_train_placeholder=True)[0]
    assert outcome.status == "prepared"
    manifest = json.loads(outcome.manifest_path.read_text())
    assert len(manifest["multi_polygon_instances"]) == 1
    return paths.prepared_root / "shot_1/train/images"


def options(native, fliplr=0.0):
    get_cfg = native[0]
    return get_cfg(overrides={"imgsz": 64, "overlap_mask": False, "mosaic": 0.0,
        "mixup": 0.0, "copy_paste": 0.0, "cutmix": 0.0, "degrees": 0.0, "translate": 0.0,
        "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": fliplr,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "bgr": 0.0, "close_mosaic": 0})


def dataset(native, prepared, hyp):
    return native[1](img_path=str(prepared), data={"names": {0: "Bat"}, "channels": 3},
                     imgsz=64, batch_size=1, augment=True, hyp=hyp, task="segment", rect=False)


@pytest.mark.parametrize("fliplr", [0.0, 1.0])
def test_prepared_multi_polygon_native_loader_preserves_instance_and_geometry(native, prepared, fliplr):
    from camo_fs.ultralytics_ext import instrument_dataset, adapt_batch_geometry
    import random

    hyp = options(native, fliplr)
    base = dataset(native, prepared, deepcopy(hyp))
    enhanced = instrument_dataset(dataset(native, prepared, deepcopy(hyp)), hyp)
    # Native cache records corrupt count and warnings, including messages not logged via warnings.warn.
    cache = native[4].load(prepared.parent / "labels.cache", allow_pickle=True).item()
    assert cache["results"][3] == 0 and cache["msgs"] == []
    assert len(enhanced.labels) == 1 and len(enhanced.labels[0]["cls"]) == 1
    assert len(enhanced.labels[0]["segments"]) == 1
    line = (prepared.parent / "labels/support.txt").read_text().split()
    assert all(0 <= float(value) <= 1 for value in line[1:])
    random.seed(2024)
    native[4].random.seed(2024)
    expected = base[0]
    random.seed(2024)
    native[4].random.seed(2024)
    actual = enhanced[0]
    for key in ("img", "masks", "cls", "bboxes", "batch_idx"):
        assert torch.equal(expected[key], actual[key]), key
    assert actual["masks"].shape == (1, 16, 16)
    assert actual["camo_geometry"]["flipped"] == bool(fliplr)
    batch = adapt_batch_geometry(enhanced.collate_fn([actual]))
    assert torch.equal(batch["camo_valid"][0], batch["img"][0, 0] == 7)
    assert batch["batch_idx"].tolist() == [0]
    assert batch["camo_valid"].sum() == 64 * 45


def test_native_model_loss_contract_and_auxiliary_backbone_gradient(native, prepared, monkeypatch):
    from camo_fs.ultralytics_ext import FGSegmentationModel, instrument_dataset, adapt_batch_geometry
    from ultralytics.nn.tasks import SegmentationModel
    from torch.utils.data import DataLoader
    import camo_fs.triplet as triplet

    hyp = options(native, 1.0)
    data = instrument_dataset(dataset(native, prepared, hyp), hyp)
    batch = adapt_batch_geometry(next(iter(DataLoader(data, batch_size=1, collate_fn=data.collate_fn))))
    batch["img"] = batch["img"].float() / 255
    model = FGSegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
    model.args = hyp
    baseline = SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
    baseline.args = hyp
    baseline.load_state_dict(model.state_dict())
    model.configure_triplet(weight=0.1, margin=2.0, count=16, seed=2024)
    real, captures = triplet.sample_and_loss, []

    def sampler(features, *args, **kwargs):
        result = real(features, *args, **kwargs)
        features.retain_grad()
        captures.append((features, result.loss))
        return result

    monkeypatch.setattr(triplet, "sample_and_loss", sampler)
    native_loss, native_items = baseline.loss(batch)
    loss, items = model.loss(batch)
    assert loss.shape == items.shape == native_loss.shape == native_items.shape == (4,)
    assert torch.allclose(items, native_items)
    assert torch.allclose(loss[1:], native_loss[1:])
    assert torch.isfinite(loss).all()
    assert model.triplet_metrics["sampled_triplets"] == 16
    assert loss.sum().item() == pytest.approx(native_loss.sum().item() + model.triplet_metrics["weighted_triplet"], abs=1e-5)
    feature, auxiliary = captures[0]
    assert feature.shape == (1, 64, 8, 8)
    assert auxiliary > 0
    auxiliary.backward(retain_graph=True)
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    backbone_grad = model.model[0].conv.weight.grad
    assert torch.isfinite(backbone_grad).all() and backbone_grad.abs().sum() > 0
    loss.sum().backward()
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_extended_trainer_uses_native_weight_loading_and_adapts_loader(native, prepared, tmp_path, monkeypatch):
    from camo_fs.ultralytics_ext import FGSegmentationTrainer, FGSegmentationModel
    from ultralytics.nn.tasks import SegmentationModel
    import ultralytics.data.utils as data_utils

    # Fonts are an unrelated network boundary; no model/data/training behavior is mocked.
    monkeypatch.setattr(data_utils, "check_font", lambda *args, **kwargs: None)
    hyp = options(native, 1.0)
    overrides = {name: getattr(hyp, name) for name in ("imgsz", "overlap_mask", "mosaic", "mixup",
        "copy_paste", "cutmix", "degrees", "translate", "scale", "shear", "perspective", "flipud",
        "fliplr", "hsv_h", "hsv_s", "hsv_v", "bgr", "close_mosaic")}
    overrides.update(model="yolo11n-seg.yaml", data=str(prepared.parent.parent / "data.yaml"),
                     device="cpu", workers=0, project=str(tmp_path / "runs"), name="compat",
                     plots=False, val=False)
    trainer = FGSegmentationTrainer(overrides=overrides, triplet_weight=0.2,
                                    triplet_margin=2.0, triplets_per_instance=3)
    weights = SegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
    with torch.no_grad():
        weights.model[0].conv.weight.fill_(0.01)
    model = trainer.get_model(cfg="yolo11n-seg.yaml", weights=weights, verbose=False)
    assert isinstance(model, FGSegmentationModel)
    assert torch.equal(model.model[0].conv.weight, weights.model[0].conv.weight)
    model.args = trainer.args
    trainer.model = model
    loader = trainer.get_dataloader(str(prepared), batch_size=1, rank=-1, mode="train")
    batch = trainer.preprocess_batch(next(iter(loader)))
    assert batch["camo_valid"].shape == (1, 64, 64)
    loss, items = model.loss(batch)
    assert loss.shape == items.shape == (4,) and torch.isfinite(loss).all()
    assert model.triplet_metrics["sampled_triplets"] == 3
    assert model.triplet_metrics["weighted_triplet"] == pytest.approx(0.2 * model.triplet_metrics["raw_triplet"])


@pytest.mark.parametrize("changed", ["missing", "source-count", "placement"])
def test_collated_geometry_rejects_missing_or_contradictory_samples(native, prepared, changed):
    from camo_fs.ultralytics_ext import instrument_dataset, adapt_batch_geometry

    hyp = options(native)
    data = instrument_dataset(dataset(native, prepared, hyp), hyp)
    batch = data.collate_fn([data[0], data[0]])
    if changed == "missing":
        batch.pop("camo_geometry")
    elif changed == "source-count":
        batch["ori_shape"] = batch["ori_shape"][:1]
    else:
        batch["camo_valid"][1][0, 0] = True
    with pytest.raises(ValueError, match="geometry"):
        adapt_batch_geometry(batch)


@pytest.mark.parametrize("name,value", [("degrees", 1), ("translate", 0.1), ("scale", 0.1),
    ("shear", 1), ("perspective", 0.1), ("flipud", 0.1), ("mosaic", 1), ("mixup", 0.1),
    ("copy_paste", 0.1), ("cutmix", 0.1), ("multi_scale", 1), ("overlap_mask", True)])
def test_loader_rejects_unsupported_geometry_before_forward(native, prepared, name, value):
    from camo_fs.ultralytics_ext import instrument_dataset

    hyp = options(native)
    data = dataset(native, prepared, hyp)
    setattr(hyp, name, value)
    with pytest.raises(ValueError, match="geometry|mask"):
        instrument_dataset(data, hyp)


def test_extended_model_serializes_and_reloads_through_native_yolo(native, tmp_path):
    from camo_fs.ultralytics_ext import FGSegmentationModel
    from ultralytics import YOLO

    model = FGSegmentationModel("yolo11n-seg.yaml", nc=1, verbose=False)
    model.args = options(native)
    model.configure_triplet(weight=0.1, margin=0.3, count=16, seed=2024)
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"model": deepcopy(model).half(), "train_args": vars(model.args)}, checkpoint)
    loaded = YOLO(str(checkpoint)).model
    assert isinstance(loaded, FGSegmentationModel)
    assert torch.equal(loaded.model[0].conv.weight, model.model[0].conv.weight.half().float())
    with torch.no_grad():
        output = loaded(torch.rand(1, 3, 64, 64))
    assert torch.isfinite(output[0]).all()
    assert all(not module._forward_pre_hooks for module in loaded.modules())


def test_kaggle_attached_checkpoint_and_prepared_multi_polygon_gate(native, monkeypatch):
    """Opt in on Kaggle; synthetic CPU evidence never satisfies this gate."""
    weights = os.environ.get("CAMO_FS_INTEGRATION_WEIGHTS")
    if not weights:
        pytest.skip("Kaggle gate requires CAMO_FS_INTEGRATION_WEIGHTS and prepared official shot")
    assert torch.cuda.is_available(), "Kaggle compatibility gate requires a CUDA GPU"
    import inspect
    from camo_fs.paths import DatasetPaths
    from camo_fs.prepare import verify_prepared, read_prepared_yaml
    from camo_fs.train import UltralyticsRuntime
    from camo_fs.ultralytics_ext import (FGSegmentationModel, FGSegmentationTrainer, instrument_dataset,
                                         adapt_batch_geometry, P3Capture)
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.nn.modules import Segment
    import camo_fs.triplet as triplet

    paths = DatasetPaths.from_root(Path(os.environ.get("CAMO_FS_DATA_ROOT", str(DatasetPaths.DEFAULT_DATA_ROOT))),
                                  Path(os.environ.get("CAMO_FS_WORK_ROOT", "/kaggle/working")))
    shot = int(os.environ.get("CAMO_FS_INTEGRATION_SHOT", "1"))
    manifest = verify_prepared(shot, paths)
    assert manifest["multi_polygon_instances"], "Prepared shot needs at least one source multi-polygon instance"
    root = paths.prepared_root / f"shot_{shot}"
    mapping = read_prepared_yaml(root / "data.yaml")["names"]
    names = dict(enumerate(mapping))
    base = UltralyticsRuntime().load(Path(weights), base=True).model
    hyp = options(native, 1.0)
    hyp.imgsz = 640
    data = native[1](img_path=str(root / "train/images"), data={"names": names, "channels": 3},
                     imgsz=640, batch_size=1, augment=True, hyp=hyp, task="segment", rect=False)
    instrument_dataset(data, hyp)
    cache = native[4].load(root / "train/labels.cache", allow_pickle=True).item()
    assert cache["results"][3] == 0 and not cache["msgs"]
    filename = manifest["multi_polygon_instances"][0]["filename"]
    index = next(i for i, file in enumerate(data.im_files) if Path(file) == root / "train/images" / filename)
    lines = (root / "train/labels" / Path(filename).with_suffix(".txt")).read_text().splitlines()
    assert all(0 <= float(value) <= 1 for line in lines for value in line.split()[1:])
    assert len(data.labels[index]["cls"]) == len(lines)
    datum = data[index]
    assert len(datum["cls"]) == len(lines), "Native transform silently dropped a prepared source instance"
    batch = adapt_batch_geometry(data.collate_fn([datum]))
    batch["img"] = batch["img"].to("cuda:0").float() / 255
    for key in ("masks", "batch_idx", "cls", "bboxes", "camo_valid"):
        batch[key] = batch[key].to("cuda:0")
    assert len(batch["masks"]) == len(batch["batch_idx"]) == len(lines)
    model = FGSegmentationModel(base.yaml, nc=len(names), verbose=False)
    model.load(base)
    model.args, model.names = hyp, names
    model.to("cuda:0").train()
    model.configure_triplet(weight=0.1, margin=0.3, count=16, seed=2024)
    captures, actual_inputs = [], []
    real = triplet.sample_and_loss
    def sampler(features, *args, **kwargs):
        result = real(features, *args, **kwargs)
        features.retain_grad()
        captures.append((features, result))
        return result
    monkeypatch.setattr(triplet, "sample_and_loss", sampler)
    head = P3Capture(model, Segment, expected_channels=model.p3_channels()).head
    handle = head.register_forward_pre_hook(lambda module, args: actual_inputs.append(args[0][0]))
    try:
        preds = model(batch["img"])
    finally:
        handle.remove()
    native_loss, native_items = SegmentationModel.loss(model, batch, preds)
    loss, items = model.loss(batch, preds)
    feature, result = captures[0]
    assert feature is actual_inputs[0]
    assert tuple(feature.shape[-2:]) == (80, 80) and feature.requires_grad
    assert type((loss, items)) is tuple and loss.shape == items.shape == (4,)
    assert torch.equal(items, native_items) and torch.equal(loss[1:], native_loss[1:])
    assert torch.isfinite(loss).all() and result.sampled_triplets > 0
    assert loss.sum().item() == pytest.approx(native_loss.sum().item() + model.triplet_metrics["weighted_triplet"], abs=1e-5)
    result.loss.backward(retain_graph=True)
    assert feature.grad.abs().sum() > 0
    first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))
    assert torch.isfinite(first_conv.weight.grad).all() and first_conv.weight.grad.abs().sum() > 0
    loss.sum().backward()
    print(json.dumps({"ultralytics": "8.3.228", "torch": torch.__version__, "device": str(feature.device),
        "get_model": str(inspect.signature(FGSegmentationTrainer.get_model)),
        "native_loss": str(inspect.signature(SegmentationModel.loss)),
        "img_shape": list(batch["img"].shape), "masks_shape": list(batch["masks"].shape),
        "batch_idx_dtype": str(batch["batch_idx"].dtype), "p3_shape": list(feature.shape),
        "source_multi_polygon": filename, "triplet_metrics": model.triplet_metrics,
        "p3_grad_nonzero": True, "backbone_grad_nonzero": True}, sort_keys=True))
