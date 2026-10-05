"""Opt-in T10 official 1-shot CUDA epoch; never substitutes CPU evidence."""

import json
import os
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.integration


def test_kaggle_enhanced_one_epoch_smoke_gate(monkeypatch):
    weights = os.environ.get("CAMO_FS_SMOKE_WEIGHTS")
    if not weights:
        pytest.skip("T10 Kaggle epoch gate requires CAMO_FS_SMOKE_WEIGHTS")
    assert torch.cuda.is_available(), "T10 smoke gate requires CUDA single GPU 0"

    from camo_fs.paths import DatasetPaths
    from camo_fs.prepare import verify_prepared
    from camo_fs.runs import sha256_file
    from camo_fs.train import UltralyticsRuntime
    from scripts.train_yolo import main
    import camo_fs.triplet as triplet

    paths = DatasetPaths.from_root(
        Path(os.environ.get("CAMO_FS_DATA_ROOT", str(DatasetPaths.DEFAULT_DATA_ROOT))),
        Path(os.environ.get("CAMO_FS_WORK_ROOT", str(DatasetPaths.DEFAULT_WORK_ROOT))))
    prepared = verify_prepared(1, paths)
    assert (prepared["image_count"], prepared["annotation_count"], len(prepared["taxonomy"]["names"])) == (47, 47, 47)
    settings = paths.work_root / "t10-ultralytics-settings"
    settings.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(settings))
    monkeypatch.setenv("YOLO_OFFLINE", "true")

    # Observe the real auxiliary objective without changing its value, sampling,
    # RNG, or accumulated trainer gradients. Keep only numbers, never graphs.
    gradients, backbone = [], []
    real_sample = triplet.sample_and_loss

    def observe_sample(features, *args, **kwargs):
        result = real_sample(features, *args, **kwargs)
        p3, conv = torch.autograd.grad(result.loss, (features, backbone[0]), retain_graph=True)
        assert torch.isfinite(p3).all() and torch.isfinite(conv).all(), "Non-finite auxiliary gradient"
        gradients.append({"p3_l1": p3.detach().float().abs().sum().item(),
                          "backbone_l1": conv.detach().float().abs().sum().item()})
        return result

    class ObservedRuntime(UltralyticsRuntime):
        def load(self, checkpoint, *, base):
            model = super().load(checkpoint, base=base)

            def ready(trainer):
                backbone.append(next(module.weight for module in trainer.model.modules()
                                     if isinstance(module, torch.nn.Conv2d)))

            model.add_callback("on_pretrain_routine_end", ready)
            return model

    monkeypatch.setattr(triplet, "sample_and_loss", observe_sample)
    runtime = ObservedRuntime()
    arguments = ["--method", "fgbg-triplet", "--shot", "1", "--epochs", "1",
                 "--device", "0", "--seed", "2024", "--run-kind", "smoke", "--weights", weights,
                 "--data-root", str(paths.data_root), "--work-root", str(paths.work_root)]
    if os.environ.get("CAMO_FS_SMOKE_OVERWRITE") == "1":
        arguments.append("--overwrite")
    assert main(arguments, runtime=runtime) == 0, "Enhanced CLI epoch failed; inspect training_attempts.json"
    attempts = json.loads((paths.results_root / "training_attempts.json").read_text(encoding="utf-8"))
    last = Path(attempts[0]["last_pt"])
    root = last.parent.parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    evidence = {"status": "not_verified", "run_dir": str(root), "config_hash": manifest["config_hash"],
                "versions": {**manifest["versions"], "gpu": torch.cuda.get_device_name(0)},
                "last_checkpoint_sha256": sha256_file(last), "auxiliary_gradients": gradients,
                "triplet_training": manifest["triplet_training"]}
    report = root / "smoke_gate.json"
    report.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert manifest["method"] == "fgbg-triplet" and manifest["run_kind"] == "smoke"
    assert manifest["completed_epochs"] == 1 and manifest["status"] == "completed"
    assert evidence["last_checkpoint_sha256"] == manifest["last_checkpoint_sha256"]
    assert manifest["triplet_training"]["nonzero_raw_batches"] > 0, "No nonzero raw triplet batch"
    assert manifest["triplet_training"]["sampled_triplets"] > 0, "No valid triplets"
    assert any(item["p3_l1"] > 0 and item["backbone_l1"] > 0 for item in gradients), "No auxiliary P3/backbone gradient"

    # Final native validation uses only T04's train placeholder. Inference below
    # uses this run's last.pt via the path reserved for T11/T12, never best.pt.
    model = runtime.load_inference(last)
    train = paths.prepared_root / "shot_1/train"
    images = sorted(train / name for name in prepared["file_checksums"] if name.startswith("images/"))
    paired = None
    tried = []
    for confidence in (0.25, 0.01, 0.001, 0.0001):
        for image in images:
            assert image.resolve().is_relative_to((train / "images").resolve())
            result = model.predict(source=str(image), imgsz=640, device="0", conf=confidence,
                                   save=False, verbose=False)[0]
            tried.append({"image": str(image), "confidence": confidence})
            if result.boxes is None or result.masks is None or len(result.boxes) == 0:
                continue
            assert len(result.boxes) == len(result.masks)
            assert torch.isfinite(result.boxes.data).all() and torch.isfinite(result.masks.data).all()
            if not result.masks.data.any():
                continue
            paired = {"image": str(image), "confidence": confidence, "boxes": result.boxes.data.cpu().tolist(),
                      "mask_shape": list(result.masks.data.shape),
                      "mask_foreground_pixels": result.masks.data.sum(dim=(-2, -1)).cpu().tolist()}
            result.save(filename=str(root / "smoke_prediction.jpg"))
            break
        if paired:
            break
    evidence.update(prediction=paired, prediction_attempts=tried)
    if paired:
        evidence["status"] = "passed"
    report.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert paired, "No paired predicted box/mask on prepared TRAIN images; T10 remains open"
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))
