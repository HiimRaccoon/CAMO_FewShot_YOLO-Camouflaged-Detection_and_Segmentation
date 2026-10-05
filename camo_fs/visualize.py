"""Deterministic, run-owned prediction renders and optional training debug views."""

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import random
import shutil
import tempfile
from typing import Sequence

from camo_fs.evaluate import discover_runs
from camo_fs.paths import DatasetPaths
from camo_fs.prepare import read_prepared_yaml
from camo_fs.runs import IDENTITY_FIELDS, prepared_data_sha256, sha256_file, summary_row
from camo_fs.train import UltralyticsRuntime


def select_images(images: Sequence[Path], num_images: int = 20, seed: int = 2024) -> list[Path]:
    """Sample a stable sorted image list with an isolated RNG and an available cap."""
    if type(num_images) is not int or num_images <= 0:
        raise ValueError("num_images must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ordered = sorted(Path(image) for image in images)
    return random.Random(seed).sample(ordered, min(num_images, len(ordered)))


def _managed_path(path: Path, work_root: Path) -> Path:
    absolute = Path(path).absolute()
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
            raise ValueError("Visualization artifacts must not contain symlinks/junctions")
    resolved = absolute.resolve()
    if (not resolved.is_relative_to(work_root.resolve())
            or resolved.is_relative_to(Path("/kaggle/input").resolve())):
        raise ValueError("Visualization artifacts must stay under the run's work root")
    return resolved


def _read_run(run_dir: Path) -> tuple[Path, dict]:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    summary_row(manifest)  # Frozen T05 verifies complete identity and declared ownership.
    root = _managed_path(run_dir, Path(manifest["runs_root"]).parent)
    if root != Path(manifest["run_dir"]).resolve():
        raise ValueError("Manifest belongs to another run directory")
    return root, manifest


def _array(value):
    import numpy as np

    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _validate_result(result, image: Path, names: dict) -> int:
    import numpy as np
    from PIL import Image

    with Image.open(image) as source:
        w, h = source.size
    if (Path(result.path).resolve() != image.resolve() or tuple(result.orig_shape) != (h, w)
            or result.orig_img.shape != (h, w, 3) or result.names != names):
        raise ValueError("Prediction image/shape/taxonomy must match the selected source")
    count = 0 if result.boxes is None else len(result.boxes.data)
    if count:
        boxes, classes, confidence = (_array(result.boxes.xyxy), _array(result.boxes.cls), _array(result.boxes.conf))
        if (boxes.shape != (count, 4) or classes.shape != (count,) or confidence.shape != (count,)
                or not all(np.isfinite(value).all() for value in (boxes, classes, confidence))
                or not np.isin(classes, list(names)).all() or ((confidence < 0) | (confidence > 1)).any()
                or (boxes < 0).any() or (boxes[:, [0, 2]] > w).any() or (boxes[:, [1, 3]] > h).any()
                or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("Prediction boxes/classes/confidences are invalid")
        if result.masks is None:
            raise ValueError("Predicted boxes must have paired instance masks")
    if result.masks is not None:
        masks = _array(result.masks.data)
        if masks.shape != (count, h, w) or not ((masks == 0) | (masks == 1)).all():
            raise ValueError("Prediction masks must align with instances at original image size")
    return count


def _owned_target(target: Path, row: dict, work_root: Path) -> None:
    if not target.exists():
        return
    try:
        report = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if any(report.get(key) != row[key] for key in IDENTITY_FIELDS):
            raise ValueError("identity mismatch")
        expected = {"manifest.json", *(record["output"] for record in report["images"])}
        actual = set()
        for path in target.rglob("*"):
            _managed_path(path, work_root)
            if path.is_file():
                actual.add(path.relative_to(target).as_posix())
        if actual != expected:
            raise ValueError("unmanaged or missing files")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Existing visualization output is not owned by this run: " + str(error)) from error


def visualize_one(run_dir: Path, data_yaml: Path, num_images: int = 20, conf: float = 0.25,
                  seed: int = 2024, *, runtime=None) -> list[Path]:
    """Render deterministic test predictions from this run's last.pt."""
    from PIL import Image

    if isinstance(conf, bool) or not isinstance(conf, Real) or not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("conf must be a finite fraction in [0, 1]")
    select_images([], num_images, seed)  # Validate selection settings before loading.
    root, manifest = _read_run(run_dir)
    row = summary_row(manifest)
    config = manifest["config"]
    if manifest["status"] != "completed":
        raise ValueError("Visualization requires a completed training run")
    if type(manifest.get("completed_epochs")) is not int or manifest["completed_epochs"] != config["epochs"]:
        raise ValueError("Run did not record completion of the requested fixed epochs")
    work_root = Path(manifest["runs_root"]).parent
    yaml = _managed_path(data_yaml, work_root)
    shot = work_root.resolve() / f"camo_fs_yolo/shot_{config['shot']}"
    if yaml != shot / "data.yaml":
        raise ValueError("Visualization YAML must belong to the run's own shot")
    data = read_prepared_yaml(yaml)
    test = _managed_path(Path(data["test"]), work_root)
    if test != shot.parent / "test/images":
        raise ValueError("Visualization must use the shared official prepared test")
    _managed_path(shot.parent / "category_mapping.json", work_root)
    if prepared_data_sha256(shot) != config["data_sha256"]:
        raise ValueError("Prepared data checksum changed since training")
    checkpoint = _managed_path(root / "weights/last.pt", work_root)
    expected_last = manifest.get("last_checkpoint_sha256")
    if not isinstance(expected_last, str) or not expected_last or not checkpoint.is_file():
        raise ValueError("Completed run is missing last checkpoint integrity evidence")
    if sha256_file(checkpoint) != expected_last:
        raise ValueError("Run last.pt checksum differs from the completed training checkpoint")
    images = select_images([path for path in test.rglob("*") if path.is_file()], num_images, seed)
    if not images:
        raise ValueError("Prepared test image list is empty")
    target = _managed_path(root / "visualizations", work_root)
    _owned_target(target, row, work_root)
    runtime = runtime if runtime is not None else UltralyticsRuntime()
    runtime.seed(seed, True)
    model = runtime.load_inference(checkpoint)
    names = dict(enumerate(manifest["prepared_dataset"]["taxonomy"]["names"]))
    if model.names != names:
        raise ValueError("Checkpoint taxonomy differs from the saved training mapping")
    records, saved = [], []
    stage = Path(tempfile.mkdtemp(prefix=".viz-stage-", dir=root))
    preserve = False
    try:
        fresh = stage / "fresh"
        fresh.mkdir()
        for image in images:
            predictions = model.predict(source=str(image), imgsz=config["imgsz"], device=config["device"],
                                        conf=conf, retina_masks=True, augment=False, save=False,
                                        save_txt=False, save_crop=False, show=False, classes=None,
                                        agnostic_nms=False, iou=0.7, max_det=300, half=False,
                                        project=str(root), name="visualizations", exist_ok=True, verbose=False)
            if len(predictions) != 1:
                raise ValueError("Prediction must return exactly one selected image")
            result = predictions[0]
            count = _validate_result(result, image, names)
            relative = image.relative_to(test).as_posix()
            filename = hashlib.sha256(relative.encode("utf-8")).hexdigest() + ".png"
            plotted = result.plot(boxes=True, masks=True, labels=True, conf=True, pil=False)
            Image.fromarray(plotted[:, :, ::-1]).save(fresh / filename)
            saved.append(target / filename)
            records.append({"source": relative, "output": filename, "detections": count})
        report = {"schema_version": 1, **{key: row[key] for key in IDENTITY_FIELDS},
                  "selection_seed": seed, "conf": conf, "num_images": num_images,
                  "last_checkpoint_sha256": expected_last, "images": records}
        (fresh / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _owned_target(target, row, work_root)
        backup = stage / "backup"
        if target.exists():
            target.replace(backup)
        try:
            fresh.replace(target)
        except OSError as error:
            if backup.exists():
                try:
                    backup.replace(target)
                except OSError as recovery_error:
                    preserve = True
                    raise ValueError(f"Visualization commit failed; recovery backup kept at {backup}: {recovery_error}") from error
            raise
    finally:
        if not preserve:
            _managed_path(stage, work_root)
            for path in stage.rglob("*"):
                _managed_path(path, work_root)
            shutil.rmtree(stage)
    return saved


def visualize_selected(shots: Sequence[int] | None, paths: DatasetPaths, *, method: str | None = None,
                       run_dir: Path | None = None, num_images: int = 20, conf: float = 0.25,
                       seed: int = 2024, include_smoke: bool = False,
                       continue_on_error: bool = False, runtime=None) -> list[dict]:
    """Render one owned run or every matching completed identity sequentially."""
    paths = DatasetPaths.from_root(paths.data_root.resolve(), paths.work_root.resolve())
    if run_dir is not None:
        if shots is not None:
            raise ValueError("Select either run_dir or shots")
        root, manifest = _read_run(run_dir)
        if not root.is_relative_to(paths.runs_root.resolve()):
            raise ValueError("Selected run must belong to this work root")
        if method is not None and method != manifest["method"]:
            raise ValueError("Selected method disagrees with the run manifest")
        if not include_smoke and manifest["run_kind"] == "smoke":
            raise ValueError("Smoke runs are excluded; opt in explicitly with --include-smoke")
        roots = [root]
    else:
        methods = ("baseline", "fgbg-triplet") if method == "all" else (method or "baseline",)
        roots = [root for selected in methods for root in discover_runs(paths.runs_root, method=selected,
                  shots=shots, include_smoke=include_smoke)]
    if not roots:
        raise ValueError("No completed matching runs found; smoke runs are excluded by default")
    outcomes = []
    for root in roots:
        _, manifest = _read_run(root)
        row = {key: manifest[key] for key in IDENTITY_FIELDS}
        row.update(run_dir=str(root), images=[], status="failed", error_message="")
        try:
            images = visualize_one(root, paths.prepared_root / f"shot_{manifest['shot']}/data.yaml",
                                   num_images, conf, seed, runtime=runtime)
            row.update(status="completed", images=[str(image) for image in images])
        except Exception as error:
            row["error_message"] = str(error) or type(error).__name__
        outcomes.append(row)
        if row["status"] == "failed" and not continue_on_error:
            break
    return outcomes


def render_sampler_debug(image, masks, batch_idx, valid, sampled_positions, feature_hw: tuple[int, int],
                         *, image_index: int = 0, split: str = "train"):
    """Return a PIL training debug image, without sampling, I/O or model hooks.

    Inputs are the SAME transformed image [3,H,W], per-instance GT masks,
    batch_idx, input validity and T08 [T,8] feature-grid positions. Validate
    every triplet against T08 nearest FG/all-intersecting BG and T07 all-valid
    geometry, then draw only the requested image. Never pass prediction masks
    or an official-test image. Float RGB images are [0,1]; uint8 RGB is [0,255].
    Left panel shows nearest-projected feature foreground and actual feature
    centers; right panel shows source GT without points. Both use green tint
    and dark padding. A/P/N colors are red/green/blue. Separate panels avoid
    suggesting that a projected foreground cell's center must lie in source GT.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image, ImageDraw
    from camo_fs.valid_region import reduce_valid_mask

    if split != "train":
        raise ValueError("Sampler debug is training-only")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3
            or image.dtype not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or not torch.isfinite(image).all()):
        raise ValueError("Debug image must be a finite RGB [3,H,W] tensor")
    if image.is_floating_point() and ((image < 0) | (image > 1)).any():
        raise ValueError("Float RGB debug image must be in [0,1]")
    h, w = image.shape[-2:]
    if (not isinstance(valid, torch.Tensor) or valid.ndim != 3 or valid.dtype != torch.bool
            or tuple(valid.shape[-2:]) != (h, w) or type(image_index) is not int
            or not 0 <= image_index < len(valid)):
        raise ValueError("Validity/image_index must describe the current transformed batch")
    if (not isinstance(masks, torch.Tensor) or masks.ndim != 3 or min(masks.shape[-2:]) <= 0
            or not ((masks == 0) | (masks == 1)).all()
            or not isinstance(batch_idx, torch.Tensor) or batch_idx.ndim != 1
            or batch_idx.dtype not in integer_types or len(batch_idx) != len(masks)
            or ((batch_idx < 0) | (batch_idx >= len(valid))).any()):
        raise ValueError("Debug masks must be binary transformed per-instance GT with image indices")
    if (not isinstance(feature_hw, (tuple, list)) or len(feature_hw) != 2
            or any(type(size) is not int or size <= 0 for size in feature_hw)
            or not isinstance(sampled_positions, torch.Tensor) or sampled_positions.ndim != 2
            or sampled_positions.shape[1] != 8 or sampled_positions.dtype != torch.int64):
        raise ValueError("Debug requires positive feature_hw and T08 int64 [T,8] positions")
    source = masks.detach().cpu().bool()
    indices = batch_idx.detach().cpu().long()
    feature_valid = reduce_valid_mask(valid.detach().cpu(), tuple(feature_hw))
    foreground = F.interpolate(source[:, None].float(), size=feature_hw, mode="nearest")[:, 0].bool()
    occupied = {}
    points = sampled_positions.detach().cpu().tolist()
    for instance, batch_image, ay, ax, py, px, ny, nx in points:
        if (not 0 <= instance < len(source) or not 0 <= batch_image < len(valid)
                or indices[instance].item() != batch_image
                or any(not 0 <= y < feature_hw[0] or not 0 <= x < feature_hw[1]
                       for y, x in ((ay, ax), (py, px), (ny, nx)))
                or (ay, ax) == (py, px)):
            raise ValueError("Debug triplet identities/coordinates violate the T08 contract")
        if batch_image not in occupied:
            union = source[indices == batch_image].any(dim=0)
            occupied[batch_image] = F.adaptive_max_pool2d(union[None, None].float(), feature_hw)[0, 0].bool()
        if (not foreground[instance, ay, ax] or not foreground[instance, py, px]
                or not all(feature_valid[batch_image, y, x] for y, x in ((ay, ax), (py, px), (ny, nx)))
                or occupied[batch_image][ny, nx]):
            raise ValueError("Debug points must stay in same-instance foreground or valid all-GT-free background")
    rgb = image.detach().cpu().float()
    if image.is_floating_point():
        rgb = rgb * 255
    array = rgb.permute(1, 2, 0).round().byte().numpy().copy()
    union = source[indices == image_index].any(dim=0)
    projected = foreground[indices == image_index].any(dim=0)
    padding = ~valid[image_index].detach().cpu().numpy()
    canvas = Image.new("RGB", (2 * w, h + 40), "white")
    for panel, mask in enumerate((projected, union)):
        overlay = F.interpolate(mask[None, None].float(), size=(h, w), mode="nearest")[0, 0].bool().numpy()
        tinted = array.copy()
        tinted[overlay] = (tinted[overlay].astype(float) * 0.7 + np.array([32, 180, 32]) * 0.3).astype(np.uint8)
        tinted[padding] = (tinted[padding].astype(float) * 0.35).astype(np.uint8)
        canvas.paste(Image.fromarray(tinted), (panel * w, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, h + 2), "FG grid", fill="black")
    draw.text((w + 2, h + 2), "GT source", fill="black")
    colors = ((255, 64, 64), (64, 255, 64), (64, 160, 255))
    radius = max(1, min(h // feature_hw[0], w // feature_hw[1]) // 4)
    for _, batch_image, *coordinates in points:
        if batch_image != image_index:
            continue
        for role, color in enumerate(colors):
            y, x = coordinates[2 * role:2 * role + 2]
            cy, cx = int((y + 0.5) * h / feature_hw[0]), int((x + 0.5) * w / feature_hw[1])
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    for role, (label, color) in enumerate(zip("APN", colors)):
        x = int((role + 0.5) * 2 * w / 3)
        draw.ellipse((x - 2, h + 18, x + 2, h + 22), fill=color)
        draw.text((x - 3, h + 24), label, fill="black")
    return canvas
