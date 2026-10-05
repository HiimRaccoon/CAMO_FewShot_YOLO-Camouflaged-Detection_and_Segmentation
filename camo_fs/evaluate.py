"""Final, identity-bound test evaluation of completed CAMO-FS last checkpoints."""

import json
import math
from pathlib import Path
from typing import Sequence

from camo_fs.paths import DatasetPaths
from camo_fs.prepare import read_prepared_yaml
from camo_fs.runs import prepared_data_sha256, summary_row, upsert_summary
from camo_fs.train import UltralyticsRuntime


def _managed_path(path: Path, work_root: Path) -> Path:
    """Reject input/output escapes and links before resolving the caller's path."""
    absolute = Path(path).absolute()
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
            raise ValueError("Evaluation artifacts must not contain symlinks/junctions")
    resolved = absolute.resolve()
    if (not resolved.is_relative_to(work_root.resolve())
            or resolved.is_relative_to(Path("/kaggle/input").resolve())):
        raise ValueError("Evaluation artifacts must stay under the run's work root")
    return resolved


def _read_run(run_dir: Path) -> tuple[Path, dict, dict]:
    root = Path(run_dir).absolute()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    row = summary_row(manifest)
    work_root = Path(manifest["runs_root"]).parent
    root = _managed_path(root, work_root)
    if root != Path(manifest["run_dir"]).resolve():
        raise ValueError("Manifest belongs to another run directory")
    return root, manifest, row


def evaluate_one(run_dir: Path, data_yaml: Path, results_dir: Path, *, runtime=None,
                 split: str = "test") -> dict:
    """Upsert official-test AP or a failed row; never alter training state.

    Untrustworthy identity or output ownership raises before reporting. Once a
    complete identity is verified, evaluation failures become empty-metric rows.
    """
    root, manifest, row = _read_run(run_dir)
    work_root = Path(manifest["runs_root"]).parent
    results = _managed_path(results_dir, work_root)
    if not results.is_relative_to(work_root.resolve() / "results"):
        raise ValueError("Evaluation summary must stay under the work results area")
    config = manifest["config"]
    row["weights_path"] = str(root / "weights/last.pt")
    try:
        if manifest["status"] != "completed":
            raise ValueError(manifest["error_message"] or "Evaluation requires a completed training run")
        if split != "test":
            raise ValueError("Final evaluation requires split='test'; val/train are forbidden")
        shot_dir = work_root.resolve() / f"camo_fs_yolo/shot_{config['shot']}"
        yaml = _managed_path(data_yaml, work_root)
        if yaml != shot_dir / "data.yaml":
            raise ValueError("Evaluation data YAML must belong to the run's own shot")
        data = read_prepared_yaml(yaml)
        test = _managed_path(Path(data["test"]), work_root)
        if test != shot_dir.parent / "test/images":
            raise ValueError("Evaluation test must reference the shared official prepared test")
        _managed_path(shot_dir.parent / "category_mapping.json", work_root)
        if prepared_data_sha256(shot_dir) != config["data_sha256"]:
            raise ValueError("Prepared data checksum changed since training")
        checkpoint = _managed_path(root / "weights/last.pt", work_root)
        if not checkpoint.is_file():
            raise ValueError("Run's own last.pt is missing; best.pt is never a fallback")
        output = _managed_path(root / "evaluation", work_root)
        runtime = runtime if runtime is not None else UltralyticsRuntime()
        runtime.seed(config["seed"], config["deterministic"])
        model = runtime.load_inference(checkpoint)
        names = dict(enumerate(manifest["prepared_dataset"]["taxonomy"]["names"]))
        if model.names != names:
            raise ValueError("Checkpoint taxonomy differs from the saved training mapping")

        def ready(validator):
            if validator.args.split != "test" or Path(validator.data["test"]).resolve() != test:
                raise ValueError("Native validator must evaluate only the shared official test")
            if Path(validator.save_dir).resolve() != output:
                raise ValueError("Native evaluation output must stay inside its own run")

        model.add_callback("on_val_start", ready)
        metrics = model.val(data=str(yaml), split="test", imgsz=config["imgsz"],
                            batch=config["batch"], device=config["device"],
                            project=str(root), name="evaluation", exist_ok=True,
                            conf=0.001, iou=0.7, max_det=300, augment=False, single_cls=False,
                            classes=None, agnostic_nms=False, overlap_mask=False, rect=True,
                            save_json=False, save_txt=False, plots=False, half=False,
                            seed=config["seed"], deterministic=config["deterministic"],
                            workers=config["training_options"].get("workers", 4),
                            mask_ratio=config["training_options"].get("mask_ratio", 4))
        scores = {part + "_" + field: float(getattr(getattr(metrics, part), field))
                  for part in ("box", "seg") for field in ("map", "map50", "map75")}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError("All six AP metrics must be finite fractions in [0, 1]")
        row.update(scores)
    except Exception as error:
        row.update(status="failed", error_message=str(error) or type(error).__name__)
    upsert_summary(results, row)
    return row


def discover_runs(runs_root: Path, *, method: str = "baseline", shots: Sequence[int] = (1, 2, 3, 5),
                  include_smoke: bool = False) -> list[Path]:
    """Find every completed matching identity, ordered by shot/seed/hash.

    Invalid manifests fail discovery rather than silently hiding a run.
    This read-only seam is shared with the later visualization task.
    """
    if method not in ("baseline", "fgbg-triplet"):
        raise ValueError("Unsupported evaluation method")
    if (not shots or len(set(shots)) != len(shots)
            or any(type(shot) is not int or shot not in (1, 2, 3, 5) for shot in shots)):
        raise ValueError("shots must be a unique selection of 1, 2, 3, 5")
    root = _managed_path(Path(runs_root), Path(runs_root).absolute().parent)
    found = []
    for shot in sorted(shots):
        folder = root / f"yolo11n-seg/{method}/shot_{shot}"
        _managed_path(folder, root.parent)
        for path in sorted(folder.glob("seed_*/*/manifest.json")):
            directory, manifest, _ = _read_run(path.parent)
            if (Path(manifest["runs_root"]).resolve() != root
                    or manifest["method"] != method or manifest["shot"] != shot):
                raise ValueError("Discovered run manifest disagrees with selection/path")
            if manifest["status"] == "completed" and (include_smoke or manifest["run_kind"] == "benchmark"):
                found.append(directory)
    return found


def evaluate_selected(shots: Sequence[int] | None, paths: DatasetPaths, *, method: str | None = None,
                      run_dir: Path | None = None, include_smoke: bool = False,
                      continue_on_error: bool = False, runtime=None) -> list[dict]:
    """Evaluate a single owned run or all completed matching hashes sequentially."""
    paths = DatasetPaths.from_root(paths.data_root.resolve(), paths.work_root.resolve())
    if run_dir is not None:
        if shots is not None:
            raise ValueError("Select either run_dir or shots")
        root, manifest, _ = _read_run(run_dir)
        if not root.is_relative_to(paths.runs_root.resolve()):
            raise ValueError("Selected run must belong to this work root")
        if method is not None and method != manifest["method"]:
            raise ValueError("Selected method disagrees with the run manifest")
        if not include_smoke and manifest["run_kind"] == "smoke":
            raise ValueError("Smoke runs are excluded; opt in explicitly with --include-smoke")
        roots = [root]
    else:
        roots = discover_runs(paths.runs_root, method=method or "baseline", shots=shots,
                              include_smoke=include_smoke)
    if not roots:
        raise ValueError("No completed matching runs found; smoke runs are excluded by default")
    outcomes = []
    for root in roots:
        _, manifest, _ = _read_run(root)
        yaml = paths.prepared_root / f"shot_{manifest['shot']}/data.yaml"
        row = evaluate_one(root, yaml, paths.results_root, runtime=runtime)
        outcomes.append(row)
        if row["status"] == "failed" and not continue_on_error:
            break
    return outcomes
