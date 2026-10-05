"""Sequential baseline/enhanced training; runtime imports stay at the YOLO seam."""

from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

from camo_fs.paths import DatasetPaths
from camo_fs.prepare import read_prepared_yaml, verify_prepared
from camo_fs.runs import (RunConfig, SUMMARY_FIELDS, fingerprint, initialize_run, prepared_data_sha256, run_path,
                          sha256_file, summary_row, transition_run, upsert_summary)


_EXTRA_OPTIONS = {"optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                  "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl", "mask_ratio",
                  "nbs", "workers", "amp", "cos_lr", "plots", "verbose", "cls_pw", "cls_remap",
                  "dropout", "compile", "channels_last", "copy_paste_mode", "auto_augment", "erasing",
                  "bgr", "cutmix", "multi_scale", "fraction", "single_cls", "rect", "freeze",
                  "distill_model", "quantize", "profile"}
_LOCKED_OPTIONS = {"bgr": 0.0, "cutmix": 0.0, "multi_scale": False, "fraction": 1.0,
                   "single_cls": False, "rect": False, "freeze": None, "distill_model": None,
                   "quantize": None, "profile": False, "compile": False, "erasing": 0.0}


class UltralyticsRuntime:
    """Lazy native adapter; enhanced bindings are imported only on dispatch.

    Base checks verify compatibility with the packaged YOLO11n segmentation
    architecture and COCO class names. They do not certify an upstream release
    digest; the user-supplied local checkpoint's actual SHA-256 binds the run.
    """

    def __init__(self):
        try:
            import ultralytics
            import numpy
            import torch
            from ultralytics.cfg import get_cfg
            from ultralytics.nn.modules import Segment
            from ultralytics.utils import YAML
            from ultralytics.utils.torch_utils import init_seeds
        except (ImportError, AttributeError) as error:
            raise ValueError("Install compatible Ultralytics, PyTorch and NumPy on Kaggle before training: " + str(error)) from error
        self.yolo, self.numpy, self.torch = ultralytics.YOLO, numpy, torch
        self.get_cfg, self.segment, self.yaml, self.init_seeds = get_cfg, Segment, YAML, init_seeds
        self.package_root = Path(ultralytics.__file__).parent

    def seed(self, seed: int, deterministic: bool) -> list[str]:
        random.seed(seed)
        self.numpy.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.init_seeds(seed, deterministic=deterministic)
        return ["Determinism requested; full CUDA/library determinism cannot be guaranteed."]

    def validate_options(self, options: dict) -> None:
        self.get_cfg(overrides=options)

    def enhanced_trainer(self, config: RunConfig):
        """Bind custom settings outside native overrides; import only for enhanced."""
        from camo_fs.ultralytics_ext import FGSegmentationTrainer

        return partial(FGSegmentationTrainer, triplet_weight=config.triplet_weight,
                       triplet_margin=config.triplet_margin,
                       triplets_per_instance=config.triplets_per_instance)

    def training_defaults(self) -> dict:
        """Resolve installed native train/segmentation settings before hashing."""
        defaults = vars(self.get_cfg(overrides={}))
        return {key: _LOCKED_OPTIONS.get(key, value) for key, value in defaults.items() if key in _EXTRA_OPTIONS}

    def load(self, checkpoint: Path, *, base: bool):
        model = self.yolo(str(checkpoint))
        if model.task != "segment":
            raise ValueError("Checkpoint must be a YOLO11n segmentation model")
        if base:
            expected = self.yaml.load(self.package_root / "cfg/models/11/yolo11-seg.yaml")
            config = model.model.yaml
            names = self.yaml.load(self.package_root / "cfg/datasets/coco.yaml")["names"]
            if (config.get("scale") != "n" or config.get("nc") != 80 or model.names != names
                    or any(config.get(key) != expected[key] for key in ("backbone", "head"))
                    or not isinstance(model.model.model[-1], self.segment)):
                raise ValueError("Base checkpoint must match COCO YOLO11n-Seg architecture/classes")
        elif (type(model.ckpt.get("epoch")) is not int or model.ckpt["epoch"] < 0
              or model.ckpt.get("optimizer") is None):
            raise ValueError("Own resume checkpoint lacks interrupted epoch/optimizer state")
        return model

    def load_inference(self, checkpoint: Path):
        """Shared native inference path for completed, possibly stripped last.pt.

        Evaluation/visualization must use this path rather than resume loading,
        since completed native checkpoints have no epoch/optimizer resume state.
        """
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise ValueError("Local .pt inference checkpoint unavailable")
        model = self.yolo(str(checkpoint))
        if model.task != "segment":
            raise ValueError("Inference checkpoint must be a segmentation model")
        return model


@dataclass(frozen=True)
class TrainingOutcome:
    shot: int
    status: str
    config: RunConfig | None = None
    last_pt: Path | None = None
    error_message: str = ""


def train_selected(shots: Sequence[int], paths: DatasetPaths, *, weights: str = "yolo11n-seg.pt",
                   settings: Mapping[str, Any] | None = None, resume: bool = False,
                   overwrite: bool = False, continue_on_error: bool = False, runtime=None) -> list[TrainingOutcome]:
    """Resolve each identity then train sequentially, recording every attempted shot.

    If resolution fails (e.g. preparation is absent), no data hash can be invented:
    training_attempts.json records that failure with a null identity, alongside the
    identity-keyed CSV rows that train_one writes for resolved configurations.
    """
    if not shots or any(type(shot) is not int or shot not in (1, 2, 3, 5) for shot in shots) or len(set(shots)) != len(shots):
        raise ValueError("shots must be a unique selection of 1, 2, 3, 5")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    paths = DatasetPaths.from_root(paths.data_root.resolve(), paths.work_root.resolve())
    outcomes: list[TrainingOutcome] = []
    for shot in shots:
        config = None
        try:
            checkpoint = Path(weights)
            if not checkpoint.is_file() or checkpoint.suffix != ".pt":
                raise ValueError("Local .pt base checkpoint unavailable; attach/download yolo11n-seg.pt before training")
            shot_dir = paths.prepared_root / f"shot_{shot}"
            _verify_training_data(read_prepared_yaml(shot_dir / "data.yaml"), shot_dir)
            data_digest = prepared_data_sha256(shot_dir)
            actual_runtime = runtime if runtime is not None else UltralyticsRuntime()
            config = resolve_config(shot, checkpoint, data_digest, settings or {}, actual_runtime)
            last = train_one(config, paths, resume, overwrite, runtime=actual_runtime)
            outcomes.append(TrainingOutcome(shot, "completed", config, last))
        except Exception as error:
            outcomes.append(TrainingOutcome(shot, "failed", config, error_message=str(error)))
        _write_attempts(paths, outcomes)
        if outcomes[-1].status == "failed" and not continue_on_error:
            break
    return outcomes


def resolve_config(shot: int, weights: Path, data_sha256: str, settings: Mapping[str, Any], runtime) -> RunConfig:
    """Bind the installed native defaults and explicit overrides to immutable identity."""
    values = dict(settings)
    extras = dict(values.pop("training_options", {}))
    if set(extras) - _EXTRA_OPTIONS:
        raise ValueError("Extra training option cannot override project protocol")
    if extras.get("optimizer") == "auto":
        raise ValueError("Choose an explicit optimizer; auto rewrites effective LR/warmup settings")
    values["training_options"] = {**runtime.training_defaults(), **extras}
    # Resolve native setup transformations before identity, rather than accepting
    # a later mismatch or letting data-dependent auto optimizer heuristics vary.
    if values["training_options"].get("optimizer") == "auto":
        values["training_options"]["optimizer"] = "SGD"
    if str(values.get("device", "0")) == "cpu" and "workers" in values["training_options"]:
        values["training_options"]["workers"] = 0
    return RunConfig(**values, shot=shot, weights=str(weights), weights_sha256=sha256_file(weights),
                     data_sha256=data_sha256)


def _write_attempts(paths: DatasetPaths, outcomes: Sequence[TrainingOutcome]) -> None:
    report = paths.results_root / "training_attempts.json"
    _plain_path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    records = [{"shot": item.shot, "status": item.status, "error_message": item.error_message,
                "config_hash": fingerprint(item.config, item.config.weights_sha256, item.config.data_sha256) if item.config else None,
                "last_pt": str(item.last_pt) if item.last_pt else None} for item in outcomes]
    with tempfile.TemporaryDirectory(prefix=".attempt-stage-", dir=report.parent) as temporary:
        staged = Path(temporary) / "training_attempts.json"
        staged.write_text(json.dumps(records, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        staged.replace(report)


def _plain_path(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("Training artifacts must not contain symlinks/junctions")


def training_options(config: RunConfig, paths: DatasetPaths, *, resume: bool = False) -> dict:
    """Common non-method options, with explicit project protocol and run routing."""
    if set(config.training_options) - _EXTRA_OPTIONS:
        raise ValueError("Extra training option cannot override project protocol")
    if config.training_options.get("optimizer") == "auto":
        raise ValueError("Project protocol requires an explicit optimizer")
    if any(value != _LOCKED_OPTIONS[key] for key, value in config.training_options.items() if key in _LOCKED_OPTIONS):
        raise ValueError("Resolved native option cannot override project protocol")
    augmentation = asdict(config.augmentation)
    if any(augmentation[name] != 0 for name in ("degrees", "translate", "scale", "shear", "perspective",
                                               "flipud", "mosaic", "mixup", "copy_paste", "close_mosaic")):
        raise ValueError("Project protocol requires unverified geometric/composition options disabled")
    if not config.deterministic or config.device not in ("0", "cpu"):
        raise ValueError("Project protocol requests determinism and a single device (0 or cpu)")
    root = run_path(config, paths.runs_root)
    return {**dict(config.training_options), **augmentation,
            "epochs": config.epochs, "imgsz": config.imgsz, "batch": config.batch,
            "device": config.device, "seed": config.seed, "deterministic": config.deterministic,
            "val": False, "overlap_mask": False, "patience": 0, "time": None,
            "save": True, "split": "val", "pretrained": True, "cache": False, "project": str(root.parent),
            "name": root.name, "exist_ok": True,
            "data": str(paths.prepared_root / f"shot_{config.shot}/data.yaml"),
            "resume": str(root / "weights/last.pt") if resume else False}


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _record(root: Path, **updates) -> dict:
    manifest = _read(root / "manifest.json")
    manifest.update(updates)
    with tempfile.TemporaryDirectory(prefix=".training-stage-", dir=root) as temporary:
        staged = Path(temporary) / "manifest.json"
        staged.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        staged.replace(root / "manifest.json")
    return manifest


def train_one(config: RunConfig, paths: DatasetPaths, resume: bool, overwrite: bool,
              *, runtime=None) -> Path:
    """Return the run's own last.pt after native training and recorded completion."""
    paths = DatasetPaths.from_root(paths.data_root.resolve(), paths.work_root.resolve())
    root = run_path(config, paths.runs_root)
    owned = False
    try:
        options = training_options(config, paths, resume=resume)
        if config.method == "fgbg-triplet" and config.device != "0":
            raise ValueError("Enhanced training requires single GPU device=0")
        weights = Path(config.weights)
        if not weights.is_file() or sha256_file(weights) != config.weights_sha256:
            raise ValueError("Missing base checkpoint or weights checksum mismatch")
        shot = paths.prepared_root / f"shot_{config.shot}"
        _verify_training_data(read_prepared_yaml(shot / "data.yaml"), shot)
        if prepared_data_sha256(shot) != config.data_sha256:
            raise ValueError("Prepared data checksum mismatch")
        provenance = verify_prepared(config.shot, paths)
        manifest = initialize_run(config, paths.runs_root, prepared_manifest=provenance,
                                  resume=resume, overwrite=overwrite)
        owned = True
        runtime = runtime if runtime is not None else UltralyticsRuntime()
        if set(runtime.training_defaults()) - set(config.training_options):
            raise ValueError("Resolve native defaults with resolve_config before constructing run identity")
        runtime.validate_options(options)
        previous_last = sha256_file(root / "weights/last.pt") if resume else None
        warnings = [*manifest["warnings"], *runtime.seed(config.seed, config.deterministic)]
        _record(root, warnings=warnings, requested_training_options=options)
        model = runtime.load(root / "weights/last.pt" if resume else weights, base=not resume)
        if manifest["status"] != "running":
            transition_run(root, "running")
        previous_options = manifest.get("effective_training_options") if resume else None

        def ready(trainer):
            _verify_trainer(trainer, options, root, provenance["taxonomy"]["names"])
            if previous_options is not None:
                effective = vars(trainer.args)
                keys = (set(previous_options) | set(effective)) - {"model", "resume"}
                for key in keys:
                    if previous_options.get(key) != effective.get(key):
                        raise ValueError("Resume effective defaults/configuration changed: " + key)
            _record(root, effective_training_options=vars(trainer.args))

        model.add_callback("on_pretrain_routine_end", ready)
        model.add_callback("on_train_epoch_start", lambda trainer: _verify_trainer(trainer, options, root, provenance["taxonomy"]["names"]))
        if config.method == "fgbg-triplet":
            import torch

            auxiliary = {**manifest["auxiliary_loss"], "feature_source": "Segment head first spatial input (P3/8)"}
            totals = dict(manifest.get("triplet_training", {"batches": 0, "nonzero_raw_batches": 0,
                                                          "sampled_triplets": 0, "skipped_instances": 0}))
            _record(root, auxiliary_loss=auxiliary, triplet_training=totals,
                    versions={**manifest["versions"], "cuda": torch.version.cuda})

            def log_triplet(trainer):
                metrics = dict(trainer.model.triplet_metrics)
                losses = [float(value) for value in trainer.loss_items]
                combined = float(trainer.loss.detach()) if hasattr(trainer.loss, "detach") else float(trainer.loss)
                if (len(losses) != 4 or not all(math.isfinite(value) for value in
                        [*losses, combined, metrics["raw_triplet"], metrics["weighted_triplet"]])):
                    raise FloatingPointError("Non-finite enhanced training loss")
                for key in ("sampled_triplets", "skipped_instances"):
                    if type(metrics[key]) is not int or metrics[key] < 0:
                        raise ValueError("Invalid enhanced training count: " + key)
                    totals[key] += metrics[key]
                totals["batches"] += 1
                totals["nonzero_raw_batches"] += int(metrics["raw_triplet"] > 0)
                auxiliary.update(sampled_triplets=totals["sampled_triplets"],
                                 skipped_instances=totals["skipped_instances"])
                record = {"epoch": trainer.epoch + 1, "batch": totals["batches"],
                          "native_losses": losses, "combined_loss": combined, **metrics}
                log = root / "triplet_batches.jsonl"
                _plain_path(log)
                with log.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                _record(root, triplet_training=totals, auxiliary_loss=auxiliary)
                print("triplet: " + json.dumps(record, sort_keys=True, allow_nan=False))

            model.add_callback("on_train_batch_end", log_triplet)
            model.train(trainer=runtime.enhanced_trainer(config), **options)
        else:
            model.train(**options)
        _verify_trainer(model.trainer, options, root, provenance["taxonomy"]["names"])
        last = root / "weights/last.pt"
        if not last.is_file() or not last.stat().st_size:
            raise ValueError("Training did not save own last.pt")
        last_digest = sha256_file(last)
        if previous_last is not None and previous_last == last_digest:
            raise ValueError("Resume did not save a new last.pt; stale checkpoint remains")
        if model.trainer.epoch + 1 != config.epochs or model.trainer.epochs != config.epochs:
            raise ValueError("Training did not complete requested fixed epochs")
        _record(root, last_checkpoint_sha256=last_digest, completed_epochs=config.epochs)
    except (Exception, KeyboardInterrupt) as error:
        message = str(error) or type(error).__name__
        if owned:
            if _read(root / "manifest.json")["status"] == "failed":
                failed = _record(root, error_message=message)
            else:
                failed = transition_run(root, "failed", error_message=message)
            upsert_summary(paths.results_root, summary_row(failed))
        elif not root.exists():
            row = {key: getattr(config, key) for key in SUMMARY_FIELDS if hasattr(config, key)}
            row.update(config_hash=fingerprint(config, config.weights_sha256, config.data_sha256),
                       weights_path=config.weights, status="failed", error_message=message)
            upsert_summary(paths.results_root, row)
        raise
    manifest = transition_run(root, "completed")
    upsert_summary(paths.results_root, summary_row(manifest))
    return last


def _verify_training_data(data: Mapping[str, Any], shot: Path) -> None:
    """Require a TRAIN val placeholder before native loaders can fall back to test."""
    train = shot / "train/images"
    val = data.get("val")
    if not isinstance(val, (str, Path)) or not str(val).strip():
        raise ValueError("Training requires val to be the TRAIN placeholder; missing or empty val "
                         "would make Ultralytics fall back to official test. "
                         "Prepare with --val-train-placeholder")
    for key, expected, message in (
        ("train", train, "Native trainer must use own prepared train data"),
        ("val", train, "Native validation data must only be a train placeholder"),
        ("test", shot.parent / "test/images", "Training data must reference shared official test"),
    ):
        value = data.get(key)
        if not isinstance(value, (str, Path)) or Path(value).resolve() != expected.resolve():
            raise ValueError(message)


def _verify_trainer(trainer, options: dict, root: Path, expected_names: Sequence[str]) -> None:
    _plain_path(root / "weights/last.pt")
    channels = trainer.data.get("channels", 3)
    if type(channels) is not int or channels != 3:
        raise ValueError("Native trainer requires RGB data with 3 channels")
    names = trainer.data.get("names")
    if isinstance(names, dict):
        matching_names = (all(type(key) is int for key in names)
                          and names == dict(enumerate(expected_names)))
    else:
        matching_names = isinstance(names, list) and names == list(expected_names)
    nc = trainer.data.get("nc")
    if not matching_names or type(nc) is not int or nc != len(expected_names):
        raise ValueError("Native trainer taxonomy/classes must match audited preparation")
    effective = vars(trainer.args)
    for key, expected in options.items():
        actual = effective.get(key)
        if key in ("data", "project"):
            matches = isinstance(actual, (str, Path)) and Path(actual).resolve() == Path(expected).resolve()
        elif key == "device":
            matches = str(actual) == str(expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError("Incompatible effective trainer option: " + key)
    if Path(trainer.save_dir).resolve() != root.resolve() or Path(trainer.last).resolve() != (root / "weights/last.pt").resolve():
        raise ValueError("Native trainer must save own run and last.pt")
    _verify_training_data(trainer.data, Path(options["data"]).parent)
