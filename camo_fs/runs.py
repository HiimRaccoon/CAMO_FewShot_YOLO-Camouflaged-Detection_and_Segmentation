"""Content-based experiment identity, owned-run recovery and atomic reporting.

This module performs no training and imports no Torch/Ultralytics runtime.
T06 must resolve weights/data hashes and effective training options before
constructing RunConfig, and supply audited preparation provenance.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence


METRIC_FIELDS = ("box_map", "box_map50", "box_map75", "seg_map", "seg_map50", "seg_map75")
IDENTITY_FIELDS = ("model", "method", "shot", "seed", "run_kind", "config_hash")
SUMMARY_FIELDS = (*IDENTITY_FIELDS, "epochs", "imgsz", "batch", "device", "weights_sha256",
                  "data_sha256", "triplet_weight", "triplet_margin", "triplets_per_instance",
                  *METRIC_FIELDS, "weights_path", "status", "error_message")
_TRANSITIONS = {"initialized": {"running", "failed"}, "running": {"completed", "failed"},
                "failed": {"running"}, "completed": set()}


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Project's proposed common preset; target-version forwarding is T06."""
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0
    mosaic: float = 0.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    close_mosaic: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "close_mosaic":
                _integer(value, name, minimum=0)
            else:
                number = _finite(value, name)
                if name in ("hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "mosaic", "mixup", "copy_paste") and not 0 <= number <= 1:
                    raise ValueError(name + " probability/fraction must be in range [0, 1]")
                object.__setattr__(self, name, number)


@dataclass(frozen=True, slots=True)
class RunConfig:
    weights_sha256: str
    data_sha256: str
    model: str = "yolo11n-seg"
    method: str = "baseline"
    shot: int = 1
    seed: int = 2024
    run_kind: str = "benchmark"
    weights: str = "yolo11n-seg.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    deterministic: bool = True
    val: bool = False
    overlap_mask: bool = False
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training_options: Mapping[str, Any] = field(default_factory=dict)
    triplet_weight: float | None = None
    triplet_margin: float | None = None
    triplets_per_instance: int | None = None

    def __post_init__(self) -> None:
        _checksum(self.weights_sha256)
        _checksum(self.data_sha256)
        if self.model != "yolo11n-seg" or self.method not in ("baseline", "fgbg-triplet"):
            raise ValueError("Unsupported model/method")
        if self.run_kind not in ("benchmark", "smoke"):
            raise ValueError("run_kind must be benchmark or smoke")
        _integer(self.shot, "shot")
        if self.shot not in (1, 2, 3, 5):
            raise ValueError("shot must be 1, 2, 3, 5")
        for name in ("epochs", "imgsz", "batch"):
            _integer(getattr(self, name), name)
        _integer(self.seed, "seed", minimum=0)
        if type(self.deterministic) is not bool or self.val is not False or self.overlap_mask is not False:
            raise ValueError("Training requires val=False and overlap_mask=False")
        if not isinstance(self.augmentation, AugmentationConfig):
            raise ValueError("augmentation must be an immutable AugmentationConfig")
        if not str(self.weights) or not str(self.device):
            raise ValueError("weights and device must be nonempty")
        object.__setattr__(self, "weights", str(self.weights))
        object.__setattr__(self, "device", str(self.device))
        options = dict(self.training_options)
        reserved = set(self.__dataclass_fields__) | set(asdict(self.augmentation)) | {"resume", "data"}
        for key, value in options.items():
            if not isinstance(key, str) or not key or key in reserved:
                raise ValueError("training_options cannot override named config fields")
            if type(value) not in (str, bool, int, float, type(None)):
                raise ValueError("training_options values must be immutable JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("training_options must be finite")
        object.__setattr__(self, "training_options", MappingProxyType(options))
        if self.method == "baseline":
            for name in ("triplet_weight", "triplet_margin", "triplets_per_instance"):
                object.__setattr__(self, name, None)
        else:
            weight = _finite(0.1 if self.triplet_weight is None else self.triplet_weight, "triplet_weight")
            margin = _finite(0.3 if self.triplet_margin is None else self.triplet_margin, "triplet_margin")
            count = 16 if self.triplets_per_instance is None else self.triplets_per_instance
            if weight < 0 or margin <= 0:
                raise ValueError("triplet weight/margin out of range")
            _integer(count, "triplets_per_instance")
            object.__setattr__(self, "triplet_weight", weight)
            object.__setattr__(self, "triplet_margin", margin)
            object.__setattr__(self, "triplets_per_instance", count)


def _config_record(config: RunConfig) -> dict:
    return {name: asdict(config.augmentation) if name == "augmentation"
            else dict(config.training_options) if name == "training_options"
            else getattr(config, name) for name in config.__dataclass_fields__}


def _identity(config: RunConfig) -> dict:
    result = _config_record(config)
    del result["weights"]  # Content is already represented by weights_sha256.
    return result


def fingerprint(config: RunConfig, weights_sha256: str, data_sha256: str) -> str:
    """Hash one canonical identity; contradictory external checksums are errors."""
    if (weights_sha256, data_sha256) != (config.weights_sha256, config.data_sha256):
        raise ValueError("Explicit checksum disagrees with bound RunConfig checksum")
    return hashlib.sha256(_canonical(_identity(config)).encode("utf-8")).hexdigest()


def run_path(config: RunConfig, runs_root: Path) -> Path:
    return Path(runs_root) / config.model / config.method / f"shot_{config.shot}" / f"seed_{config.seed}" / fingerprint(config, config.weights_sha256, config.data_sha256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared_data_sha256(shot_dir: Path) -> str:
    """Hash actual selected train/shared test data, independent of its location.

    This is a content fingerprint, not a replacement for T04's integrity audit.
    T04 emits JSON-compatible YAML; another YAML format requires an adapter.
    """
    shot = Path(shot_dir).resolve()
    root = shot.parent
    train_manifest = _read_dict(shot / "manifest.json")
    test_manifest = _read_dict(root / "test/manifest.json")
    yaml = _read_dict(shot / "data.yaml")
    if Path(yaml["train"]).resolve() != shot / "train/images" or Path(yaml["test"]).resolve() != root / "test/images":
        raise ValueError("Prepared YAML must reference own train and shared test")
    if "val" in yaml and Path(yaml["val"]).resolve() != shot / "train/images":
        raise ValueError("Prepared val must never reference official test")
    payload: dict[str, Any] = {"mapping": _read_dict(root / "category_mapping.json"),
                             "yaml": {"names": yaml["names"], "val_train_placeholder": "val" in yaml},
                             "files": {}, "sources": []}
    for label, directory in (("train", shot / "train"), ("test", root / "test")):
        for part in ("images", "labels"):
            if not (directory / part).is_dir():
                raise ValueError("Missing prepared image/label directory")
        _reject_links(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != directory / "manifest.json":
                payload["files"][label + "/" + path.relative_to(directory).as_posix()] = sha256_file(path)
    for manifest in (train_manifest, test_manifest):
        for source in manifest["source_jsons"]:
            _checksum(source["sha256"])
            payload["sources"].append((Path(source["path"]).name, source["sha256"]))
    payload["sources"] = sorted(set(payload["sources"]))
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def build_manifest(config: RunConfig, runs_root: Path, *, prepared_manifest: dict,
                   warnings: Sequence[str] = ()) -> dict:
    _validate_prepared_provenance(prepared_manifest, config)
    root = Path(runs_root).resolve()
    path = run_path(config, root)
    # Snapshot the caller's provenance so later edits cannot mutate the record.
    provenance = json.loads(_canonical(prepared_manifest))
    return {
        "schema_version": 1, "identity": _identity(config), "config": _config_record(config),
        **{name: getattr(config, name) for name in ("model", "method", "shot", "seed", "run_kind", "weights_sha256", "data_sha256")},
        "config_hash": fingerprint(config, config.weights_sha256, config.data_sha256),
        "weights_path": config.weights, "runs_root": str(root), "run_dir": str(path),
        "last_checkpoint": str(path / "weights/last.pt"), "status": "initialized", "error_message": "",
        "timestamp_utc": _now(), "updated_at_utc": _now(), "versions": _versions(),
        "prepared_dataset": provenance, "warnings": list(warnings),
        "auxiliary_loss": None if config.method == "baseline" else {
            "feature_source": "Segment head first spatial input (P3/8), unverified until T09",
            "triplet_weight": config.triplet_weight, "triplet_margin": config.triplet_margin,
            "triplets_per_instance": config.triplets_per_instance,
            "sampled_triplets": 0, "skipped_instances": 0,
        },
    }


def _verify_manifest(manifest: dict, config: RunConfig) -> Path:
    expected = _identity(config)
    recorded = dict(manifest.get("config", {}))
    recorded.pop("weights", None)
    if manifest.get("identity") != expected or recorded != expected:
        raise ValueError("Run identity/configuration mismatch")
    digest = fingerprint(config, config.weights_sha256, config.data_sha256)
    for key in (*IDENTITY_FIELDS, "weights_sha256", "data_sha256"):
        value = digest if key == "config_hash" else getattr(config, key)
        if manifest.get(key) != value:
            raise ValueError("Run identity/checksum mismatch: " + key)
    _validate_prepared_provenance(manifest.get("prepared_dataset"), config)
    root = Path(manifest["runs_root"]).resolve()
    path = run_path(config, root)
    if Path(manifest["run_dir"]).resolve() != path:
        raise ValueError("Run directory disagrees with identity")
    _safe_output(path)
    if Path(manifest["last_checkpoint"]).absolute() != path / "weights/last.pt":
        raise ValueError("Resume must use run's own last checkpoint")
    return path


def verify_resume(manifest: dict, config: RunConfig, weights_sha256: str, data_sha256: str) -> None:
    fingerprint(config, weights_sha256, data_sha256)
    path = _verify_manifest(manifest, config)
    if manifest.get("status") not in ("running", "failed"):
        raise ValueError("Only an interrupted run may resume; initialized/completed runs cannot")
    checkpoint = path / "weights/last.pt"
    _safe_output(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("Run's own last checkpoint is missing")


def initialize_run(config: RunConfig, runs_root: Path, *, prepared_manifest: dict,
                   resume: bool = False, overwrite: bool = False, warnings: Sequence[str] = ()) -> dict:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    root = Path(runs_root).absolute()
    _safe_output(root)
    root = root.resolve()
    target = run_path(config, root)
    _safe_output(target)
    if target.exists():
        if not resume and not overwrite:
            raise ValueError("Run target exists; use explicit resume or overwrite")
        existing = _read_dict(target / "manifest.json")
        _verify_manifest(existing, config)
        if Path(existing["run_dir"]) != target or Path(existing["runs_root"]) != root:
            raise ValueError("Run manifest belongs to another directory")
        if resume:
            verify_resume(existing, config, config.weights_sha256, config.data_sha256)
            return existing
        _reject_links(target)
    elif resume:
        raise ValueError("Cannot resume: own run manifest/checkpoint is missing")
    manifest = build_manifest(config, root, prepared_manifest=prepared_manifest, warnings=warnings)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run-stage-", dir=target.parent))
    preserve = False
    try:
        fresh = stage / "fresh"
        fresh.mkdir()
        _write_json(fresh / "manifest.json", manifest)
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
                    raise ValueError(f"Run overwrite failed; recovery backup kept at {backup}: {recovery_error}") from error
            raise
    finally:
        if not preserve:
            _remove_owned(stage, root)
    return manifest


def transition_run(run_dir: Path, status: str, *, error_message: str = "") -> dict:
    path = Path(run_dir).absolute()
    _safe_output(path)
    manifest = _read_dict(path / "manifest.json")
    config = _from_record(manifest["config"])
    if _verify_manifest(manifest, config) != path.resolve():
        raise ValueError("Manifest belongs to another run")
    current = manifest.get("status")
    if status not in _TRANSITIONS.get(current, set()):
        raise ValueError(f"Invalid run state transition: {current} -> {status}")
    if status == "failed" and not error_message:
        raise ValueError("Failed runs require error_message")
    if current == "failed" and status == "running":
        verify_resume(manifest, config, config.weights_sha256, config.data_sha256)
    manifest.update(status=status, error_message=error_message if status == "failed" else "", updated_at_utc=_now())
    _atomic_json(path / "manifest.json", manifest)
    return manifest


def summary_row(manifest: dict, metrics: Mapping[str, float] | None = None) -> dict:
    config = _from_record(manifest["config"])
    _verify_manifest(manifest, config)
    row = {key: manifest[key] for key in IDENTITY_FIELDS}
    row.update({key: getattr(config, key) for key in ("epochs", "imgsz", "batch", "device", "weights_sha256", "data_sha256",
                "triplet_weight", "triplet_margin", "triplets_per_instance")})
    row.update(weights_path=manifest["weights_path"], status=manifest["status"], error_message=manifest["error_message"])
    if metrics:
        if set(metrics) - set(METRIC_FIELDS):
            raise ValueError("Unknown summary metric")
        row.update(metrics)
    return row


def upsert_summary(results_dir: Path, row: dict) -> None:
    """Atomically upsert a fixed-schema row by the complete run identity.

    Calls are sequential in this project's single-process orchestration;
    concurrent writers require a future lock rather than an atomic rename alone.
    """
    root = Path(results_dir).absolute()
    _safe_output(root)
    target = root / "summary.csv"
    _safe_output(target)
    if set(row) - set(SUMMARY_FIELDS):
        raise ValueError("Unknown summary column")
    if any(key not in row or row[key] in (None, "") for key in IDENTITY_FIELDS):
        raise ValueError("Summary requires complete run identity")
    _checksum(row["config_hash"])
    if row.get("status") not in _TRANSITIONS:
        raise ValueError("Invalid summary status")
    if row["status"] == "failed" and not row.get("error_message"):
        raise ValueError("Failed summary requires error_message")
    normalized = {key: "" if row.get(key) is None else str(row.get(key, "")) for key in SUMMARY_FIELDS}
    if row["status"] == "failed":
        normalized.update({key: "" for key in METRIC_FIELDS})
    key = tuple(normalized[name] for name in IDENTITY_FIELDS)
    rows: dict[tuple[str, ...], dict] = {}
    if target.exists():
        with target.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != SUMMARY_FIELDS:
                raise ValueError("Existing summary schema is incompatible")
            for entry in reader:
                identity = tuple(entry[name] for name in IDENTITY_FIELDS)
                if identity in rows or None in entry or any(value is None for value in entry.values()):
                    raise ValueError("Existing summary is malformed or has duplicate identities")
                rows[identity] = entry
    rows[key] = normalized
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".summary-stage-", dir=root) as temporary:
        staged = Path(temporary) / "summary.csv"
        with staged.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows[identity] for identity in sorted(rows))
        staged.replace(target)


def _from_record(record: dict) -> RunConfig:
    values = dict(record)
    values["augmentation"] = AugmentationConfig(**values["augmentation"])
    return RunConfig(**values)


def _validate_prepared_provenance(provenance: Any, config: RunConfig) -> None:
    """Validate T04 metadata without requiring root or timestamp equality.

    Arguments and file_checksums are snapshotted as supplied; validation of
    actual copied bytes remains T04's audit plus the resolved data fingerprint.
    """
    try:
        if not isinstance(provenance, dict):
            raise ValueError("prepared dataset must be an object")
        if type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1:
            raise ValueError("unsupported preparation schema_version")
        _integer(provenance["shot"], "shot")
        if provenance["shot"] != config.shot:
            raise ValueError("prepared shot does not match run shot")
        for name in ("image_count", "annotation_count", "generated_label_count", "label_file_count"):
            _integer(provenance[name], name)
        if (provenance["generated_label_count"] != provenance["annotation_count"]
                or provenance["label_file_count"] != provenance["image_count"]):
            raise ValueError("prepared counts disagree with one label file per image and one line per instance")
        taxonomy = provenance["taxonomy"]
        if not isinstance(taxonomy, dict):
            raise ValueError("taxonomy must be an object")
        names, mapping = taxonomy.get("names"), taxonomy.get("category_to_index")
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or not isinstance(mapping, dict) or len(mapping) != len(names)):
            raise ValueError("taxonomy requires names and a matching category mapping")
        if any(not isinstance(key, str) or str(int(key)) != key for key in mapping):
            raise ValueError("category mapping keys must be canonical integer strings")
        indices = [mapping[key] for key in sorted(mapping, key=int)]
        if any(type(index) is not int for index in indices) or indices != list(range(len(names))):
            raise ValueError("taxonomy indices must be sorted and contiguous")
        stamp = provenance["timestamp_utc"]
        if not isinstance(stamp, str) or datetime.fromisoformat(stamp.replace("Z", "+00:00")).utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must be an aware UTC timestamp")
        versions = provenance["versions"]
        if (not isinstance(versions, dict) or not isinstance(versions.get("python"), str)
                or not versions["python"].strip()):
            raise ValueError("versions requires a Python version")
        for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pycocotools"):
            if name not in versions or (versions[name] is not None and (not isinstance(versions[name], str) or not versions[name].strip())):
                raise ValueError("versions must record installed version or null: " + name)
        for name in ("category_mapping_path", "output_path"):
            if not isinstance(provenance[name], str) or not provenance[name].strip():
                raise ValueError(name + " must be nonempty")
        sources = provenance["source_jsons"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("source_jsons must be a nonempty list")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str) or not source["path"].strip():
                raise ValueError("source path must be nonempty")
            _checksum(source["sha256"])
    except (KeyError, ValueError) as error:
        raise ValueError("Invalid prepared dataset provenance: " + str(error)) from error


def _safe_output(path: Path) -> None:
    absolute = path.absolute()
    if absolute.resolve().is_relative_to(Path("/kaggle/input").resolve()):
        raise ValueError("Outputs must not be under /kaggle/input")
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
            raise ValueError("Output symlinks/junctions are unsupported")


def _reject_links(path: Path) -> None:
    _safe_output(path)
    for child in path.rglob("*"):
        if child.is_symlink() or getattr(child, "is_junction", lambda: False)():
            raise ValueError("Managed artifacts must not contain links")


def _remove_owned(path: Path, root: Path) -> None:
    _reject_links(path)
    resolved = path.resolve()
    if resolved == root.resolve() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Refusing unscoped cleanup")
    shutil.rmtree(path)


def _atomic_json(path: Path, value: dict) -> None:
    _safe_output(path)
    with tempfile.TemporaryDirectory(prefix=".manifest-stage-", dir=path.parent) as temporary:
        staged = Path(temporary) / "manifest.json"
        _write_json(staged, value)
        staged.replace(path)


def _read_dict(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as source:
            result = json.load(source)
    except (OSError, ValueError) as error:
        raise ValueError("Missing or malformed manifest/artifact: " + str(path)) from error
    if not isinstance(result, dict):
        raise ValueError("Manifest/artifact must be a JSON object")
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8", newline="\n")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _checksum(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("checksum must be a lowercase SHA-256 hex digest")


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer within range")


def _finite(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return float(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _versions() -> dict:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pycocotools"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result
