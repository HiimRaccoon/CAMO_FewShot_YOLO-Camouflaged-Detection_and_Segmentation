"""Audit-first preparation of official CAMO-FS segmentation datasets.

Source trees are read-only. Fail-fast audits every selected split before any
prepared artifact is staged. Promotion happens only after complete staging;
overwrite backs up affected targets and rolls back a failed promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path, PureWindowsPath
import platform
import shutil
import tempfile
from typing import Any

from camo_fs.annotations import AuditIssue, AuditReport, Taxonomy, audit_shot, audit_test
from camo_fs.paths import DatasetPaths
from camo_fs.segments import DataIntegrityError, annotation_to_yolo


class PreparationError(DataIntegrityError):
    """Preparation failed; contextual evidence is written to results/audit.json."""


class _RecoveryError(PreparationError):
    """Rollback could not finish; keep the staging directory's original backups."""


@dataclass(frozen=True, slots=True)
class PreparedShot:
    shot: int
    status: str
    data_yaml: Path | None = None
    manifest_path: Path | None = None
    error_message: str = ""


@dataclass(slots=True)
class _Split:
    report: AuditReport
    labels: dict[str, str] = field(default_factory=dict)
    multi_polygon_instances: list[dict[str, Any]] = field(default_factory=list)
    status: str = "audited"
    error_message: str = ""


def prepare_selected(
    shots: list[int],
    paths: DatasetPaths,
    overwrite: bool,
    continue_on_error: bool,
    *,
    val_train_placeholder: bool = False,
) -> list[PreparedShot]:
    """Prepare selected official shots, or record failures and stop.

    A failed shared test always raises. With continue_on_error, failed shots
    return status='failed' alongside successful shots. No variant trains here.
    val_train_placeholder is an explicit parser-compatibility option; training
    must still use val=False and verify its target version in T06.
    """
    if not shots or any(type(shot) is not int or shot not in (1, 2, 3, 5) for shot in shots):
        raise PreparationError("shots must be a non-empty selection of 1, 2, 3, 5")
    if len(set(shots)) != len(shots):
        raise PreparationError("shots must not contain duplicates")
    # Resolve again at the write boundary, including symlinked input/work roots.
    paths = DatasetPaths.from_root(paths.data_root.resolve(), paths.work_root.resolve())
    _check_output(paths.prepared_root, paths)
    _check_output(paths.results_root / "audit.json", paths)
    arguments = {
        "shots": shots, "data_root": str(paths.data_root), "work_root": str(paths.work_root),
        "overwrite": overwrite, "continue_on_error": continue_on_error,
        "val_train_placeholder": val_train_placeholder,
    }
    try:
        taxonomy = Taxonomy.from_test_json(paths.test_json)
    except (OSError, ValueError) as error:
        test = _Split(AuditReport(shot=0, source_files=[paths.test_json], errors=[
            AuditIssue("invalid_test_taxonomy", str(error), filename=paths.test_json.name)]))
        selected = [_Split(AuditReport(shot=shot)) for shot in shots]
        _block(test, selected, "shared test audit failed: " + str(error))
        _write_audit(paths, test, selected, arguments)
        raise PreparationError(test.error_message) from error

    test = _inspect(0, paths, taxonomy)
    selected: list[_Split] = []
    if test.report.errors:
        selected = [_Split(AuditReport(shot=shot)) for shot in shots]
        _block(test, selected, "shared test audit failed: " + _errors(test))
        _write_audit(paths, test, selected, arguments)
        raise PreparationError(test.error_message)

    # Shared artifacts are checked before any shot can be materialized.
    try:
        _check_target(paths.prepared_root / "test", paths)
        mapping_path = paths.prepared_root / "category_mapping.json"
        _check_target(mapping_path, paths)
        if overwrite:
            # Rebuilding shared mapping/test must not invalidate retained shots.
            selected_names = {f"shot_{shot}" for shot in shots}
            for retained in paths.prepared_root.glob("shot_*"):
                if retained.name not in selected_names:
                    _check_target(retained, paths)
                    try:
                        compatible = _read_json(retained / "manifest.json").get("taxonomy") == _mapping(taxonomy)
                    except (OSError, ValueError, AttributeError):
                        compatible = False
                    if not compatible:
                        raise PreparationError("Cannot overwrite shared mapping: unselected shot taxonomy is incompatible: " + str(retained))
        elif mapping_path.exists() and _read_json(mapping_path) != _mapping(taxonomy):
            raise PreparationError("Existing category mapping differs from canonical taxonomy; use --overwrite")
        reuse_test = (paths.prepared_root / "test").exists() and not overwrite
        if reuse_test:
            _verify_shared_test(test, taxonomy, paths)
            test.status = "reused"
    except (OSError, ValueError) as error:
        test.report.errors.append(AuditIssue("shared_target_error", str(error)))
        selected = [_Split(AuditReport(shot=shot)) for shot in shots]
        _block(test, selected, "shared test target failed: " + str(error))
        _write_audit(paths, test, selected, arguments)
        raise PreparationError(test.error_message) from error

    if not continue_on_error:
        selected = [_inspect(shot, paths, taxonomy) for shot in shots]
        for split in selected:
            _inspect_target(split, paths, overwrite)
        if any(split.report.errors for split in selected):
            message = "selected shot audit failed: " + "; ".join(_errors(split) for split in selected if split.report.errors)
            for split in selected:
                split.status = "failed" if split.report.errors else "blocked"
                split.error_message = _errors(split) if split.report.errors else message
            _write_audit(paths, test, selected, arguments)
            raise PreparationError(message)
        try:
            _materialize(selected, test, taxonomy, paths, arguments, reuse_test)
        except (OSError, ValueError) as error:
            for split in selected:
                split.status = "failed"
                split.error_message = str(error)
                split.report.errors.append(AuditIssue("materialization_error", str(error)))
            _write_audit(paths, test, selected, arguments)
            raise PreparationError("materialization failed: " + str(error)) from error
    else:
        for shot in shots:
            split = _inspect(shot, paths, taxonomy)
            selected.append(split)
            _inspect_target(split, paths, overwrite)
            if split.report.errors:
                split.status = "failed"
                split.error_message = _errors(split)
            else:
                try:
                    _materialize([split], test, taxonomy, paths, arguments, reuse_test)
                    reuse_test = True  # Common test is promoted only once, even with overwrite.
                except (OSError, ValueError) as error:
                    split.status = "failed"
                    split.error_message = str(error)
                    split.report.errors.append(AuditIssue("materialization_error", str(error)))
            _write_audit(paths, test, selected, arguments)
    _write_audit(paths, test, selected, arguments)
    return [PreparedShot(
        shot=split.report.shot, status=split.status,
        data_yaml=paths.prepared_root / f"shot_{split.report.shot}/data.yaml" if split.status == "prepared" else None,
        manifest_path=paths.prepared_root / f"shot_{split.report.shot}/manifest.json" if split.status == "prepared" else None,
        error_message=split.error_message,
    ) for split in selected]


def _inspect(shot: int, paths: DatasetPaths, taxonomy: Taxonomy) -> _Split:
    try:
        report = audit_test(paths, taxonomy) if shot == 0 else audit_shot(shot, paths, taxonomy)
    except (OSError, ValueError) as error:
        report = AuditReport(shot=shot, errors=[AuditIssue("source_read_error", str(error))])
    split = _Split(report)
    if not report.images or not report.annotations:
        report.errors.append(AuditIssue("empty_split", "Split must contain images and annotated instances"))
    lines: dict[str, list[str]] = {}
    label_owners: dict[str, str] = {}
    for image_id, image in report.images.items():
        filename = image["file_name"]
        try:
            relative = _source_relative(filename, paths)
            label = relative.with_suffix(".txt").as_posix()
            previous = label_owners.setdefault(label.casefold(), filename)
            if previous != filename:
                raise PreparationError("Different image names produce the same label path")
            lines.setdefault(label, [])
        except ValueError as error:
            code = "label_filename_collision" if "same label" in str(error) else "unsafe_image_filename"
            report.errors.append(AuditIssue(code, str(error), image_id, filename=filename))
    for annotation in report.annotations:
        image = report.images.get(annotation["image_id"])
        if image is None:
            continue  # Existing metadata error already blocks this split.
        filename = image["file_name"]
        try:
            label = _source_relative(filename, paths).with_suffix(".txt").as_posix()
            line, multi = annotation_to_yolo(annotation, image, taxonomy.category_to_index)
            lines.setdefault(label, []).append(line)
            if multi:
                split.multi_polygon_instances.append({"image_id": image["id"], "annotation_id": annotation["id"], "filename": filename})
        except ValueError as error:
            report.errors.append(AuditIssue("invalid_annotation_geometry", str(error), annotation["image_id"], annotation["id"], filename))
    split.labels = {name: "\n".join(values) + ("\n" if values else "") for name, values in lines.items()}
    return split


def _source_relative(filename: str, paths: DatasetPaths) -> Path:
    relative = Path(filename)
    if (relative.is_absolute() or PureWindowsPath(filename).drive or "\\" in filename
            or any(part in ("..", ".") for part in filename.split("/"))):
        raise PreparationError("Image filename must be a safe relative path: " + filename)
    if relative.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise PreparationError("Unsupported image filename: " + filename)
    source = (paths.images_dir / relative).resolve()
    if not source.is_relative_to(paths.images_dir.resolve()):
        raise PreparationError("Image path escapes the source images directory: " + filename)
    return relative


def _check_output(path: Path, paths: DatasetPaths) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(paths.work_root) or resolved == paths.work_root:
        raise PreparationError("Output must remain within work_root: " + str(path))
    if resolved.is_relative_to(paths.data_root) or resolved.is_relative_to(Path("/kaggle/input").resolve()):
        raise PreparationError("Output must not be inside dataset input: " + str(path))
    for ancestor in (path, *path.parents):
        if ancestor == paths.work_root:
            break
        if _is_link(ancestor):
            raise PreparationError("Output symlink/junction is unsupported: " + str(ancestor))


def _is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _check_target(target: Path, paths: DatasetPaths) -> None:
    _check_output(target, paths)
    if target.is_dir() and any(_is_link(child) for child in target.rglob("*")):
        raise PreparationError("Target contains symlinks/junctions: " + str(target))


def _inspect_target(split: _Split, paths: DatasetPaths, overwrite: bool) -> None:
    target = paths.prepared_root / f"shot_{split.report.shot}"
    try:
        _check_target(target, paths)
        if target.exists() and not overwrite:
            raise PreparationError("Selected target exists; use --overwrite: " + str(target))
    except (OSError, ValueError) as error:
        split.report.errors.append(AuditIssue("target_error", str(error)))


def _expected_files(split: _Split, paths: DatasetPaths) -> dict[str, str]:
    files = {"images/" + _source_relative(image["file_name"], paths).as_posix():
             _sha256(paths.images_dir / image["file_name"]) for image in split.report.images.values()}
    files.update({"labels/" + name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in split.labels.items()})
    return files


def _verify_shared_test(test: _Split, taxonomy: Taxonomy, paths: DatasetPaths) -> None:
    root = paths.prepared_root / "test"
    try:
        manifest = _read_json(root / "manifest.json")
        if manifest["source_jsons"] != _sources(test, paths):
            raise PreparationError("source JSON provenance changed")
        if manifest["taxonomy"] != _mapping(taxonomy):
            raise PreparationError("taxonomy changed")
        expected = _expected_files(test, paths)
        actual = {path.relative_to(root).as_posix(): _sha256(path) for path in root.rglob("*")
                  if path.is_file() and path != root / "manifest.json"}
        if expected != actual or manifest["file_checksums"] != expected:
            raise PreparationError("generated images/labels are stale or corrupt")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PreparationError("Cannot reuse shared test; use --overwrite: " + str(error)) from error


def _sources(split: _Split, paths: DatasetPaths) -> list[dict[str, str]]:
    sources = sorted(set([*split.report.source_files, paths.test_json]))
    return [{"path": str(path), "sha256": _sha256(path)} for path in sources]


def _materialize(
    selected: list[_Split], test: _Split, taxonomy: Taxonomy, paths: DatasetPaths,
    arguments: dict, reuse_test: bool,
) -> None:
    paths.prepared_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare-stage-", dir=paths.prepared_root))
    preserve_stage = False
    try:
        entries: list[tuple[Path, Path]] = []
        if not reuse_test:
            test_stage = stage / "test"
            _write_split(test_stage, test_stage, test, taxonomy, paths, arguments)
            entries.append((test_stage, paths.prepared_root / "test"))
        mapping_target = paths.prepared_root / "category_mapping.json"
        if not mapping_target.exists() or arguments["overwrite"]:
            _write_json(stage / "category_mapping.json", _mapping(taxonomy))
            entries.append((stage / "category_mapping.json", mapping_target))
        for split in selected:
            name = f"shot_{split.report.shot}"
            shot_stage = stage / name
            _write_split(shot_stage / "train", shot_stage, split, taxonomy, paths, arguments)
            yaml = {
                "train": str(paths.prepared_root / name / "train/images"),
                "test": str(paths.prepared_root / "test/images"), "names": taxonomy.names,
            }
            if arguments["val_train_placeholder"]:
                yaml["val"] = yaml["train"]
            # JSON is valid YAML and safely escapes platform-dependent paths.
            _write_json(shot_stage / "data.yaml", yaml)
            entries.append((shot_stage, paths.prepared_root / name))
        _promote(entries, stage, paths)
    except _RecoveryError:
        preserve_stage = True
        raise
    finally:
        if not preserve_stage:
            _remove_target(stage, paths)
    if not reuse_test:
        test.status = "prepared"
    for split in selected:
        split.status = "prepared"


def _write_split(root: Path, manifest_root: Path, split: _Split, taxonomy: Taxonomy,
                 paths: DatasetPaths, arguments: dict) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for image in split.report.images.values():
        relative = _source_relative(image["file_name"], paths)
        destination = root / "images" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.images_dir / relative, destination)
    for name, text in split.labels.items():
        destination = root / "labels" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    expected = _expected_files(split, paths)
    actual = {path.relative_to(root).as_posix(): _sha256(path) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise PreparationError("Staged image/label checksums disagree with audited sources")
    final = paths.prepared_root / ("test" if split.report.shot == 0 else f"shot_{split.report.shot}")
    manifest = {
        "schema_version": 1, "shot": split.report.shot, "arguments": arguments,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "versions": _versions(),
        "source_jsons": _sources(split, paths), "taxonomy": _mapping(taxonomy),
        "category_mapping_path": str(paths.prepared_root / "category_mapping.json"),
        "output_path": str(final), "image_count": split.report.image_count,
        "annotation_count": split.report.annotation_count,
        "generated_label_count": sum(len(text.splitlines()) for text in split.labels.values()),
        "label_file_count": len(split.labels), "file_checksums": expected,
        "multi_polygon_instances": split.multi_polygon_instances,
        "warnings": [asdict(issue) for issue in split.report.warnings],
        "seed": None, "weights_path": None,
    }
    _write_json(manifest_root / "manifest.json", manifest)


def _promote(entries: list[tuple[Path, Path]], stage: Path, paths: DatasetPaths) -> None:
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for staged, target in entries:
            _check_target(target, paths)
            if target.exists():
                backup = stage / ("backup-" + target.name)
                target.replace(backup)
                backups.append((backup, target))
            staged.replace(target)
            promoted.append(target)
    except (OSError, ValueError) as error:
        rollback_errors = []
        for target in reversed(promoted):
            try:
                _remove_target(target, paths)
            except (OSError, ValueError) as rollback_error:
                rollback_errors.append(str(rollback_error))
        for backup, target in reversed(backups):
            try:
                backup.replace(target)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        if rollback_errors:
            raise _RecoveryError(
                f"{error}; rollback incomplete; recovery backups retained at {stage}: "
                + "; ".join(rollback_errors)
            ) from error
        raise


def _remove_target(target: Path, paths: DatasetPaths) -> None:
    # Always prove the resolved target is beneath the exact prepared root.
    _check_target(target, paths)
    if target.resolve() == paths.prepared_root or not target.resolve().is_relative_to(paths.prepared_root):
        raise PreparationError("Refusing to remove an unscoped target: " + str(target))
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def _block(test: _Split, selected: list[_Split], message: str) -> None:
    test.status = "failed"
    test.error_message = message
    for split in selected:
        split.status = "failed"
        split.error_message = message


def _errors(split: _Split) -> str:
    return "; ".join(f"{issue.code}: {issue.message}" for issue in split.report.errors)


def _write_audit(paths: DatasetPaths, test: _Split, selected: list[_Split], arguments: dict) -> None:
    def record(split: _Split) -> dict:
        return {
            "shot": split.report.shot, "status": split.status, "error_message": split.error_message,
            "image_count": split.report.image_count, "annotation_count": split.report.annotation_count,
            "source_files": [str(path) for path in split.report.source_files],
            "errors": [asdict(issue) for issue in split.report.errors],
            "warnings": [asdict(issue) for issue in split.report.warnings],
            "multi_polygon_instances": split.multi_polygon_instances,
        }

    target = paths.results_root / "audit.json"
    _check_output(target, paths)
    paths.results_root.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": arguments, "input_layout": {
            "images_dir": str(paths.images_dir), "images_dir_exists": paths.images_dir.is_dir(),
            "few_shot_dir": str(paths.few_shot_dir), "few_shot_dir_exists": paths.few_shot_dir.is_dir(),
            "test_json": str(paths.test_json),
        }, "test": record(test), "shots": [record(split) for split in selected],
    }
    # A partial error report must not replace the last complete report.
    with tempfile.TemporaryDirectory(prefix=".audit-stage-", dir=paths.results_root) as temporary:
        staged = Path(temporary) / "audit.json"
        _write_json(staged, document)
        staged.replace(target)


def _mapping(taxonomy: Taxonomy) -> dict:
    return {"names": taxonomy.names, "category_to_index": {str(key): value for key, value in taxonomy.category_to_index.items()}}


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pycocotools"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
