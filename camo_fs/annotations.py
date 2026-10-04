"""Read-only COCO taxonomy and official split auditing."""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from camo_fs.paths import DatasetPaths


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """Canonical COCO category identity and contiguous YOLO indices."""

    names: list[str]
    category_to_index: dict[int, int]
    category_pairs: tuple[tuple[int, str], ...]

    @classmethod
    def from_test_json(cls, path: Path) -> "Taxonomy":
        document = _load_json(path)
        pairs = _category_pairs(document, path)
        return cls(
            names=[name for _, name in pairs],
            category_to_index={category_id: index for index, (category_id, _) in enumerate(pairs)},
            category_pairs=tuple(pairs),
        )


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    message: str
    image_id: int | None = None
    annotation_id: int | None = None
    filename: str | None = None


@dataclass(slots=True)
class AuditReport:
    """Read-only evidence gathered for one selected official shot."""

    shot: int
    images: dict[int, dict[str, Any]] = field(default_factory=dict)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    source_files: list[Path] = field(default_factory=list)
    errors: list[AuditIssue] = field(default_factory=list)
    warnings: list[AuditIssue] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)


def audit_shot(shot: int, paths: DatasetPaths, taxonomy: Taxonomy) -> AuditReport:
    """Audit one official train shot and its overlap with the common test JSON."""
    report = AuditReport(shot=shot)
    source_files = sorted(paths.few_shot_dir.glob(f"camo5_*_{shot}shot_split1.json"))
    report.source_files.extend(source_files)
    _audit_shot_file_set(shot, source_files, taxonomy, report)

    seen_annotation_content: set[str] = set()
    seen_annotation_ids: dict[int, dict[tuple[int, int], str]] = {}
    filename_to_image_id: dict[str, int] = {}

    for source_file in source_files:
        try:
            document = _load_json(source_file)
        except (OSError, ValueError) as error:
            report.errors.append(AuditIssue("invalid_json", str(error), filename=source_file.name))
            continue
        _audit_taxonomy(document, source_file, taxonomy, report)
        _audit_training_document(
            document,
            paths,
            taxonomy,
            report,
            seen_annotation_content,
            seen_annotation_ids,
            filename_to_image_id,
        )

    _audit_test_overlap(paths, taxonomy, report)
    return report


def audit_test(paths: DatasetPaths, taxonomy: Taxonomy) -> AuditReport:
    """Audit shared-test metadata and object identity without writing data.

    Shot zero denotes the shared test. Polygon conversion/validation is the
    preparation preflight's responsibility for both train and test records.
    """
    report = AuditReport(shot=0, source_files=[paths.test_json])
    try:
        document = _load_json(paths.test_json)
    except (OSError, ValueError) as error:
        report.errors.append(AuditIssue("invalid_test_json", str(error), filename=paths.test_json.name))
        return report
    _audit_taxonomy(document, paths.test_json, taxonomy, report)
    _audit_training_document(document, paths, taxonomy, report, set(), {}, {})
    return report


def _audit_shot_file_set(
    shot: int, source_files: list[Path], taxonomy: Taxonomy, report: AuditReport
) -> None:
    expected = {
        f"camo5_{_filename_token(name)}_{shot}shot_split1.json"
        for _, name in taxonomy.category_pairs
    }
    found = {path.name for path in source_files}
    for missing in sorted(expected - found):
        report.errors.append(AuditIssue("missing_shot_file", f"Missing official shot file: {missing}", filename=missing))
    for unexpected in sorted(found - expected):
        report.errors.append(
            AuditIssue("unexpected_shot_file", f"Unexpected shot file: {unexpected}", filename=unexpected)
        )


def _audit_training_document(
    document: dict[str, Any],
    paths: DatasetPaths,
    taxonomy: Taxonomy,
    report: AuditReport,
    seen_annotation_content: set[str],
    seen_annotation_ids: dict[int, dict[tuple[int, int], str]],
    filename_to_image_id: dict[str, int],
) -> None:
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        report.errors.append(AuditIssue("invalid_coco_schema", "COCO images and annotations must be lists"))
        return

    source_image_ids = {
        image.get("id") for image in images if isinstance(image, dict) and isinstance(image.get("id"), int)
    }
    for image in images:
        if not isinstance(image, dict):
            report.errors.append(AuditIssue("invalid_image", "Image record must be an object"))
            continue
        _audit_image(image, paths, report, filename_to_image_id)

    for annotation in annotations:
        if not isinstance(annotation, dict):
            report.errors.append(AuditIssue("invalid_annotation", "Annotation record must be an object"))
            continue
        _audit_annotation(
            annotation,
            taxonomy,
            report,
            source_image_ids,
            seen_annotation_content,
            seen_annotation_ids,
        )


def _audit_image(
    image: dict[str, Any],
    paths: DatasetPaths,
    report: AuditReport,
    filename_to_image_id: dict[str, int],
) -> None:
    image_id = image.get("id")
    filename = image.get("file_name")
    width = image.get("width")
    height = image.get("height")
    if not isinstance(image_id, int) or not isinstance(filename, str) or not filename:
        report.errors.append(AuditIssue("invalid_image", "Image requires integer id and non-empty file_name"))
        return
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        report.errors.append(
            AuditIssue("invalid_image_dimensions", "Image requires positive width and height", image_id, filename=filename)
        )
        return

    existing = report.images.get(image_id)
    identity = (filename, width, height)
    if existing is not None and (existing["file_name"], existing["width"], existing["height"]) != identity:
        report.errors.append(
            AuditIssue("conflicting_image_metadata", "Image id has conflicting metadata", image_id, filename=filename)
        )
        return
    previous_image_id = filename_to_image_id.setdefault(filename, image_id)
    if previous_image_id != image_id:
        report.errors.append(
            AuditIssue("filename_mapped_to_multiple_image_ids", "Filename maps to multiple image ids", image_id, filename=filename)
        )
        return

    report.images.setdefault(image_id, image)
    image_path = paths.images_dir / filename
    if not image_path.is_file():
        report.errors.append(AuditIssue("missing_image", "Referenced image is missing", image_id, filename=filename))
        return
    try:
        actual_width, actual_height = _image_dimensions(image_path)
    except ValueError as error:
        report.errors.append(AuditIssue("unsupported_image", str(error), image_id, filename=filename))
        return
    if (actual_width, actual_height) != (width, height):
        report.errors.append(
            AuditIssue(
                "image_dimension_mismatch",
                f"Declared {width}x{height}, source is {actual_width}x{actual_height}",
                image_id,
                filename=filename,
            )
        )


def _audit_annotation(
    annotation: dict[str, Any],
    taxonomy: Taxonomy,
    report: AuditReport,
    source_image_ids: set[int],
    seen_annotation_content: set[str],
    seen_annotation_ids: dict[int, dict[tuple[int, int], str]],
) -> None:
    annotation_id = annotation.get("id")
    image_id = annotation.get("image_id")
    category_id = annotation.get("category_id")
    if not all(isinstance(value, int) for value in (annotation_id, image_id, category_id)):
        report.errors.append(AuditIssue("invalid_annotation", "Annotation requires integer id, image_id, and category_id"))
        return
    if category_id not in taxonomy.category_to_index:
        report.errors.append(
            AuditIssue("unmapped_category", "Annotation category is absent from taxonomy", image_id, annotation_id)
        )
        return
    if image_id not in source_image_ids:
        report.errors.append(
            AuditIssue("annotation_image_missing", "Annotation references no image in this source JSON", image_id, annotation_id)
        )
        return

    try:
        geometry_key = _canonical_json(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": annotation.get("bbox"),
                "segmentation": annotation.get("segmentation"),
            }
        )
    except (TypeError, ValueError):
        report.errors.append(
            AuditIssue(
                "invalid_annotation_geometry",
                "Annotation geometry cannot be serialized as finite JSON",
                image_id,
                annotation_id,
            )
        )
        return
    if geometry_key in seen_annotation_content:
        report.errors.append(
            AuditIssue("duplicate_annotation", "Duplicate annotation geometry", image_id, annotation_id)
        )
        return
    seen_annotation_content.add(geometry_key)

    context = (image_id, category_id)
    known_contexts = seen_annotation_ids.setdefault(annotation_id, {})
    previous_geometry = known_contexts.get(context)
    if previous_geometry is None:
        if known_contexts:
            report.warnings.append(
                AuditIssue("reused_annotation_id", "Annotation id is reused by a distinct object", image_id, annotation_id)
            )
        known_contexts[context] = geometry_key
    elif previous_geometry != geometry_key:
        report.errors.append(
            AuditIssue("conflicting_annotation_id", "Annotation id has conflicting geometry", image_id, annotation_id)
        )
    report.annotations.append(annotation)


def _audit_test_overlap(paths: DatasetPaths, taxonomy: Taxonomy, report: AuditReport) -> None:
    if not paths.test_json.is_file():
        report.errors.append(AuditIssue("missing_test_json", "Official test JSON is missing", filename=paths.test_json.name))
        return
    try:
        document = _load_json(paths.test_json)
    except (OSError, ValueError) as error:
        report.errors.append(AuditIssue("invalid_test_json", str(error), filename=paths.test_json.name))
        return
    _audit_taxonomy(document, paths.test_json, taxonomy, report)
    test_images = document.get("images")
    if not isinstance(test_images, list):
        report.errors.append(AuditIssue("invalid_coco_schema", "Test JSON images must be a list"))
        return
    test_ids = {image.get("id") for image in test_images if isinstance(image, dict)}
    test_filenames = {image.get("file_name") for image in test_images if isinstance(image, dict)}
    for image_id, image in report.images.items():
        if image_id in test_ids:
            report.errors.append(
                AuditIssue("train_test_image_id_overlap", "Train and test share image id", image_id, filename=image["file_name"])
            )
        if image["file_name"] in test_filenames:
            report.errors.append(
                AuditIssue("train_test_filename_overlap", "Train and test share filename", image_id, filename=image["file_name"])
            )


def _audit_taxonomy(document: dict[str, Any], source: Path, taxonomy: Taxonomy, report: AuditReport) -> None:
    try:
        pairs = tuple(_category_pairs(document, source))
    except ValueError as error:
        report.errors.append(AuditIssue("invalid_taxonomy", str(error), filename=source.name))
        return
    if pairs != taxonomy.category_pairs:
        report.errors.append(AuditIssue("taxonomy_mismatch", "Categories differ from official test taxonomy", filename=source.name))


def _category_pairs(document: dict[str, Any], source: Path) -> list[tuple[int, str]]:
    categories = document.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"{source}: categories must be a list")
    pairs: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"{source}: category must be an object")
        category_id = category.get("id")
        name = category.get("name")
        if not isinstance(category_id, int) or not isinstance(name, str) or not name:
            raise ValueError(f"{source}: category requires integer id and non-empty name")
        if category_id in seen_ids:
            raise ValueError(f"{source}: duplicate category id {category_id}")
        seen_ids.add(category_id)
        pairs.append((category_id, name))
    return sorted(pairs)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return document


def _filename_token(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _image_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24 and header[12:16] == b"IHDR":
            return struct.unpack(">II", header[16:24])
        if header.startswith(b"\xff\xd8"):
            file.seek(2)
            return _jpeg_dimensions(file, path)
    raise ValueError(f"Unsupported image format: {path.name}")


def _jpeg_dimensions(file: Any, path: Path) -> tuple[int, int]:
    while True:
        marker_start = file.read(1)
        if not marker_start:
            break
        if marker_start != b"\xff":
            continue
        marker = file.read(1)
        while marker == b"\xff":
            marker = file.read(1)
        if not marker or marker in {b"\xd8", b"\xd9"} or b"\xd0" <= marker <= b"\xd7":
            continue
        length_bytes = file.read(2)
        if len(length_bytes) != 2:
            break
        length = struct.unpack(">H", length_bytes)[0]
        if length < 2:
            break
        if marker[0] in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            payload = file.read(5)
            if len(payload) != 5:
                break
            height, width = struct.unpack(">HH", payload[1:5])
            return width, height
        file.seek(length - 2, 1)
    raise ValueError(f"Invalid JPEG image: {path.name}")
