"""Synthetic preparation tests: no local CAMO-FS data or YOLO dependency."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

from camo_fs.paths import DatasetPaths


CATEGORIES = [{"id": 20, "name": "Fox"}, {"id": 5, "name": "Bat"}]
REPO = Path(__file__).resolve().parents[1]


def _png(path: Path) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\x80" * 12) * 4))
        + chunk(b"IEND", b"")
    )


def _document(image_id: int, filename: str, category_id: int = 5) -> dict:
    return {
        "categories": CATEGORIES,
        "images": [{"id": image_id, "file_name": filename, "width": 4, "height": 4}],
        "annotations": [{"id": 826, "image_id": image_id, "category_id": category_id,
                         "bbox": [0, 0, 2, 2], "segmentation": [[0, 0, 2, 0, 2, 2, 0, 2]]}],
    }


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _fixture(tmp_path: Path) -> DatasetPaths:
    paths = DatasetPaths.from_root(tmp_path / "input", tmp_path / "work")
    _png(paths.images_dir / "test.png")
    _write(paths.test_json, _document(900, "test.png"))
    for shot in (1, 2, 3, 5):
        for category_id, name in ((5, "Bat"), (20, "Fox")):
            filename = f"{name.lower()}_{shot}.png"
            _png(paths.images_dir / filename)
            _write(paths.few_shot_dir / f"camo5_{name}_{shot}shot_split1.json",
                   _document(shot * 10 + category_id, filename, category_id))
    return paths


def _change(path: Path, change) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    _write(path, document)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _prepare(shots, paths, **kwargs):
    # Import inside the test so missing T04 behavior produces a focused RED.
    from camo_fs.prepare import prepare_selected

    return prepare_selected(shots, paths, overwrite=False, continue_on_error=False, **kwargs)


def test_shot_5_copies_images_and_preserves_instances_and_provenance(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    fox = paths.few_shot_dir / "camo5_Fox_5shot_split1.json"
    _change(fox, lambda doc: doc["annotations"][0].update(
        segmentation=[[0, 0, 1, 0, 1, 1, 0, 1], [2, 2, 3, 2, 3, 3, 2, 3]], bbox=[0, 0, 3, 3]))
    before = _snapshot(paths.data_root)

    outcomes = _prepare([5], paths)

    assert [(item.shot, item.status) for item in outcomes] == [(5, "prepared")]
    root = paths.prepared_root
    assert {p.name for p in (root / "shot_5/train/images").iterdir()} == {"bat_5.png", "fox_5.png"}
    assert (root / "shot_5/train/labels/bat_5.txt").read_text() == "0 0 0 0.5 0 0.5 0.5 0 0.5\n"
    fox_lines = (root / "shot_5/train/labels/fox_5.txt").read_text().splitlines()
    assert len(fox_lines) == 1 and fox_lines[0].startswith("1 ")
    assert len(fox_lines[0].split()) > 9  # Both disconnected components survive.
    assert (root / "test/labels/test.txt").read_text().startswith("0 ")
    for image in (root / "shot_5/train/images").iterdir():
        assert not image.is_symlink()
        assert image.read_bytes() == (paths.images_dir / image.name).read_bytes()
    assert _snapshot(paths.data_root) == before
    mapping = json.loads((root / "category_mapping.json").read_text())
    assert mapping["names"] == ["Bat", "Fox"]
    assert mapping["category_to_index"] == {"5": 0, "20": 1}
    manifest = json.loads(outcomes[0].manifest_path.read_text())
    assert manifest["shot"] == 5
    assert manifest["image_count"] == 2 and manifest["annotation_count"] == 2
    assert manifest["generated_label_count"] == 2 and manifest["label_file_count"] == 2
    assert manifest["multi_polygon_instances"] == [{"image_id": 70, "annotation_id": 826, "filename": "fox_5.png"}]
    assert len(manifest["source_jsons"]) == 3  # Both support files plus canonical test.
    for entry in manifest["source_jsons"]:
        assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    assert manifest["versions"]["python"] == sys.version.split()[0]
    assert manifest["timestamp_utc"].endswith("+00:00")
    assert manifest["category_mapping_path"] == str(root / "category_mapping.json")
    audit = json.loads((paths.results_root / "audit.json").read_text())
    assert audit["test"]["image_count"] == 1
    assert audit["shots"][0]["warnings"][0]["code"] == "reused_annotation_id"


def test_shot_all_copies_shared_test_once_and_each_split_once(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    import shutil
    calls = []
    original = shutil.copy2

    def copy(source, destination, *args, **kwargs):
        calls.append(Path(source).name)
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", copy)
    outcomes = _prepare([1, 2, 3, 5], paths)
    assert [(item.shot, item.status) for item in outcomes] == [(1, "prepared"), (2, "prepared"), (3, "prepared"), (5, "prepared")]
    assert calls.count("test.png") == 1 and len(calls) == 9
    assert all(calls.count(f"{name}_{shot}.png") == 1 for shot in (1, 2, 3, 5) for name in ("bat", "fox"))
    assert all(len(list((paths.prepared_root / f"shot_{shot}/train/labels").glob("*.txt"))) == 2 for shot in (1, 2, 3, 5))


def test_common_test_reused_for_later_shot_without_copying_or_modifying_it(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    _prepare([1], paths)
    before = _snapshot(paths.prepared_root / "test")
    import shutil
    original = shutil.copy2

    def copy(source, destination, *args, **kwargs):
        assert Path(source).name != "test.png"
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", copy)
    _prepare([2], paths)
    assert _snapshot(paths.prepared_root / "test") == before


@pytest.mark.parametrize("placeholder", [False, True])
def test_val_never_points_to_test(tmp_path: Path, placeholder: bool) -> None:
    paths = _fixture(tmp_path)
    result = _prepare([5], paths, val_train_placeholder=placeholder)[0]
    # JSON is YAML-compatible; parse without needing a training dependency.
    yaml = json.loads(result.data_yaml.read_text())
    assert yaml["train"] == str(paths.prepared_root / "shot_5/train/images")
    assert yaml["test"] == str(paths.prepared_root / "test/images")
    assert yaml["names"] == ["Bat", "Fox"]
    if placeholder:
        assert yaml["val"] == yaml["train"] and yaml["val"] != yaml["test"]
    else:
        assert "val" not in yaml


def test_all_fail_fast_audits_every_shot_before_materialization(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths.few_shot_dir / "camo5_Fox_3shot_split1.json").unlink()
    with pytest.raises(ValueError, match="audit"):
        _prepare([1, 2, 3, 5], paths)
    assert not paths.prepared_root.exists()
    audit = json.loads((paths.results_root / "audit.json").read_text())
    assert [entry["shot"] for entry in audit["shots"]] == [1, 2, 3, 5]
    assert audit["shots"][2]["errors"][0]["code"] == "missing_shot_file"


def test_continue_on_error_materializes_successes_and_reports_every_outcome(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths.few_shot_dir / "camo5_Fox_3shot_split1.json").unlink()
    from camo_fs.prepare import prepare_selected
    outcomes = prepare_selected([1, 2, 3, 5], paths, overwrite=False, continue_on_error=True)
    assert [(item.shot, item.status) for item in outcomes] == [(1, "prepared"), (2, "prepared"), (3, "failed"), (5, "prepared")]
    assert not (paths.prepared_root / "shot_3").exists()
    assert "missing_shot_file" in outcomes[2].error_message
    audit = json.loads((paths.results_root / "audit.json").read_text())
    assert [entry["status"] for entry in audit["shots"]] == ["prepared", "prepared", "failed", "prepared"]


@pytest.mark.parametrize("continue_on_error", [False, True])
@pytest.mark.parametrize("defect", ["image", "geometry", "schema", "taxonomy", "duplicate"])
def test_failed_shared_test_blocks_every_shot(tmp_path: Path, continue_on_error: bool, defect: str) -> None:
    paths = _fixture(tmp_path)
    if defect == "image":
        (paths.images_dir / "test.png").unlink()
    elif defect == "geometry":
        _change(paths.test_json, lambda doc: doc["annotations"][0].update(segmentation={"counts": "bad", "size": [4, 4]}))
    elif defect == "schema":
        _change(paths.test_json, lambda doc: doc.update(annotations=None))
    elif defect == "taxonomy":
        _change(paths.test_json, lambda doc: doc.update(categories=None))
    else:
        _change(paths.test_json, lambda doc: doc["annotations"].append(doc["annotations"][0].copy()))
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="test.*audit"):
        prepare_selected([1, 2], paths, overwrite=False, continue_on_error=continue_on_error)
    assert not paths.prepared_root.exists()
    audit = json.loads((paths.results_root / "audit.json").read_text())
    assert audit["test"]["errors"]
    assert [entry["status"] for entry in audit["shots"]] == ["failed", "failed"]


@pytest.mark.parametrize("geometry", [{"bbox": [-1, 0, 1, 1]}, {"segmentation": [[0, 0, 1, 1, 2, 2]]}, {"segmentation": [[0, 0, 1]]}])
def test_training_geometry_failure_is_reported_before_any_write(tmp_path: Path, geometry: dict) -> None:
    paths = _fixture(tmp_path)
    _change(paths.few_shot_dir / "camo5_Bat_5shot_split1.json", lambda doc: doc["annotations"][0].update(geometry))
    with pytest.raises(ValueError, match="audit"):
        _prepare([5], paths)
    assert not paths.prepared_root.exists()
    error = json.loads((paths.results_root / "audit.json").read_text())["shots"][0]["errors"][0]
    assert error["image_id"] == 55 and error["annotation_id"] == 826
    assert error["filename"] == "bat_5.png"


def test_camo_half_pixel_boundaries_prepare_test_and_train_without_source_writes(tmp_path):
    paths = _fixture(tmp_path)
    geometry = {"bbox": [-0.5, -0.5, 2.5, 2.5],
                "segmentation": [[-0.5, -0.5, 2, -0.5, 2, 2, -0.5, 2]]}
    for path in (paths.test_json, paths.few_shot_dir / "camo5_Bat_1shot_split1.json"):
        _change(path, lambda doc: doc["annotations"][0].update(geometry))
    source_before = _snapshot(paths.data_root)
    outcomes = _prepare([1], paths)
    assert outcomes[0].status == "prepared"
    assert (paths.prepared_root / "test/labels/test.txt").read_text() == "0 0 0 0.5 0 0.5 0.5 0 0.5\n"
    assert (paths.prepared_root / "shot_1/train/labels/bat_1.txt").read_text() == "0 0 0 0.5 0 0.5 0.5 0 0.5\n"
    assert _snapshot(paths.data_root) == source_before


def test_existing_selected_target_fails_without_overwrite(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _prepare([5], paths)
    before = _snapshot(paths.prepared_root)
    with pytest.raises(ValueError, match="exists"):
        _prepare([5], paths)
    assert _snapshot(paths.prepared_root) == before


def test_overwrite_rebuilds_selected_targets_without_stale_files(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    untouched = _snapshot(paths.prepared_root / "shot_1")
    for relative in ("shot_5/train/labels/stale.txt", "shot_5/train/images/stale.png", "test/labels/stale.txt"):
        (paths.prepared_root / relative).write_text("stale")
    from camo_fs.prepare import prepare_selected
    prepare_selected([5], paths, overwrite=True, continue_on_error=False)
    assert not list(paths.prepared_root.rglob("stale.*"))
    assert _snapshot(paths.prepared_root / "shot_1") == untouched
    assert len(list((paths.prepared_root / "test/labels").iterdir())) == 1


def test_failed_overwrite_preserves_all_existing_artifacts(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 2, 3, 5], paths)
    before = _snapshot(paths.prepared_root)
    (paths.few_shot_dir / "camo5_Fox_3shot_split1.json").unlink()
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="audit"):
        prepare_selected([1, 2, 3, 5], paths, overwrite=True, continue_on_error=False)
    assert _snapshot(paths.prepared_root) == before


def test_copy_failure_does_not_promote_partial_split_or_destroy_previous_target(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    _prepare([5], paths)
    before = _snapshot(paths.prepared_root)
    import shutil
    original = shutil.copy2

    def copy(source, destination, *args, **kwargs):
        if Path(source).name == "fox_5.png":
            raise OSError("synthetic copy failure")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", copy)
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="copy failure"):
        prepare_selected([5], paths, overwrite=True, continue_on_error=False)
    assert _snapshot(paths.prepared_root) == before
    assert not list(paths.prepared_root.glob(".*stage*"))


@pytest.mark.parametrize("tamper", ["source_json", "label", "extra"])
def test_shared_test_reuse_rejects_stale_or_corrupt_artifacts(tmp_path: Path, tamper: str) -> None:
    paths = _fixture(tmp_path)
    _prepare([1], paths)
    if tamper == "source_json":
        _change(paths.test_json, lambda doc: doc.update(info={"version": "changed"}))
    elif tamper == "label":
        (paths.prepared_root / "test/labels/test.txt").write_text("bad label")
    else:
        (paths.prepared_root / "test/images/stale.png").write_text("stale")
    with pytest.raises(ValueError, match="shared test"):
        _prepare([2], paths)
    assert not (paths.prepared_root / "shot_2").exists()


@pytest.mark.parametrize("filename", ["../escape.png", "/absolute.png", "C:\\escape.png"])
def test_unsafe_source_filename_blocks_materialization(tmp_path: Path, filename: str) -> None:
    paths = _fixture(tmp_path)
    _change(paths.few_shot_dir / "camo5_Bat_5shot_split1.json", lambda doc: doc["images"][0].update(file_name=filename))
    with pytest.raises(ValueError, match="audit"):
        _prepare([5], paths)
    assert not paths.prepared_root.exists()


def test_image_names_colliding_as_labels_fail_before_write(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _png(paths.images_dir / "bat_5.jpg")
    _change(paths.few_shot_dir / "camo5_Fox_5shot_split1.json", lambda doc: doc["images"][0].update(file_name="bat_5.jpg"))
    with pytest.raises(ValueError, match="audit"):
        _prepare([5], paths)
    assert not paths.prepared_root.exists()
    errors = json.loads((paths.results_root / "audit.json").read_text())["shots"][0]["errors"]
    assert "label_filename_collision" in {item["code"] for item in errors}


def _cli(paths: DatasetPaths, *flags: str):
    return subprocess.run([sys.executable, str(REPO / "scripts/prepare_dataset.py"),
                           "--data-root", str(paths.data_root), "--work-root", str(paths.work_root),
                           *flags], cwd=REPO, capture_output=True, text=True)


@pytest.mark.parametrize("shot", ["5", "all"])
def test_cli_prepares_one_or_all_shots(tmp_path: Path, shot: str) -> None:
    paths = _fixture(tmp_path)
    result = _cli(paths, "--shot", shot)
    assert result.returncode == 0, result.stderr
    actual = sorted(path.name for path in paths.prepared_root.glob("shot_*"))
    assert actual == (["shot_5"] if shot == "5" else ["shot_1", "shot_2", "shot_3", "shot_5"])


def test_cli_failure_exit_and_continue_on_error_and_overwrite(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths.few_shot_dir / "camo5_Fox_3shot_split1.json").unlink()
    result = _cli(paths, "--shot", "all", "--continue-on-error", "--val-train-placeholder")
    assert result.returncode == 1
    assert (paths.prepared_root / "shot_5/data.yaml").is_file()
    assert "missing_shot_file" in result.stdout + result.stderr
    result = _cli(paths, "--shot", "5", "--overwrite")
    assert result.returncode == 0, result.stderr
    assert "val" not in json.loads((paths.prepared_root / "shot_5/data.yaml").read_text())


def test_cli_help_and_invalid_shot_are_handled_before_io(tmp_path: Path) -> None:
    paths = DatasetPaths.from_root(tmp_path / "absent", tmp_path / "work")
    help_result = _cli(paths, "--help")
    assert help_result.returncode == 0
    assert "/kaggle/input/datasets/danhnt/camo-fs-dataset" in help_result.stdout
    assert "/kaggle/working" in help_result.stdout
    assert all(flag in help_result.stdout for flag in ("--overwrite", "--continue-on-error", "--shot"))
    invalid = _cli(paths, "--shot", "4")
    assert invalid.returncode == 2
    assert not paths.work_root.exists()


def test_validation_failure_during_promotion_rolls_back_prior_targets(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    _prepare([5], paths)
    before = _snapshot(paths.prepared_root)
    import camo_fs.prepare as preparation
    original = preparation._check_target

    def check(target, checked_paths):
        if target == paths.prepared_root / "shot_5" and list(paths.prepared_root.glob(".prepare-stage-*")):
            raise preparation.PreparationError("synthetic promotion validation failure")
        return original(target, checked_paths)

    monkeypatch.setattr(preparation, "_check_target", check)
    with pytest.raises(ValueError, match="promotion validation failure"):
        preparation.prepare_selected([5], paths, overwrite=True, continue_on_error=False)
    assert _snapshot(paths.prepared_root) == before


def test_failed_rollback_keeps_original_backup_and_reports_recovery_directory(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    _prepare([5], paths)
    test_before = _snapshot(paths.prepared_root / "test")
    original = Path.replace

    def replace(source, destination):
        if source.name == "shot_5" and source.parent.name.startswith(".prepare-stage-"):
            raise OSError("synthetic promotion failure")
        if source.name == "backup-test":
            raise OSError("synthetic restore failure")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="recovery") as error:
        prepare_selected([5], paths, overwrite=True, continue_on_error=False)
    backups = list(paths.prepared_root.glob(".prepare-stage-*/backup-test"))
    assert len(backups) == 1
    assert _snapshot(backups[0]) == test_before
    assert str(backups[0].parent) in str(error.value)


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_empty_shared_test_is_rejected_before_preparation(tmp_path: Path, continue_on_error: bool) -> None:
    paths = _fixture(tmp_path)
    _change(paths.test_json, lambda doc: doc.update(images=[], annotations=[]))
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="test.*audit"):
        prepare_selected([1], paths, overwrite=False, continue_on_error=continue_on_error)
    assert not paths.prepared_root.exists()
    assert json.loads((paths.results_root / "audit.json").read_text())["test"]["errors"]


@pytest.mark.parametrize("corruption", ["invalid JSON", "{}"])
def test_overwrite_repairs_corrupt_mapping_without_changing_unselected_shot(tmp_path: Path, corruption: str) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    before = _snapshot(paths.prepared_root / "shot_1")
    (paths.prepared_root / "category_mapping.json").write_text(corruption)
    from camo_fs.prepare import prepare_selected
    prepare_selected([5], paths, overwrite=True, continue_on_error=False)
    assert _snapshot(paths.prepared_root / "shot_1") == before
    assert json.loads((paths.prepared_root / "category_mapping.json").read_text())["names"] == ["Bat", "Fox"]


def test_overwrite_rejects_incompatible_unselected_taxonomy_before_any_write(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    _change(paths.prepared_root / "shot_1/manifest.json", lambda doc: doc.update(taxonomy={"names": ["Wrong"]}))
    before = _snapshot(paths.prepared_root)
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="unselected.*taxonomy"):
        prepare_selected([5], paths, overwrite=True, continue_on_error=False)
    assert _snapshot(paths.prepared_root) == before


@pytest.mark.parametrize("continue_on_error", [False, True])
@pytest.mark.parametrize("change", ["metadata", "geometry"])
def test_partial_overwrite_rejects_changed_shared_test_provenance(
    tmp_path: Path, continue_on_error: bool, change: str,
) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    before = _snapshot(paths.prepared_root)
    if change == "metadata":
        _change(paths.test_json, lambda doc: doc.update(info={"version": "changed"}))
    else:
        _change(paths.test_json, lambda doc: doc["annotations"][0].update(
            bbox=[1, 1, 2, 2], segmentation=[[1, 1, 3, 1, 3, 3, 1, 3]]))
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="shared test.*provenance"):
        prepare_selected([5], paths, overwrite=True, continue_on_error=continue_on_error)
    assert _snapshot(paths.prepared_root) == before
    audit = json.loads((paths.results_root / "audit.json").read_text())
    assert audit["test"]["errors"][0]["code"] == "shared_target_error"
    assert audit["shots"][0]["status"] == "failed"


@pytest.mark.parametrize("sources", [None, [], [42]])
def test_partial_overwrite_rejects_unverifiable_retained_test_provenance(tmp_path: Path, sources) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    _change(paths.prepared_root / "shot_1/manifest.json", lambda doc: doc.update(source_jsons=sources))
    before = _snapshot(paths.prepared_root)
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="shared test.*provenance"):
        prepare_selected([5], paths, overwrite=True, continue_on_error=False)
    assert _snapshot(paths.prepared_root) == before


def test_overwrite_all_retained_shots_accepts_changed_test_provenance(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    _change(paths.test_json, lambda doc: doc["annotations"][0].update(
        bbox=[1, 1, 2, 2], segmentation=[[1, 1, 3, 1, 3, 3, 1, 3]]))
    from camo_fs.prepare import prepare_selected
    prepare_selected([1, 5], paths, overwrite=True, continue_on_error=False)
    expected_sha = hashlib.sha256(paths.test_json.read_bytes()).hexdigest()
    for relative in ("test/manifest.json", "shot_1/manifest.json", "shot_5/manifest.json"):
        manifest = json.loads((paths.prepared_root / relative).read_text())
        entries = [entry for entry in manifest["source_jsons"] if entry["path"] == str(paths.test_json)]
        assert entries == [{"path": str(paths.test_json), "sha256": expected_sha}]
    assert (paths.prepared_root / "test/labels/test.txt").read_text() == "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"


@pytest.mark.parametrize("field,value", [
    ("image_count", 999), ("annotation_count", 999), ("generated_label_count", 999),
    ("label_file_count", 999), ("multi_polygon_instances", [{"image_id": 999}]),
])
def test_shared_test_reuse_rejects_corrupt_deterministic_metadata(tmp_path: Path, field: str, value) -> None:
    paths = _fixture(tmp_path)
    _prepare([1], paths)
    _change(paths.prepared_root / "test/manifest.json", lambda doc: doc.update({field: value}))
    before = _snapshot(paths.prepared_root)
    with pytest.raises(ValueError, match="shared test"):
        _prepare([5], paths)
    assert _snapshot(paths.prepared_root) == before


@pytest.mark.parametrize("change", ["metadata", "geometry"])
def test_changed_shared_test_with_continue_on_error_does_not_leave_stale_selected_target(tmp_path: Path, change: str) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    before = _snapshot(paths.prepared_root)
    if change == "metadata":
        _change(paths.test_json, lambda doc: doc.update(info={"version": "changed"}))
    else:
        _change(paths.test_json, lambda doc: doc["annotations"][0].update(
            bbox=[1, 1, 2, 2], segmentation=[[1, 1, 3, 1, 3, 3, 1, 3]]))
    (paths.few_shot_dir / "camo5_Fox_5shot_split1.json").unlink()
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="shared test|audit"):
        prepare_selected([1, 2, 3, 5], paths, overwrite=True, continue_on_error=True)
    assert _snapshot(paths.prepared_root) == before
    audit = json.loads((paths.results_root / "audit.json").read_text())
    assert [entry["shot"] for entry in audit["shots"]] == [1, 2, 3, 5]
    assert "missing_shot_file" in {issue["code"] for issue in audit["shots"][-1]["errors"]}


def test_changed_shared_test_copy_failure_keeps_all_existing_selected_targets(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    before = _snapshot(paths.prepared_root)
    _change(paths.test_json, lambda doc: doc.update(info={"version": "changed"}))
    import shutil
    original = shutil.copy2

    def copy(source, destination, *args, **kwargs):
        if Path(source).name == "fox_5.png":
            raise OSError("synthetic selected copy failure")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", copy)
    from camo_fs.prepare import prepare_selected
    with pytest.raises(ValueError, match="copy failure"):
        prepare_selected([1, 5], paths, overwrite=True, continue_on_error=True)
    assert _snapshot(paths.prepared_root) == before


def test_changed_shared_test_rebuilds_old_targets_but_reports_failed_new_shot(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _prepare([1, 5], paths)
    _change(paths.test_json, lambda doc: doc.update(info={"version": "changed"}))
    (paths.few_shot_dir / "camo5_Fox_3shot_split1.json").unlink()
    from camo_fs.prepare import prepare_selected
    outcomes = prepare_selected([1, 2, 3, 5], paths, overwrite=True, continue_on_error=True)
    assert [(item.shot, item.status) for item in outcomes] == [(1, "prepared"), (2, "prepared"), (3, "failed"), (5, "prepared")]
    assert not (paths.prepared_root / "shot_3").exists()
    expected = {"path": str(paths.test_json), "sha256": hashlib.sha256(paths.test_json.read_bytes()).hexdigest()}
    for name in ("test", "shot_1", "shot_2", "shot_5"):
        manifest = json.loads((paths.prepared_root / name / "manifest.json").read_text())
        assert [entry for entry in manifest["source_jsons"] if entry["path"] == str(paths.test_json)] == [expected]
