"""Run identity and recovery behavior, using only disposable synthetic files."""

import csv
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import sys

import pytest


WEIGHTS_SHA = "a" * 64
DATA_SHA = "b" * 64


def _api():
    from camo_fs import runs
    return runs


def _config(**kwargs):
    return _api().RunConfig(weights_sha256=WEIGHTS_SHA, data_sha256=DATA_SHA, **kwargs)


def _provenance(shot: int = 1) -> dict:
    return {"schema_version": 1, "shot": shot,
            "timestamp_utc": "2026-10-04T00:00:00+00:00",
            "versions": {"python": sys.version.split()[0], "ultralytics": None, "torch": None,
                         "torchvision": None, "numpy": None, "opencv-python": None, "pycocotools": None},
            "taxonomy": {"names": ["Bat", "Fox"], "category_to_index": {"5": 0, "20": 1}},
            "label_file_count": 2, "arguments": {"shots": [shot]}, "file_checksums": {},
            "category_mapping_path": "/kaggle/working/camo_fs_yolo/category_mapping.json",
            "source_jsons": [{"path": "/kaggle/input/official.json", "sha256": DATA_SHA}],
            "image_count": 2, "annotation_count": 3, "generated_label_count": 3,
            "output_path": f"/kaggle/working/camo_fs_yolo/shot_{shot}"}


def _initialize(tmp_path: Path, config=None, **kwargs):
    config = config or _config()
    return _api().initialize_run(config, tmp_path / "runs", prepared_manifest=_provenance(config.shot), **kwargs)


def _row(config, status="failed", error="synthetic failure") -> dict:
    api = _api()
    manifest = api.build_manifest(config, Path("/kaggle/working/runs"), prepared_manifest=_provenance(config.shot))
    manifest["status"] = status
    manifest["error_message"] = error
    return api.summary_row(manifest)


def test_stable_fingerprint() -> None:
    api = _api()
    config = _config(training_options={"lr0": 0.01, "optimizer": "SGD"})
    other = _config(training_options={"optimizer": "SGD", "lr0": 0.01})
    assert api.fingerprint(config, WEIGHTS_SHA, DATA_SHA) == api.fingerprint(other, WEIGHTS_SHA, DATA_SHA)
    assert len(api.fingerprint(config, WEIGHTS_SHA, DATA_SHA)) == 64


@pytest.mark.parametrize("changes", [{"epochs": 101}, {"imgsz": 320}, {"batch": 8}, {"seed": 2025},
                                     {"shot": 5}, {"device": "cpu"}, {"training_options": {"lr0": 0.02}}])
def test_training_config_changes_identity(tmp_path: Path, changes: dict) -> None:
    api = _api()
    config = _config()
    assert api.run_path(config, tmp_path) != api.run_path(replace(config, **changes), tmp_path)


def test_augmentation_and_applicable_triplet_options_change_identity(tmp_path: Path) -> None:
    api = _api()
    enhanced = _config(method="fgbg-triplet")
    for changes in ({"triplet_weight": 0.2}, {"triplet_margin": 0.4}, {"triplets_per_instance": 8},
                    {"augmentation": replace(enhanced.augmentation, fliplr=0.0)}):
        assert api.run_path(enhanced, tmp_path) != api.run_path(replace(enhanced, **changes), tmp_path)
    baseline = _config()
    assert api.run_path(baseline, tmp_path) == api.run_path(replace(baseline, triplet_weight=0.9), tmp_path)


def test_methods_have_separate_paths(tmp_path: Path) -> None:
    api = _api()
    baseline = api.run_path(_config(shot=5), tmp_path)
    enhanced = api.run_path(_config(method="fgbg-triplet", shot=5), tmp_path)
    assert baseline.parts[-5:-1] == ("yolo11n-seg", "baseline", "shot_5", "seed_2024")
    assert enhanced.parts[-4] == "fgbg-triplet"
    assert baseline != enhanced


def test_data_checksum_changes_identity(tmp_path: Path) -> None:
    api = _api()
    config = _config()
    assert api.run_path(config, tmp_path) != api.run_path(replace(config, data_sha256="c" * 64), tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        api.fingerprint(config, WEIGHTS_SHA, "c" * 64)


def test_smoke_and_benchmark_have_distinct_identity(tmp_path: Path) -> None:
    api = _api()
    benchmark = _config()
    smoke = replace(benchmark, run_kind="smoke")
    assert api.run_path(benchmark, tmp_path) != api.run_path(smoke, tmp_path)
    api.upsert_summary(tmp_path / "results", _row(benchmark))
    api.upsert_summary(tmp_path / "results", _row(smoke))
    with (tmp_path / "results/summary.csv").open(newline="") as source:
        assert {row["run_kind"] for row in csv.DictReader(source)} == {"smoke", "benchmark"}


def test_config_is_deeply_immutable_and_weight_location_does_not_change_hash(tmp_path: Path) -> None:
    options = {"lr0": 0.01}
    config = _config(training_options=options, weights="yolo11n-seg.pt")
    options["lr0"] = 0.5
    assert config.training_options["lr0"] == 0.01
    with pytest.raises(FrozenInstanceError):
        config.epochs = 7
    with pytest.raises(TypeError):
        config.training_options["lr0"] = 7
    with pytest.raises(FrozenInstanceError):
        config.augmentation.fliplr = 0.0
    assert _api().run_path(config, tmp_path) == _api().run_path(replace(config, weights="/kaggle/input/weights/yolo11n-seg.pt"), tmp_path)


@pytest.mark.parametrize("changes", [{"method": "../bad"}, {"model": "../bad"}, {"run_kind": "bad"},
                                    {"shot": True}, {"epochs": 0}, {"batch": 0}, {"seed": -1},
                                    {"triplet_margin": float("nan"), "method": "fgbg-triplet"},
                                    {"training_options": {"lr0": float("inf")}}])
def test_invalid_config_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        _config(**changes)


def test_manifest_records_resolved_identity_provenance_versions_and_warnings(tmp_path: Path) -> None:
    config = _config(method="fgbg-triplet")
    provenance = _provenance()
    manifest = _api().build_manifest(config, tmp_path / "runs", prepared_manifest=provenance, warnings=["CUDA determinism unverified"])
    provenance["image_count"] = 999
    assert manifest["prepared_dataset"]["image_count"] == 2
    assert manifest["weights_sha256"] == WEIGHTS_SHA and manifest["data_sha256"] == DATA_SHA
    assert manifest["config_hash"] == _api().fingerprint(config, WEIGHTS_SHA, DATA_SHA)
    assert manifest["versions"]["python"] == sys.version.split()[0]
    assert manifest["timestamp_utc"].endswith("+00:00")
    assert manifest["warnings"] == ["CUDA determinism unverified"]
    assert manifest["auxiliary_loss"]["triplet_weight"] == 0.1
    assert manifest["config"]["augmentation"]["mosaic"] == 0.0
    assert manifest["config"]["augmentation"]["scale"] == 0.0
    assert manifest["config"]["weights"] == "yolo11n-seg.pt"
    assert _api().build_manifest(_config(), tmp_path, prepared_manifest=_provenance())["auxiliary_loss"] is None


@pytest.mark.parametrize("changes", [{"method": "fgbg-triplet"}, {"shot": 5}, {"run_kind": "smoke"},
                                    {"weights_sha256": "c" * 64}, {"data_sha256": "d" * 64}, {"epochs": 101}])
def test_resume_rejects_method_or_weight_mismatch(tmp_path: Path, changes: dict) -> None:
    api = _api()
    config = _config()
    manifest = _initialize(tmp_path, config)
    path = Path(manifest["last_checkpoint"])
    path.parent.mkdir()
    path.write_bytes(b"synthetic checkpoint")
    manifest = api.transition_run(Path(manifest["run_dir"]), "running")
    other = replace(config, **changes)
    with pytest.raises(ValueError, match="identity|configuration|checksum"):
        api.verify_resume(manifest, other, other.weights_sha256, other.data_sha256)


def test_resume_requires_own_last_checkpoint(tmp_path: Path) -> None:
    api = _api()
    config = _config()
    manifest = _initialize(tmp_path, config)
    manifest = api.transition_run(Path(manifest["run_dir"]), "running")
    with pytest.raises(ValueError, match="checkpoint"):
        api.verify_resume(manifest, config, WEIGHTS_SHA, DATA_SHA)
    checkpoint = Path(manifest["last_checkpoint"])
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"own checkpoint")
    api.verify_resume(manifest, config, WEIGHTS_SHA, DATA_SHA)
    manifest["last_checkpoint"] = str(tmp_path / "another-run/last.pt")
    with pytest.raises(ValueError, match="own|checkpoint"):
        api.verify_resume(manifest, config, WEIGHTS_SHA, DATA_SHA)


def test_state_transitions_and_explicit_resume(tmp_path: Path) -> None:
    api = _api()
    manifest = _initialize(tmp_path)
    root = Path(manifest["run_dir"])
    with pytest.raises(ValueError, match="transition"):
        api.transition_run(root, "completed")
    api.transition_run(root, "running")
    api.transition_run(root, "failed", error_message="interrupted")
    checkpoint = root / "weights/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    resumed = _initialize(tmp_path, resume=True)
    assert resumed["status"] == "failed" and checkpoint.read_bytes() == b"checkpoint"
    api.transition_run(root, "running")
    api.transition_run(root, "completed")
    with pytest.raises(ValueError, match="interrupted|completed"):
        _initialize(tmp_path, resume=True)


def test_overwrite_starts_fresh_preserves_other_runs_and_rejects_unmanaged_target(tmp_path: Path) -> None:
    api = _api()
    manifest = _initialize(tmp_path)
    root = Path(manifest["run_dir"])
    stale = root / "weights/last.pt"
    stale.parent.mkdir()
    stale.write_bytes(b"old fine-tuned weights")
    other = _initialize(tmp_path, _config(shot=5))
    other_bytes = (Path(other["run_dir"]) / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        _initialize(tmp_path)
    fresh = _initialize(tmp_path, overwrite=True)
    assert fresh["status"] == "initialized" and fresh["weights_path"] == "yolo11n-seg.pt"
    assert not stale.exists()
    assert (Path(other["run_dir"]) / "manifest.json").read_bytes() == other_bytes
    (root / "manifest.json").unlink()
    (root / "user-owned.txt").write_text("preserve")
    with pytest.raises(ValueError, match="manifest|unmanaged"):
        _initialize(tmp_path, overwrite=True)
    assert (root / "user-owned.txt").read_text() == "preserve"
    with pytest.raises(ValueError, match="resume.*overwrite"):
        _initialize(tmp_path, overwrite=True, resume=True)


def test_run_and_summary_reject_kaggle_input_outputs() -> None:
    api = _api()
    with pytest.raises(ValueError, match="input"):
        api.initialize_run(_config(), Path("/kaggle/input/runs"), prepared_manifest=_provenance())
    with pytest.raises(ValueError, match="input"):
        api.upsert_summary(Path("/kaggle/input/results"), _row(_config()))


def test_failed_status_upsert(tmp_path: Path) -> None:
    api = _api()
    results = tmp_path / "results"
    row = _row(_config())
    row["box_map"] = 0.5
    api.upsert_summary(results, row)
    row["error_message"] = "retry also failed"
    api.upsert_summary(results, row)
    with (results / "summary.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames) == api.SUMMARY_FIELDS
        actual = list(reader)
    assert len(actual) == 1 and actual[0]["error_message"] == "retry also failed"
    assert all(actual[0][field] == "" for field in ("box_map", "box_map50", "box_map75", "seg_map", "seg_map50", "seg_map75"))


def test_summary_preserves_distinct_config_hashes(tmp_path: Path) -> None:
    api = _api()
    api.upsert_summary(tmp_path, _row(_config()))
    api.upsert_summary(tmp_path, _row(_config(epochs=200)))
    with (tmp_path / "summary.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2 and len({row["config_hash"] for row in rows}) == 2


def test_csv_failure_preserves_previous_summary(tmp_path: Path, monkeypatch) -> None:
    api = _api()
    api.upsert_summary(tmp_path, _row(_config()))
    before = (tmp_path / "summary.csv").read_bytes()
    original = Path.replace

    def replace_file(source, target):
        if Path(target).name == "summary.csv":
            raise OSError("synthetic replace failure")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace_file)
    with pytest.raises(OSError, match="replace failure"):
        api.upsert_summary(tmp_path, _row(_config(epochs=200)))
    assert (tmp_path / "summary.csv").read_bytes() == before


def test_prepared_data_hash_tracks_bytes_sources_and_ignores_root_and_timestamp(tmp_path: Path) -> None:
    from test_prepare import _fixture, _prepare
    api = _api()
    left = _fixture(tmp_path / "left")
    right = _fixture(tmp_path / "right")
    _prepare([1], left)
    _prepare([1], right)
    left_shot = left.prepared_root / "shot_1"
    right_shot = right.prepared_root / "shot_1"
    before = api.prepared_data_sha256(left_shot)
    assert before == api.prepared_data_sha256(right_shot)
    label = left.prepared_root / "test/labels/test.txt"
    original = label.read_bytes()
    label.write_text("0 0 0 0.25 0 0.25 0.25\n")
    assert api.prepared_data_sha256(left_shot) != before
    label.write_bytes(original)
    manifest_path = left_shot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_jsons"][0]["sha256"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert api.prepared_data_sha256(left_shot) != before
    weights = tmp_path / "base.pt"
    weights.write_bytes(b"abc")
    assert api.sha256_file(weights) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("defect", ["missing", "shot", "counts", "mapping_path", "output_path", "sources", "sha256", "source_path"])
def test_manifest_rejects_missing_or_incompatible_prepared_provenance(tmp_path: Path, defect: str) -> None:
    provenance = _provenance()
    if defect == "missing":
        provenance = {}
    elif defect == "shot":
        provenance["shot"] = 5
    elif defect == "counts":
        provenance["annotation_count"] = True
    elif defect in ("mapping_path", "output_path"):
        provenance["category_mapping_path" if defect == "mapping_path" else "output_path"] = ""
    elif defect == "sources":
        provenance["source_jsons"] = []
    elif defect == "sha256":
        provenance["source_jsons"][0]["sha256"] = "not-a-sha256"
    else:
        provenance["source_jsons"][0]["path"] = ""
    with pytest.raises(ValueError, match="provenance"):
        _api().initialize_run(_config(), tmp_path / "runs", prepared_manifest=provenance)
    assert not (tmp_path / "runs").exists()


def test_resume_rejects_saved_manifest_with_missing_prepared_provenance(tmp_path: Path) -> None:
    api = _api()
    manifest = _initialize(tmp_path)
    root = Path(manifest["run_dir"])
    manifest = api.transition_run(root, "running")
    checkpoint = root / "weights/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"own checkpoint")
    manifest["prepared_dataset"] = {}
    (root / "manifest.json").write_text(json.dumps(manifest))
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="provenance"):
        _initialize(tmp_path, resume=True)
    assert (root / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("schema_version", 0), ("schema_version", True), ("image_count", 0),
    ("annotation_count", 0), ("generated_label_count", 0), ("label_file_count", 0),
    ("taxonomy", {}), ("taxonomy", {"names": ["Bat"], "category_to_index": {"5": 4}}),
    ("timestamp_utc", "invalid"), ("timestamp_utc", "2026-10-04T00:00:00"),
    ("versions", {}), ("versions", {"python": 313}),
])
def test_run_manifest_rejects_invalid_t04_provenance_fields(tmp_path: Path, field: str, value) -> None:
    provenance = _provenance()
    provenance[field] = value
    with pytest.raises(ValueError, match="provenance"):
        _api().build_manifest(_config(), tmp_path, prepared_manifest=provenance)


@pytest.mark.parametrize("field", ["schema_version", "label_file_count", "taxonomy", "timestamp_utc", "versions"])
def test_run_manifest_requires_t04_provenance_fields(tmp_path: Path, field: str) -> None:
    provenance = _provenance()
    del provenance[field]
    with pytest.raises(ValueError, match="provenance"):
        _api().build_manifest(_config(), tmp_path, prepared_manifest=provenance)


def test_run_manifest_accepts_actual_synthetic_t04_provenance(tmp_path: Path) -> None:
    from test_prepare import _fixture, _prepare
    paths = _fixture(tmp_path)
    _prepare([1], paths)
    provenance = json.loads((paths.prepared_root / "shot_1/manifest.json").read_text())
    config = _api().RunConfig(weights_sha256=WEIGHTS_SHA,
                             data_sha256=_api().prepared_data_sha256(paths.prepared_root / "shot_1"))
    manifest = _api().build_manifest(config, paths.runs_root, prepared_manifest=provenance)
    assert manifest["prepared_dataset"] == provenance


@pytest.mark.parametrize("field,value", [("fliplr", 4.0), ("mosaic", -3.0), ("hsv_h", 2.0)])
def test_augmentation_rejects_out_of_range_probabilities(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="range|probability"):
        _api().AugmentationConfig(**{field: value})
