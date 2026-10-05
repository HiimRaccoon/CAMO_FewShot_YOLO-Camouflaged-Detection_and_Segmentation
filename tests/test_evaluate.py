"""T11 evaluation uses synthetic preparation and a native validator double."""

import csv
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from camo_fs.runs import (METRIC_FIELDS, RunConfig, initialize_run, prepared_data_sha256,
                          run_path, sha256_file, transition_run)
from test_prepare import _fixture, _prepare, _snapshot


EXPECTED = dict(box_map=0.12, box_map50=0.34, box_map75=0.23,
                seg_map=0.45, seg_map50=0.67, seg_map75=0.56)


class FakeModel:
    def __init__(self, runtime):
        self.runtime = runtime
        self.names = runtime.names
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def val(self, **options):
        self.runtime.calls.append(options)
        data = json.loads(Path(options["data"]).read_text())
        validator = SimpleNamespace(args=SimpleNamespace(**options), data=data,
                                    save_dir=Path(options["project"]) / options["name"])
        self.runtime.mutate(validator)
        for callback in self.callbacks.get("on_val_start", []):
            callback(validator)
        if self.runtime.error:
            raise RuntimeError(self.runtime.error)
        return self.runtime.metrics


class FakeRuntime:
    def __init__(self, *, error="", mutate=lambda validator: None):
        self.error, self.mutate = error, mutate
        self.loads, self.calls, self.seeds = [], [], []
        self.names = {0: "Bat", 1: "Fox"}
        self.metrics = SimpleNamespace(box=SimpleNamespace(map=0.12, map50=0.34, map75=0.23),
                                       seg=SimpleNamespace(map=0.45, map50=0.67, map75=0.56))

    def load_inference(self, checkpoint):
        self.loads.append(checkpoint)
        return FakeModel(self)

    def seed(self, seed, deterministic):
        self.seeds.append((seed, deterministic))
        return []


def _api():
    return importlib.import_module("camo_fs.evaluate")


def _setup(tmp_path, shots=(1,), *, image_size=4):
    paths = _fixture(tmp_path)
    if image_size != 4:
        from PIL import Image

        for path in paths.images_dir.glob("*.png"):
            Image.new("RGB", (image_size, image_size), (128, 128, 128)).save(path)
        for path in paths.annotations_root.rglob("*.json"):
            document = json.loads(path.read_text())
            for image in document["images"]:
                image.update(width=image_size, height=image_size)
            for annotation in document["annotations"]:
                annotation["bbox"] = [value * image_size / 4 for value in annotation["bbox"]]
                annotation["segmentation"] = [[value * image_size / 4 for value in polygon]
                                               for polygon in annotation["segmentation"]]
            path.write_text(json.dumps(document))
    _prepare(list(shots), paths, val_train_placeholder=True)
    weights = tmp_path / "base.pt"
    weights.write_bytes(b"synthetic base weights")
    return paths, RunConfig(weights=str(weights), weights_sha256=sha256_file(weights),
                            data_sha256=prepared_data_sha256(paths.prepared_root / f"shot_{shots[0]}"),
                            shot=shots[0], epochs=2, device="cpu")


def _run(paths, config, *, status="completed"):
    provenance = json.loads((paths.prepared_root / f"shot_{config.shot}/manifest.json").read_text())
    initialize_run(config, paths.runs_root, prepared_manifest=provenance)
    root = run_path(config, paths.runs_root)
    transition_run(root, "running")
    if status == "completed":
        transition_run(root, "completed")
    elif status == "failed":
        transition_run(root, "failed", error_message="training broke")
    (root / "weights").mkdir()
    (root / "weights/last.pt").write_bytes(b"synthetic completed checkpoint")
    (root / "weights/best.pt").write_bytes(b"must never load best")
    return root


def _rows(paths):
    with (paths.results_root / "summary.csv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
def test_evaluates_only_own_last_on_test_and_extracts_all_six_metrics(tmp_path, method):
    paths, config = _setup(tmp_path)
    root = _run(paths, replace(config, method=method))
    source_before = _snapshot(paths.data_root)
    run_before = _snapshot(root)
    runtime = FakeRuntime()
    row = _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml",
                               paths.results_root, runtime=runtime)
    assert runtime.loads == [root / "weights/last.pt"]
    assert len(runtime.calls) == 1 and runtime.calls[0]["split"] == "test"
    assert runtime.calls[0]["data"] == str(paths.prepared_root / "shot_1/data.yaml")
    assert runtime.calls[0]["imgsz"] == 640 and runtime.calls[0]["device"] == "cpu"
    assert {key: row[key] for key in METRIC_FIELDS} == EXPECTED
    assert row["status"] == "completed" and row["error_message"] == ""
    saved = _rows(paths)
    assert len(saved) == 1 and {key: float(saved[0][key]) for key in METRIC_FIELDS} == EXPECTED
    assert _snapshot(paths.data_root) == source_before
    assert all((root / name).read_bytes() == content for name, content in run_before.items())


@pytest.mark.parametrize("failure", ["missing last", "validator exception", "failed", "running"])
def test_evaluation_failure_is_visible_with_empty_metrics_and_untouched_training_state(tmp_path, failure):
    paths, config = _setup(tmp_path)
    root = _run(paths, config, status=failure if failure in ("failed", "running") else "completed")
    if failure == "missing last":
        (root / "weights/last.pt").unlink()
    runtime = FakeRuntime(error="validator broke" if failure == "validator exception" else "")
    before = (root / "manifest.json").read_bytes()
    row = _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml",
                               paths.results_root, runtime=runtime)
    assert row["status"] == "failed" and row["error_message"]
    assert all(row.get(key) is None for key in METRIC_FIELDS)
    saved = _rows(paths)
    assert saved[0]["status"] == "failed" and all(saved[0][key] == "" for key in METRIC_FIELDS)
    assert (root / "manifest.json").read_bytes() == before
    if failure != "validator exception":
        assert runtime.loads == [] and runtime.calls == []


@pytest.mark.parametrize("part,field,value", [("box", "map", float("nan")),
                                                ("seg", "map50", float("inf")),
                                                ("seg", "map75", -0.1), ("box", "map50", 1.1),
                                                ("seg", "map75", None)])
def test_invalid_or_missing_metrics_fail_without_partial_scores(tmp_path, part, field, value):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    runtime = FakeRuntime()
    if value is None:
        delattr(getattr(runtime.metrics, part), field)
    else:
        setattr(getattr(runtime.metrics, part), field, value)
    row = _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml",
                               paths.results_root, runtime=runtime)
    assert row["status"] == "failed" and row["error_message"]
    assert all(_rows(paths)[0][key] == "" for key in METRIC_FIELDS)


@pytest.mark.parametrize("change", ["val split", "test is train", "different shot YAML", "changed test bytes"])
def test_wrong_split_or_changed_prepared_data_rejected_before_model_load(tmp_path, change):
    paths, config = _setup(tmp_path, shots=(1, 2))
    root = _run(paths, config)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    kwargs = {}
    if change == "val split":
        kwargs["split"] = "val"
    elif change == "test is train":
        document = json.loads(yaml.read_text())
        document["test"] = document["train"]
        yaml.write_text(json.dumps(document))
    elif change == "different shot YAML":
        yaml = paths.prepared_root / "shot_2/data.yaml"
    else:
        (paths.prepared_root / "test/labels/test.txt").write_text("0 0 0 1 0 1 1\n")
    runtime = FakeRuntime()
    row = _api().evaluate_one(root, yaml, paths.results_root, runtime=runtime, **kwargs)
    assert row["status"] == "failed" and row["error_message"]
    assert runtime.loads == [] and runtime.calls == []


@pytest.mark.parametrize("mutation", ["split", "test", "save_dir"])
def test_native_validator_cannot_redirect_split_test_or_output(tmp_path, mutation):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    def mutate(validator):
        if mutation == "split":
            validator.args.split = "val"
        elif mutation == "test":
            validator.data["test"] = validator.data["train"]
        else:
            validator.save_dir = paths.data_root
    runtime = FakeRuntime(mutate=mutate)
    row = _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml",
                               paths.results_root, runtime=runtime)
    assert row["status"] == "failed" and row["error_message"]
    assert all(_rows(paths)[0][key] == "" for key in METRIC_FIELDS)


def test_checkpoint_taxonomy_must_match_saved_mapping(tmp_path):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    runtime = FakeRuntime()
    runtime.names = {0: "Fox", 1: "Bat"}
    row = _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml",
                               paths.results_root, runtime=runtime)
    assert row["status"] == "failed" and "taxonomy" in row["error_message"].lower()
    assert runtime.calls == []


@pytest.mark.parametrize("destination", ["source", "outside work", "prepared"])
def test_summary_destination_cannot_write_outside_results_work_area(tmp_path, destination):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    target = {"source": paths.data_root, "outside work": tmp_path / "elsewhere",
              "prepared": paths.prepared_root}[destination]
    runtime = FakeRuntime()
    with pytest.raises(ValueError):
        _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml", target, runtime=runtime)
    assert not (target / "summary.csv").exists()
    assert runtime.loads == []


def test_relocated_run_manifest_cannot_claim_another_run_identity(tmp_path):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    copied = paths.runs_root / "impostor"
    copied.mkdir()
    (copied / "manifest.json").write_bytes((root / "manifest.json").read_bytes())
    (copied / "weights").mkdir()
    (copied / "weights/last.pt").write_bytes(b"wrong checkpoint")
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="run"):
        _api().evaluate_one(copied, paths.prepared_root / "shot_1/data.yaml",
                            paths.results_root, runtime=runtime)
    assert runtime.loads == [] and not (paths.results_root / "summary.csv").exists()


def test_benchmark_options_do_not_reuse_smoke_confidence_or_augmentations(tmp_path):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    runtime = FakeRuntime()
    _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml", paths.results_root, runtime=runtime)
    options = runtime.calls[0]
    assert options["conf"] == 0.001 and options["iou"] == 0.7 and options["max_det"] == 300
    assert options["augment"] is False and options["single_cls"] is False
    assert options["overlap_mask"] is False and options["save_json"] is False
    assert options["plots"] is False and options["rect"] is True
    assert runtime.seeds == [(2024, True)]


def test_repeat_evaluation_upserts_scores_then_failure_then_recovery(tmp_path):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    runtime = FakeRuntime()
    yaml = paths.prepared_root / "shot_1/data.yaml"
    first = _api().evaluate_one(root, yaml, paths.results_root, runtime=runtime)
    assert first["weights_path"] == str(root / "weights/last.pt")
    runtime.metrics.box.map75 = 0.89
    _api().evaluate_one(root, yaml, paths.results_root, runtime=runtime)
    assert len(_rows(paths)) == 1 and float(_rows(paths)[0]["box_map75"]) == 0.89
    runtime.error = "evaluation interrupted"
    _api().evaluate_one(root, yaml, paths.results_root, runtime=runtime)
    assert len(_rows(paths)) == 1 and all(_rows(paths)[0][key] == "" for key in METRIC_FIELDS)
    runtime.error = ""
    _api().evaluate_one(root, yaml, paths.results_root, runtime=runtime)
    assert _rows(paths)[0]["status"] == "completed" and _rows(paths)[0]["error_message"] == ""


def test_upsert_preserves_distinct_config_hashes(tmp_path):
    paths, config = _setup(tmp_path, shots=(5,))
    roots = [_run(paths, replace(config, epochs=epochs)) for epochs in (2, 3)]
    for root in roots:
        _api().evaluate_one(root, paths.prepared_root / "shot_5/data.yaml",
                            paths.results_root, runtime=FakeRuntime())
    rows = _rows(paths)
    assert len(rows) == 2 and len({row["config_hash"] for row in rows}) == 2
    assert {row["epochs"] for row in rows} == {"2", "3"}


def test_valid_zero_metrics_are_reported_as_completed(tmp_path):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    runtime = FakeRuntime()
    runtime.metrics = SimpleNamespace(box=SimpleNamespace(map=0, map50=0, map75=0),
                                       seg=SimpleNamespace(map=0, map50=0, map75=0))
    row = _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml",
                               paths.results_root, runtime=runtime)
    assert row["status"] == "completed" and all(row[key] == 0 for key in METRIC_FIELDS)


def test_discovery_all_shots_preserves_every_completed_benchmark_hash_and_excludes_smoke(tmp_path):
    paths, config = _setup(tmp_path, shots=(1, 2, 3, 5))
    expected = []
    for shot in (1, 2, 3, 5):
        selected = replace(config, shot=shot,
                            data_sha256=prepared_data_sha256(paths.prepared_root / f"shot_{shot}"))
        expected.append(_run(paths, selected))
    expected.append(_run(paths, replace(selected, epochs=3)))
    smoke = _run(paths, replace(config, run_kind="smoke"))
    _run(paths, replace(config, method="fgbg-triplet"))
    _run(paths, replace(config, seed=7), status="failed")
    _run(paths, replace(config, seed=8), status="running")
    found = _api().discover_runs(paths.runs_root, method="baseline", shots=(1, 2, 3, 5))
    assert set(found) == set(expected)
    assert [json.loads((root / "manifest.json").read_text())["shot"] for root in found] == [1, 2, 3, 5, 5]
    assert set(_api().discover_runs(paths.runs_root, method="baseline", shots=(1,), include_smoke=True)) == {expected[0], smoke}
    assert _api().discover_runs(paths.runs_root, method="fgbg-triplet", shots=(2,)) == []


@pytest.mark.parametrize("continue_on_error,expected_count", [(False, 1), (True, 2)])
def test_selected_evaluation_stops_on_failure_unless_explicitly_continued(tmp_path, continue_on_error, expected_count):
    paths, config = _setup(tmp_path, shots=(1, 2))
    first = _run(paths, config)
    second = _run(paths, replace(config, shot=2,
                                 data_sha256=prepared_data_sha256(paths.prepared_root / "shot_2")))
    (first / "weights/last.pt").unlink()
    runtime = FakeRuntime()
    outcomes = _api().evaluate_selected((1, 2), paths, method="baseline", runtime=runtime,
                                         continue_on_error=continue_on_error)
    assert len(outcomes) == expected_count and outcomes[0]["status"] == "failed"
    assert len(_rows(paths)) == expected_count
    if continue_on_error:
        assert outcomes[1]["status"] == "completed" and runtime.loads == [second / "weights/last.pt"]


def test_cli_all_selects_all_completed_hashes_without_loading_smoke(tmp_path, capsys):
    paths, config = _setup(tmp_path, shots=(1, 2, 3, 5))
    roots = []
    for shot in (1, 2, 3, 5):
        selected = replace(config, method="fgbg-triplet", shot=shot,
                            data_sha256=prepared_data_sha256(paths.prepared_root / f"shot_{shot}"))
        roots.append(_run(paths, selected))
    roots.append(_run(paths, replace(selected, triplet_weight=0.2)))
    _run(paths, replace(selected, run_kind="smoke"))
    runtime = FakeRuntime()
    cli = importlib.import_module("scripts.evaluate_yolo")
    result = cli.main(["--shot", "all", "--method", "fgbg-triplet", "--data-root", str(paths.data_root),
                       "--work-root", str(paths.work_root)], runtime=runtime)
    assert result == 0 and len(_rows(paths)) == 5
    assert set(runtime.loads) == {root / "weights/last.pt" for root in roots}
    output = capsys.readouterr().out
    assert output.count("completed") == 5 and roots[-1].name in output


def test_cli_reports_empty_selection_and_unsupported_flags_without_runtime_load(tmp_path, capsys):
    paths, _ = _setup(tmp_path)
    runtime = FakeRuntime()
    cli = importlib.import_module("scripts.evaluate_yolo")
    args = ["--shot", "1", "--work-root", str(paths.work_root), "--data-root", str(paths.data_root)]
    assert cli.main(args, runtime=runtime) == 1
    assert "No completed" in capsys.readouterr().err
    assert runtime.loads == []
    for flag in (["--split", "val"], ["--weights", "best.pt"], ["--conf", "0.0001"]):
        with pytest.raises(SystemExit) as error:
            cli.main([*args, *flag], runtime=runtime)
        assert error.value.code == 2


@pytest.mark.parametrize("include_smoke,expected", [(False, 1), (True, 0)])
def test_cli_smoke_requires_explicit_opt_in_and_preserves_run_kind(tmp_path, include_smoke, expected):
    paths, config = _setup(tmp_path)
    _run(paths, replace(config, run_kind="smoke"))
    cli = importlib.import_module("scripts.evaluate_yolo")
    args = ["--shot", "1", "--work-root", str(paths.work_root), "--data-root", str(paths.data_root)]
    assert cli.main([*args, *(["--include-smoke"] if include_smoke else [])], runtime=FakeRuntime()) == expected
    if include_smoke:
        assert _rows(paths)[0]["run_kind"] == "smoke"


def test_cli_single_run_uses_manifest_shot_and_failed_result_returns_nonzero(tmp_path):
    paths, config = _setup(tmp_path, shots=(5,))
    root = _run(paths, config)
    cli = importlib.import_module("scripts.evaluate_yolo")
    args = ["--run-dir", str(root), "--work-root", str(paths.work_root), "--data-root", str(paths.data_root)]
    runtime = FakeRuntime(error="inference failed")
    assert cli.main(args, runtime=runtime) == 1
    assert runtime.calls[0]["data"] == str(paths.prepared_root / "shot_5/data.yaml")
    assert _rows(paths)[0]["status"] == "failed"


@pytest.mark.integration
@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
def test_native_pinned_validator_evaluates_only_synthetic_test_and_returns_six_metrics(tmp_path, monkeypatch, method):
    """Native CPU compatibility proof, never an official-test benchmark result."""
    ultralytics = pytest.importorskip("ultralytics")
    if ultralytics.__version__ != "8.3.228":
        pytest.skip("Native evaluation proof requires the exact project pin 8.3.228")
    torch = pytest.importorskip("torch")
    from ultralytics.nn.tasks import SegmentationModel
    from camo_fs.train import UltralyticsRuntime
    from camo_fs.ultralytics_ext import FGSegmentationModel

    # Native verification requires images >=10 pixels; T04's tiny fixture is 4x4.
    # Font lookup does not affect AP and must not download outside the fixture.
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *args, **kwargs: None)
    paths, config = _setup(tmp_path, image_size=64)
    root = _run(paths, replace(config, method=method, imgsz=64, batch=1))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cls = FGSegmentationModel if method == "fgbg-triplet" else SegmentationModel
        model = cls("yolo11n-seg.yaml", nc=2, verbose=False)
        model.names = {0: "Bat", 1: "Fox"}
        if method == "fgbg-triplet":
            model.configure_triplet(weight=0.1, margin=0.3, count=16, seed=2024)
        torch.save({"model": model.half(), "epoch": -1, "optimizer": None,
                    "train_args": {"task": "segment", "imgsz": 64}}, root / "weights/last.pt")
        observed = []

        class ObservedRuntime(UltralyticsRuntime):
            def load_inference(self, checkpoint):
                loaded = super().load_inference(checkpoint)
                loaded.add_callback("on_val_start", lambda validator: observed.extend(validator.dataloader.dataset.im_files))
                return loaded

        row = _api().evaluate_one(root, paths.prepared_root / "shot_1/data.yaml",
                                   paths.results_root, runtime=ObservedRuntime())
        assert row["status"] == "completed", row["error_message"]
        assert all(isinstance(row[field], float) and 0 <= row[field] <= 1 for field in METRIC_FIELDS)
        assert observed == [str(paths.prepared_root / "test/images/test.png")]
        assert len(_rows(paths)) == 1
    finally:
        torch.set_num_threads(threads)
