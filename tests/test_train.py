"""T06 orchestration uses real synthetic preparation and an external YOLO double."""

import csv
from dataclasses import replace
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from camo_fs.runs import RunConfig, prepared_data_sha256, run_path, sha256_file, transition_run
from test_prepare import _fixture, _prepare, _snapshot


class FakeYOLO:
    def __init__(self, runtime, checkpoint):
        self.runtime, self.checkpoint = runtime, checkpoint
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, trainer=None, **options):
        self.runtime.calls.append((self.checkpoint, {**options, **({"trainer": trainer} if trainer else {})}))
        root = Path(options["project"]) / options["name"]
        data = json.loads(Path(options["data"]).read_text())
        self.trainer = SimpleNamespace(args=SimpleNamespace(**options), save_dir=root,
                                       last=root / "weights/last.pt", epoch=options["epochs"] - 1,
                                       epochs=options["epochs"], data={**data,
                                           "names": dict(enumerate(data["names"])), "nc": len(data["names"]), "channels": 3})
        self.runtime.mutate(self.trainer)
        for event in ("on_pretrain_routine_end", "on_train_epoch_start"):
            for callback in self.callbacks.get(event, []):
                callback(self.trainer)
        for batch in getattr(self.runtime, "batches", []):
            self.trainer.model = SimpleNamespace(triplet_metrics=batch["triplet"])
            self.trainer.loss_items = batch["native"]
            self.trainer.loss = batch["combined"]
            for callback in self.callbacks.get("on_train_batch_end", []):
                callback(self.trainer)
        if self.runtime.error:
            raise RuntimeError(self.runtime.error)
        if self.runtime.save:
            self.trainer.last.parent.mkdir(exist_ok=True)
            self.trainer.last.write_bytes(b"synthetic trained checkpoint")
            # Native label caches are generated alongside labels, not training data.
            (Path(data["train"]).parent / "labels.cache").write_bytes(b"synthetic loader cache")


class FakeRuntime:
    def __init__(self, *, error="", save=True, mutate=lambda trainer: None, defaults=None):
        self.error, self.save, self.mutate = error, save, mutate
        self.calls, self.loads, self.seeds = [], [], []
        self.defaults = dict(defaults or {})

    def training_defaults(self):
        return dict(self.defaults)

    def seed(self, seed, deterministic):
        self.seeds.append((seed, deterministic))
        return ["synthetic determinism warning"]

    def load(self, checkpoint, *, base):
        self.loads.append((checkpoint, base))
        return FakeYOLO(self, checkpoint)

    def validate_options(self, options):
        return None


def _setup(tmp_path, shots=(1,), **kwargs):
    paths = _fixture(tmp_path)
    _prepare(list(shots), paths, val_train_placeholder=True)
    weights = tmp_path / "yolo11n-seg.pt"
    weights.write_bytes(b"synthetic COCO base checkpoint")
    config = RunConfig(weights=str(weights), weights_sha256=sha256_file(weights),
                       data_sha256=prepared_data_sha256(paths.prepared_root / f"shot_{shots[0]}"),
                       shot=shots[0], epochs=2, **kwargs)
    return paths, config


def _api():
    return importlib.import_module("camo_fs.train")


def _rows(paths):
    with (paths.results_root / "summary.csv").open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_baseline_returns_own_last_and_records_native_protocol(tmp_path):
    # Break caught: official test forwarded as val, custom trainer activation,
    # dropped shared settings, or completion/summary absent after native training.
    paths, config = _setup(tmp_path)
    source_before = _snapshot(paths.data_root)
    runtime = FakeRuntime()
    last = _api().train_one(config, paths, resume=False, overwrite=False, runtime=runtime)
    root = run_path(config, paths.runs_root)
    assert last == root / "weights/last.pt" and last.is_file()
    assert runtime.loads == [(Path(config.weights), True)]
    assert runtime.seeds == [(2024, True)]
    options = runtime.calls[0][1]
    assert options["data"] == str(paths.prepared_root / "shot_1/data.yaml")
    assert {key: options[key] for key in ("val", "overlap_mask", "mosaic", "mixup", "copy_paste")} == {
        "val": False, "overlap_mask": False, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0}
    assert options["epochs"] == 2 and options["seed"] == 2024 and options["deterministic"] is True
    assert options["patience"] == 0 and options["time"] is None and options["save"] is True
    assert options["resume"] is False and "trainer" not in options and "triplet_weight" not in options
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["auxiliary_loss"] is None
    assert manifest["effective_training_options"]["epochs"] == 2
    assert "synthetic determinism warning" in manifest["warnings"]
    assert _rows(paths)[0]["status"] == "completed" and _rows(paths)[0]["box_map"] == ""
    assert _snapshot(paths.data_root) == source_before


@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
@pytest.mark.parametrize("val", ["missing", None, ""])
def test_missing_val_rejected_before_loading_or_creating_run(tmp_path, method, val):
    # Break caught: missing val lets native final validation fall back to the
    # official test split even with val=False.
    paths, config = _setup(tmp_path, method=method)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    document = json.loads(yaml.read_text())
    if val == "missing":
        document.pop("val")
    else:
        document["val"] = val
    yaml.write_text(json.dumps(document))
    if val == "missing":
        config = replace(config, data_sha256=prepared_data_sha256(yaml.parent))
    runtime = FakeRuntime() if method == "baseline" else EnhancedRuntime()
    before = _snapshot(paths.prepared_root)
    with pytest.raises(ValueError, match="TRAIN placeholder|missing val"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert runtime.loads == [] and runtime.calls == [] and runtime.seeds == []
    assert not run_path(config, paths.runs_root).exists()
    assert _snapshot(paths.prepared_root) == before
    assert _rows(paths)[0]["status"] == "failed"


@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
@pytest.mark.parametrize("field,target", [("val", "test/images"), ("test", "shot_1/train/images")])
def test_yaml_split_misrouting_rejected_before_loading_or_creating_run(tmp_path, method, field, target):
    paths, config = _setup(tmp_path, method=method)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    document = json.loads(yaml.read_text())
    document[field] = str(paths.prepared_root / target)
    yaml.write_text(json.dumps(document))
    runtime = FakeRuntime() if method == "baseline" else EnhancedRuntime()
    before = _snapshot(paths.prepared_root)
    with pytest.raises(ValueError, match="train placeholder|shared official test"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert runtime.loads == [] and runtime.calls == [] and runtime.seeds == []
    assert not run_path(config, paths.runs_root).exists()
    assert _snapshot(paths.prepared_root) == before
    assert _rows(paths)[0]["status"] == "failed"


@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
@pytest.mark.parametrize("val", ["missing", None])
def test_parsed_trainer_missing_val_cannot_pass_runtime_guard(tmp_path, method, val):
    # Break caught: a native parser/resume drops the preflighted val key and the
    # runtime guard treats absence/null as permission to use official test.
    paths, config = _setup(tmp_path, method=method)

    def mutate(trainer):
        if val == "missing":
            trainer.data.pop("val")
        else:
            trainer.data["val"] = val

    runtime = FakeRuntime(mutate=mutate) if method == "baseline" else EnhancedRuntime(mutate=mutate)
    with pytest.raises(ValueError, match="TRAIN placeholder"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert _rows(paths)[0]["status"] == "failed"
    assert not (run_path(config, paths.runs_root) / "weights/last.pt").exists()


@pytest.mark.parametrize("failure", ["runtime", "missing_last", "short_run"])
def test_incomplete_training_records_failed_attempt_without_metrics(tmp_path, failure):
    # Break caught: exception or early return leaves a running/completed manifest.
    paths, config = _setup(tmp_path)
    runtime = FakeRuntime(error="synthetic CUDA error" if failure == "runtime" else "",
                          save=failure != "missing_last",
                          mutate=(lambda trainer: setattr(trainer, "epoch", 0)) if failure == "short_run" else lambda trainer: None)
    with pytest.raises((ValueError, RuntimeError), match="CUDA|last.pt|epochs"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    manifest = json.loads((run_path(config, paths.runs_root) / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error_message"]
    row = _rows(paths)[0]
    assert row["status"] == "failed" and row["error_message"]
    assert all(row[name] == "" for name in ("box_map", "box_map50", "box_map75", "seg_map", "seg_map50", "seg_map75"))


@pytest.mark.parametrize("change", ["val", "mosaic", "epochs", "data", "save_dir", "last", "val_data"])
def test_trainer_cannot_silently_change_protocol_or_paths(tmp_path, change):
    # Break caught: native resume/defaults override the hashed config or read test as val.
    paths, config = _setup(tmp_path)

    def mutate(trainer):
        if change == "val_data":
            trainer.data["val"] = str(paths.prepared_root / "test/images")
        elif change in ("save_dir", "last"):
            setattr(trainer, change, tmp_path / "unexpected")
        else:
            setattr(trainer.args, change, {"val": True, "mosaic": 1.0, "epochs": 3,
                                         "data": "other.yaml"}[change])

    runtime = FakeRuntime(mutate=mutate)
    with pytest.raises(ValueError, match="effective|trainer|validation|own"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert _rows(paths)[0]["status"] == "failed"
    assert not (run_path(config, paths.runs_root) / "weights/last.pt").exists()


@pytest.mark.parametrize("options", [{"project": "elsewhere"}, {"cfg": "unsafe.yaml"},
                                     {"fraction": 0.5}, {"single_cls": True}, {"save": False}])
def test_extra_options_cannot_override_project_protocol(tmp_path, options):
    # Break caught: scalar escape hatch changes support-set membership or output routing.
    paths, config = _setup(tmp_path, training_options=options)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="protocol|option"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert runtime.loads == [] and runtime.calls == []


class EnhancedRuntime(FakeRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batches = [{"triplet": {"raw_triplet": 0.25, "weighted_triplet": 0.1,
                                    "sampled_triplets": 16, "skipped_instances": 1},
                         "native": [1.0, 2.0, 3.0, 4.0], "combined": 40.1}]

    def enhanced_trainer(self, config):
        from functools import partial

        return partial(SimpleNamespace, triplet_weight=config.triplet_weight,
                       triplet_margin=config.triplet_margin,
                       triplets_per_instance=config.triplets_per_instance)


def test_enhanced_cli_dispatches_custom_trainer_without_native_triplet_overrides(tmp_path):
    # Break caught: enhanced CLI remains gated or silently uses native baseline.
    paths, config = _setup(tmp_path, method="fgbg-triplet")
    runtime = EnhancedRuntime()
    result = _cli().main(["--shot", "1", "--method", "fgbg-triplet", "--weights", config.weights,
                         "--data-root", str(paths.data_root), "--work-root", str(paths.work_root),
                         "--epochs", "2", "--triplet-weight", "0.2", "--triplet-margin", "0.4",
                         "--triplets-per-instance", "7"], runtime=runtime)
    assert result == 0
    options = runtime.calls[0][1]
    bound = options["trainer"]()
    assert vars(bound) == {"triplet_weight": 0.2, "triplet_margin": 0.4, "triplets_per_instance": 7}
    assert not {"triplet_weight", "triplet_margin", "triplets_per_instance"} & options.keys()
    assert _rows(paths)[0]["method"] == "fgbg-triplet"


@pytest.mark.parametrize("device", ["cpu", "0,1", "1", "-1", "[0,1]"])
def test_enhanced_rejects_devices_outside_single_gpu_zero_before_loading(tmp_path, device):
    paths, config = _setup(tmp_path, method="fgbg-triplet", device=device)
    runtime = EnhancedRuntime()
    with pytest.raises(ValueError, match="single|device"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert runtime.loads == [] and runtime.calls == []


def test_dispatched_methods_keep_non_method_options_equal_and_baseline_bypasses_extension(tmp_path):
    paths, config = _setup(tmp_path)

    class BaselineRuntime(FakeRuntime):
        def enhanced_trainer(self, config):
            raise AssertionError("baseline touched enhanced factory")

    native, enhanced = BaselineRuntime(), EnhancedRuntime()
    _api().train_one(config, paths, False, False, runtime=native)
    _api().train_one(replace(config, method="fgbg-triplet"), paths, False, False, runtime=enhanced)
    excluded = {"trainer", "project", "name"}
    assert {key: value for key, value in native.calls[0][1].items() if key not in excluded} == {
        key: value for key, value in enhanced.calls[0][1].items() if key not in excluded}
    assert "trainer" not in native.calls[0][1]


def test_enhanced_resume_keeps_custom_dispatch_and_own_checkpoint(tmp_path):
    paths, config = _setup(tmp_path, method="fgbg-triplet")
    root = run_path(config, paths.runs_root)
    with pytest.raises(RuntimeError, match="interrupted"):
        _api().train_one(config, paths, False, False, runtime=EnhancedRuntime(error="interrupted"))
    (root / "weights").mkdir()
    (root / "weights/last.pt").write_bytes(b"synthetic interrupted enhanced checkpoint")
    runtime = EnhancedRuntime()
    _api().train_one(config, paths, True, False, runtime=runtime)
    assert runtime.loads == [(root / "weights/last.pt", False)]
    assert "trainer" in runtime.calls[0][1]
    assert runtime.calls[0][1]["resume"] == str(root / "weights/last.pt")


def test_enhanced_records_finite_batch_losses_and_accumulated_triplet_counts(tmp_path):
    # Break caught: last-batch metrics are lost or native items get overwritten
    # instead of recording a separate auxiliary logging channel.
    paths, config = _setup(tmp_path, method="fgbg-triplet", run_kind="smoke")
    runtime = EnhancedRuntime()
    runtime.batches *= 2
    _api().train_one(config, paths, False, False, runtime=runtime)
    root = run_path(config, paths.runs_root)
    records = [json.loads(line) for line in (root / "triplet_batches.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["native_losses"] == [1.0, 2.0, 3.0, 4.0]
    assert records[0]["combined_loss"] == 40.1
    assert records[0]["raw_triplet"] == 0.25 and records[0]["weighted_triplet"] == 0.1
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["triplet_training"] == {"batches": 2, "nonzero_raw_batches": 2,
                                            "sampled_triplets": 32, "skipped_instances": 2}
    assert manifest["auxiliary_loss"]["feature_source"] == "Segment head first spatial input (P3/8)"
    assert manifest["auxiliary_loss"]["sampled_triplets"] == 32
    assert manifest["auxiliary_loss"]["skipped_instances"] == 2
    assert "cuda" in manifest["versions"]


@pytest.mark.parametrize("field", ["native", "combined", "raw_triplet", "weighted_triplet"])
def test_enhanced_nonfinite_batch_fails_without_reporting_completion(tmp_path, field):
    paths, config = _setup(tmp_path, method="fgbg-triplet")
    runtime = EnhancedRuntime()
    if field == "native":
        runtime.batches[0][field][2] = float("nan")
    elif field == "combined":
        runtime.batches[0][field] = float("inf")
    else:
        runtime.batches[0]["triplet"][field] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert _rows(paths)[0]["status"] == "failed"
    assert not (run_path(config, paths.runs_root) / "weights/last.pt").exists()


@pytest.mark.parametrize("failure", ["weights", "data", "labels", "source", "taxonomy"])
def test_preflight_failure_is_recorded_before_loading_or_creating_run(tmp_path, failure):
    # Break caught: matching a caller's digest bypasses T04 checksums/source integrity.
    paths, config = _setup(tmp_path)
    if failure == "weights":
        Path(config.weights).unlink()
    elif failure in ("data", "labels"):
        (paths.prepared_root / "shot_1/train/labels/bat_1.txt").write_text("0 0 0\n")
        if failure == "labels":
            config = replace(config, data_sha256=prepared_data_sha256(paths.prepared_root / "shot_1"))
    elif failure == "source":
        source = paths.few_shot_dir / "camo5_Bat_1shot_split1.json"
        source.write_text(source.read_text() + "\n")
    else:
        yaml = paths.prepared_root / "shot_1/data.yaml"
        doc = json.loads(yaml.read_text())
        doc["names"] = ["wrong", "taxonomy"]
        yaml.write_text(json.dumps(doc))
        config = replace(config, data_sha256=prepared_data_sha256(paths.prepared_root / "shot_1"))
    runtime = FakeRuntime()
    with pytest.raises((ValueError, OSError), match="checksum|checkpoint|taxonomy|integrity"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert runtime.loads == []
    assert not run_path(config, paths.runs_root).exists()
    assert _rows(paths)[0]["status"] == "failed"


def test_existing_run_is_preserved_and_rejected_before_model_loading(tmp_path):
    paths, config = _setup(tmp_path)
    _api().train_one(config, paths, False, False, runtime=FakeRuntime())
    before = _snapshot(paths.runs_root)
    summary = (paths.results_root / "summary.csv").read_bytes()
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="exists"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert runtime.loads == [] and _snapshot(paths.runs_root) == before
    assert (paths.results_root / "summary.csv").read_bytes() == summary


def test_resume_loads_only_interrupted_runs_own_checkpoint(tmp_path):
    paths, config = _setup(tmp_path)
    root = run_path(config, paths.runs_root)
    runtime = FakeRuntime(error="interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    (root / "weights").mkdir()
    (root / "weights/last.pt").write_bytes(b"synthetic interrupted checkpoint")
    resumed = FakeRuntime()
    assert _api().train_one(config, paths, True, False, runtime=resumed) == root / "weights/last.pt"
    assert resumed.loads == [(root / "weights/last.pt", False)]
    assert resumed.calls[0][1]["resume"] == str(root / "weights/last.pt")
    assert len(_rows(paths)) == 1 and _rows(paths)[0]["status"] == "completed"


def test_resume_accepts_manifest_left_running_by_abrupt_process_exit(tmp_path):
    paths, config = _setup(tmp_path)
    root = run_path(config, paths.runs_root)
    with pytest.raises(RuntimeError):
        _api().train_one(config, paths, False, False, runtime=FakeRuntime(error="interrupted"))
    (root / "weights").mkdir()
    (root / "weights/last.pt").write_bytes(b"synthetic interrupted checkpoint")
    transition_run(root, "running")
    assert _api().train_one(config, paths, True, False, runtime=FakeRuntime()) == root / "weights/last.pt"


def test_resume_rejects_changes_to_unrequested_effective_defaults(tmp_path):
    paths, config = _setup(tmp_path)
    root = run_path(config, paths.runs_root)
    first = FakeRuntime(error="interrupted", mutate=lambda trainer: setattr(trainer.args, "lr0", 0.01))
    with pytest.raises(RuntimeError):
        _api().train_one(config, paths, False, False, runtime=first)
    (root / "weights").mkdir()
    (root / "weights/last.pt").write_bytes(b"synthetic interrupted checkpoint")
    resumed = FakeRuntime(mutate=lambda trainer: setattr(trainer.args, "lr0", 0.05))
    with pytest.raises(ValueError, match="effective|resume"):
        _api().train_one(config, paths, True, False, runtime=resumed)
    assert _rows(paths)[0]["status"] == "failed"


def test_resume_cannot_report_completion_using_stale_last_checkpoint(tmp_path):
    paths, config = _setup(tmp_path)
    root = run_path(config, paths.runs_root)
    with pytest.raises(RuntimeError):
        _api().train_one(config, paths, False, False, runtime=FakeRuntime(error="interrupted"))
    (root / "weights").mkdir()
    (root / "weights/last.pt").write_bytes(b"synthetic interrupted checkpoint")
    with pytest.raises(ValueError, match="last.pt|stale"):
        _api().train_one(config, paths, True, False, runtime=FakeRuntime(save=False))
    assert _rows(paths)[0]["status"] == "failed"


def test_overwrite_restarts_from_original_weights_and_removes_old_artifacts(tmp_path):
    paths, config = _setup(tmp_path)
    root = run_path(config, paths.runs_root)
    _api().train_one(config, paths, False, False, runtime=FakeRuntime())
    (root / "old.txt").write_text("old artifact")
    runtime = FakeRuntime()
    _api().train_one(config, paths, False, True, runtime=runtime)
    assert runtime.loads == [(Path(config.weights), True)]
    assert runtime.calls[0][1]["resume"] is False
    assert not (root / "old.txt").exists()


def test_resume_without_last_fails_before_model_loading(tmp_path):
    paths, config = _setup(tmp_path)
    with pytest.raises(RuntimeError):
        _api().train_one(config, paths, False, False, runtime=FakeRuntime(error="interrupted"))
    before = _snapshot(paths.runs_root)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="checkpoint"):
        _api().train_one(config, paths, True, False, runtime=runtime)
    assert runtime.loads == [] and _snapshot(paths.runs_root) == before


@pytest.mark.parametrize("keep_going,expected_shots", [(False, [1, 2]), (True, [1, 2, 3, 5])])
def test_all_shots_are_independent_sequential_attempts_with_explicit_failure_mode(tmp_path, keep_going, expected_shots):
    paths, config = _setup(tmp_path, shots=(1, 2, 3, 5))
    runtime = FakeRuntime()

    def fail_shot_2(trainer):
        if "shot_2" in trainer.args.data:
            raise RuntimeError("synthetic shot 2 failure")

    runtime.mutate = fail_shot_2
    outcomes = _api().train_selected([1, 2, 3, 5], paths, weights=config.weights,
                                     settings={"epochs": 2}, continue_on_error=keep_going, runtime=runtime)
    assert [item.shot for item in outcomes] == expected_shots
    assert [item.status for item in outcomes] == (["completed", "failed"] if not keep_going else ["completed", "failed", "completed", "completed"])
    assert runtime.loads == [(Path(config.weights), True)] * len(expected_shots)
    assert [Path(options["data"]).parent.name for _, options in runtime.calls] == [f"shot_{shot}" for shot in expected_shots]
    assert len(_rows(paths)) == len(expected_shots)
    assert next(row for row in _rows(paths) if row["shot"] == "2")["status"] == "failed"


def test_batch_records_unresolved_identity_and_continues_missing_preparation(tmp_path):
    paths, config = _setup(tmp_path, shots=(1, 5))
    runtime = FakeRuntime()
    outcomes = _api().train_selected([1, 2, 5], paths, weights=config.weights,
                                     settings={"epochs": 2}, continue_on_error=True, runtime=runtime)
    assert [(item.shot, item.status) for item in outcomes] == [(1, "completed"), (2, "failed"), (5, "completed")]
    assert outcomes[1].config is None and outcomes[1].error_message
    failures = json.loads((paths.results_root / "training_attempts.json").read_text())
    assert failures[1]["shot"] == 2 and failures[1]["status"] == "failed" and failures[1]["config_hash"] is None
    assert runtime.loads == [(Path(config.weights), True)] * 2


def _native_modules(monkeypatch, tmp_path, *, defect=""):
    # External installed-library boundary; no project modules are replaced.
    class Segment:
        pass

    architecture = {"nc": 80, "scale": "n", "backbone": [[-1, 1, "Conv", [64, 3, 2]]],
                    "head": [[[16, 19, 22], 1, "Segment", ["nc", 32, 256]]]}
    names = {i: "coco class " + str(i) for i in range(80)}
    loaded = []
    seed_events = []

    def yolo(checkpoint):
        model = SimpleNamespace(task="segment", ckpt={"epoch": -1, "optimizer": None}, names=names,
                                model=SimpleNamespace(yaml=dict(architecture), model=[Segment()]))
        if defect == "detect":
            model.task = "detect"
        elif defect == "classes":
            model.names = {0: "CAMO class"}
        elif defect == "architecture":
            model.model.yaml["backbone"] = []
        elif defect == "resume":
            model.ckpt = {"epoch": 0, "optimizer": {"state": {}}}
        loaded.append(checkpoint)
        return model

    yaml = SimpleNamespace(load=lambda path: {"names": names} if Path(path).name == "coco.yaml" else architecture)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=yolo, __file__=str(tmp_path / "ultralytics/__init__.py")))
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", SimpleNamespace(get_cfg=lambda **kwargs: SimpleNamespace(**kwargs["overrides"])))
    monkeypatch.setitem(sys.modules, "ultralytics.utils", SimpleNamespace(YAML=yaml))
    monkeypatch.setitem(sys.modules, "ultralytics.nn.modules", SimpleNamespace(Segment=Segment))
    monkeypatch.setitem(sys.modules, "ultralytics.utils.torch_utils", SimpleNamespace(
        init_seeds=lambda seed, deterministic: seed_events.append(("ultralytics", seed, deterministic))))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(random=SimpleNamespace(seed=lambda seed: seed_events.append(("numpy", seed)))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: seed_events.append(("torch", seed)),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: seed_events.append(("cuda", seed)))))
    return loaded, seed_events


@pytest.mark.parametrize("defect", ["detect", "classes", "architecture"])
def test_native_adapter_rejects_non_coco_yolo11n_seg_base(tmp_path, monkeypatch, defect):
    _native_modules(monkeypatch, tmp_path, defect=defect)
    runtime = _api().UltralyticsRuntime()
    with pytest.raises(ValueError, match="COCO|YOLO11n|segment"):
        runtime.load(tmp_path / "weights.pt", base=True)


def test_native_adapter_seeds_all_requested_libraries_and_loads_compatible_base(tmp_path, monkeypatch):
    loaded, events = _native_modules(monkeypatch, tmp_path)
    runtime = _api().UltralyticsRuntime()
    warnings = runtime.seed(2024, True)
    assert abs(random.random() - 0.47009071843107064) < 1e-12
    assert events == [("numpy", 2024), ("torch", 2024), ("cuda", 2024), ("ultralytics", 2024, True)]
    assert warnings and "determin" in warnings[0].lower()
    model = runtime.load(tmp_path / "weights.pt", base=True)
    assert model.task == "segment" and loaded == [str(tmp_path / "weights.pt")]


def test_native_adapter_requires_optimizer_and_epoch_for_resume(tmp_path, monkeypatch):
    _native_modules(monkeypatch, tmp_path)
    runtime = _api().UltralyticsRuntime()
    with pytest.raises(ValueError, match="resume|optimizer"):
        runtime.load(tmp_path / "last.pt", base=False)


def test_inference_loader_accepts_stripped_last_without_resume_state(tmp_path, monkeypatch):
    # Break caught: inference reuses resume checks and rejects completed last.pt.
    loaded, _ = _native_modules(monkeypatch, tmp_path)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"synthetic completed checkpoint")
    model = _api().UltralyticsRuntime().load_inference(checkpoint)
    assert model.task == "segment" and loaded == [str(checkpoint)]


def test_inference_loader_fails_for_missing_local_checkpoint_without_native_resolution(tmp_path, monkeypatch):
    loaded, _ = _native_modules(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="checkpoint"):
        _api().UltralyticsRuntime().load_inference(tmp_path / "missing.pt")
    assert loaded == []


def test_inference_loader_rejects_detection_only_checkpoint(tmp_path, monkeypatch):
    _native_modules(monkeypatch, tmp_path, defect="detect")
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"synthetic detection checkpoint")
    with pytest.raises(ValueError, match="segmentation"):
        _api().UltralyticsRuntime().load_inference(checkpoint)


def test_missing_runtime_records_actionable_failed_attempt(tmp_path, monkeypatch):
    paths, config = _setup(tmp_path)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ValueError, match="Ultralytics|Kaggle"):
        _api().train_one(config, paths, False, False)
    assert _rows(paths)[0]["status"] == "failed"


def _cli():
    script = Path(__file__).resolve().parents[1] / "scripts/train_yolo.py"
    spec = importlib.util.spec_from_file_location("train_yolo_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_cli_parses_all_protocol_options_and_uses_shared_orchestration(tmp_path, capsys):
    paths, config = _setup(tmp_path, shots=(1, 2, 3, 5))
    runtime = FakeRuntime()
    result = _cli().main(["--shot", "all", "--method", "baseline", "--weights", config.weights,
                         "--data-root", str(paths.data_root), "--work-root", str(paths.work_root),
                         "--epochs", "2", "--imgsz", "320", "--batch", "4", "--device", "cpu",
                         "--seed", "12", "--run-kind", "smoke"], runtime=runtime)
    assert result == 0 and "shot 5: completed" in capsys.readouterr().out
    assert len(_rows(paths)) == 4
    assert all(row["run_kind"] == "smoke" and row["seed"] == "12" and row["imgsz"] == "320" for row in _rows(paths))
    assert [options["batch"] for _, options in runtime.calls] == [4] * 4


def test_train_cli_reports_failed_attempt_with_nonzero_exit(tmp_path, capsys):
    paths, config = _setup(tmp_path)
    result = _cli().main(["--shot", "1", "--weights", config.weights, "--data-root", str(paths.data_root),
                         "--work-root", str(paths.work_root)], runtime=FakeRuntime(error="synthetic failure"))
    assert result == 1 and "synthetic failure" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
@pytest.mark.parametrize("val", [None, ""])
def test_train_cli_rejects_empty_val_with_actionable_preflight_error(tmp_path, capsys, method, val):
    # Break caught: CLI hashes an invalid val before running the shared preflight
    # and exposes a Path type error instead of the mandatory TRAIN placeholder.
    paths, config = _setup(tmp_path, method=method)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    document = json.loads(yaml.read_text())
    document["val"] = val
    yaml.write_text(json.dumps(document))
    runtime = FakeRuntime() if method == "baseline" else EnhancedRuntime()
    result = _cli().main(["--shot", "1", "--method", method, "--weights", config.weights,
                         "--data-root", str(paths.data_root), "--work-root", str(paths.work_root)], runtime=runtime)
    assert result == 1
    assert "TRAIN placeholder" in capsys.readouterr().out
    assert runtime.loads == [] and runtime.calls == [] and runtime.seeds == []
    assert not paths.runs_root.exists()


def test_train_cli_help_does_not_import_training_dependencies():
    script = Path(__file__).resolve().parents[1] / "scripts/train_yolo.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--resume" in result.stdout and "--continue-on-error" in result.stdout


def test_shared_option_builder_keeps_non_method_protocol_equal(tmp_path):
    paths, baseline = _setup(tmp_path)
    enhanced = replace(baseline, method="fgbg-triplet")
    native = _api().training_options(baseline, paths)
    auxiliary = _api().training_options(enhanced, paths)
    assert {key: value for key, value in native.items() if key not in ("project", "name")} == {
        key: value for key, value in auxiliary.items() if key not in ("project", "name")}
    assert native["hsv_h"] == 0.015 and native["fliplr"] == 0.5
    assert all(native[key] == 0 for key in ("degrees", "translate", "scale", "shear", "perspective", "flipud"))


def test_native_defaults_are_bound_before_run_identity_and_forwarded(tmp_path):
    paths, config = _setup(tmp_path)
    first = FakeRuntime(defaults={"lr0": 0.01, "optimizer": "SGD", "mask_ratio": 4})
    second = FakeRuntime(defaults={"lr0": 0.02, "optimizer": "SGD", "mask_ratio": 4})
    one = _api().train_selected([1], paths, weights=config.weights, settings={"epochs": 2}, runtime=first)[0]
    two = _api().train_selected([1], paths, weights=config.weights, settings={"epochs": 2}, runtime=second)[0]
    assert one.status == two.status == "completed"
    assert one.last_pt != two.last_pt and len(_rows(paths)) == 2
    assert one.config.training_options["lr0"] == 0.01 and two.config.training_options["lr0"] == 0.02
    assert first.calls[0][1]["mask_ratio"] == 4 and second.calls[0][1]["lr0"] == 0.02


def test_explicit_training_settings_override_resolved_defaults(tmp_path):
    paths, config = _setup(tmp_path)
    runtime = FakeRuntime(defaults={"lr0": 0.01, "optimizer": "SGD"})
    item = _api().train_selected([1], paths, weights=config.weights,
                                 settings={"epochs": 2, "training_options": {"lr0": 0.03}}, runtime=runtime)[0]
    assert item.status == "completed" and item.config.training_options == {"lr0": 0.03, "optimizer": "SGD"}
    assert runtime.calls[0][1]["lr0"] == 0.03


def test_shared_label_cache_does_not_block_training_integrity_gate(tmp_path):
    paths, config = _setup(tmp_path)
    before = config.data_sha256
    (paths.prepared_root / "test/labels.cache").write_bytes(b"synthetic native evaluator cache")
    assert prepared_data_sha256(paths.prepared_root / "shot_1") == before
    last = _api().train_one(config, paths, False, False, runtime=FakeRuntime())
    assert last.is_file()


def test_failed_resume_load_error_replaces_failure_message_without_masking_it(tmp_path):
    paths, config = _setup(tmp_path)
    root = run_path(config, paths.runs_root)
    with pytest.raises(RuntimeError, match="first interruption"):
        _api().train_one(config, paths, False, False, runtime=FakeRuntime(error="first interruption"))
    (root / "weights").mkdir()
    (root / "weights/last.pt").write_bytes(b"synthetic interrupted checkpoint")

    class BrokenLoad(FakeRuntime):
        def load(self, checkpoint, *, base):
            raise ValueError("new checkpoint load failure")

    with pytest.raises(ValueError, match="new checkpoint load failure"):
        _api().train_one(config, paths, True, False, runtime=BrokenLoad())
    assert _rows(paths)[0]["error_message"] == "new checkpoint load failure"
    assert json.loads((root / "manifest.json").read_text())["error_message"] == "new checkpoint load failure"


@pytest.mark.parametrize("device,workers", [("cpu", 0), ("0", 8)])
def test_native_optimizer_and_cpu_worker_normalization_are_resolved_before_hash(tmp_path, device, workers):
    paths, config = _setup(tmp_path)

    def native_setup(trainer):
        if trainer.args.device == "cpu":
            trainer.args.workers = 0
        if trainer.args.optimizer == "auto":
            trainer.args.warmup_bias_lr = 0.0

    runtime = FakeRuntime(defaults={"optimizer": "auto", "lr0": 0.01, "warmup_bias_lr": 0.1, "workers": 8},
                          mutate=native_setup)
    item = _api().train_selected([1], paths, weights=config.weights,
                                 settings={"epochs": 2, "device": device}, runtime=runtime)[0]
    assert item.status == "completed", item.error_message
    assert item.config.training_options["optimizer"] == "SGD"
    assert item.config.training_options["workers"] == workers
    manifest = json.loads(item.last_pt.parent.parent.joinpath("manifest.json").read_text())
    assert manifest["effective_training_options"]["warmup_bias_lr"] == 0.1


def test_explicit_auto_optimizer_is_rejected_before_model_load(tmp_path):
    paths, config = _setup(tmp_path)
    runtime = FakeRuntime(defaults={"optimizer": "auto", "lr0": 0.01})
    item = _api().train_selected([1], paths, weights=config.weights,
                                 settings={"training_options": {"optimizer": "auto"}}, runtime=runtime)[0]
    assert item.status == "failed" and "optimizer" in item.error_message
    assert runtime.loads == []


@pytest.mark.parametrize("extra", [{"channels": 1}, {"nc": 47}, {"path": "elsewhere"}])
@pytest.mark.parametrize("entrypoint", ["fingerprint", "audit", "training"])
def test_training_rejects_semantic_yaml_fields_outside_t04_contract(tmp_path, extra, entrypoint):
    # Break caught: an unbound YAML field changes native model/dataset semantics.
    paths, config = _setup(tmp_path)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    document = json.loads(yaml.read_text())
    document.update(extra)
    yaml.write_text(json.dumps(document))
    before = _snapshot(paths.prepared_root)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="YAML|unsupported|integrity"):
        if entrypoint == "fingerprint":
            prepared_data_sha256(yaml.parent)
        elif entrypoint == "audit":
            from camo_fs.prepare import verify_prepared
            verify_prepared(1, paths)
        else:
            _api().train_one(config, paths, False, False, runtime=runtime)
    assert runtime.loads == [] and not run_path(config, paths.runs_root).exists()
    assert _snapshot(paths.prepared_root) == before


@pytest.mark.parametrize("field,value", [("channels", 1), ("names", {0: "Fox", 1: "Bat"}),
                                       ("names", {0: "Bat", 3: "Fox"}), ("names", None),
                                       ("nc", 47), ("nc", True)])
def test_training_rejects_parsed_native_channel_or_taxonomy_mismatch(tmp_path, field, value):
    # Break caught: loader/model metadata diverges from the audited canonical data.
    paths, config = _setup(tmp_path)
    before = _snapshot(paths.prepared_root)
    runtime = FakeRuntime(mutate=lambda trainer: trainer.data.update({field: value}))
    with pytest.raises(ValueError, match="RGB|channels|taxonomy|classes"):
        _api().train_one(config, paths, False, False, runtime=runtime)
    assert _rows(paths)[0]["status"] == "failed"
    assert not (run_path(config, paths.runs_root) / "weights/last.pt").exists()
    assert _snapshot(paths.prepared_root) == before


@pytest.mark.parametrize("names", [["Bat", "Fox"], {0: "Bat", 1: "Fox"}])
@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
def test_t04_yaml_train_placeholder_and_canonical_parsed_taxonomy_remain_accepted(tmp_path, names, method):
    paths, config = _setup(tmp_path, method=method)
    runtime_type = FakeRuntime if method == "baseline" else EnhancedRuntime
    runtime = runtime_type(mutate=lambda trainer: trainer.data.update(names=names))
    last = _api().train_one(config, paths, False, False, runtime=runtime)
    assert last.is_file() and _rows(paths)[0]["status"] == "completed"
