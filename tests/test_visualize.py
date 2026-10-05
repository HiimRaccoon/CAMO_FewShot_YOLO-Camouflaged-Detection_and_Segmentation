"""T12 public visualization seams, synthetic data and prediction boundary doubles."""

from dataclasses import replace
import importlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from test_evaluate import _setup, _run, _record_completion
from test_prepare import _snapshot
from camo_fs.runs import prepared_data_sha256


def _api():
    return importlib.import_module("camo_fs.visualize")


def test_selection_defaults_to_twenty_is_order_independent_and_does_not_change_global_rng():
    images = [Path(f"test/images/image_{index:02}.png") for index in range(40)]
    state = random.getstate()
    selected = _api().select_images(images)
    assert len(selected) == 20 and len(set(selected)) == 20
    assert selected == _api().select_images(list(reversed(images)))
    assert selected != _api().select_images(images, seed=7)
    assert random.getstate() == state


def test_selection_caps_at_available_images():
    images = [Path("test/images/a.png"), Path("test/images/b.png")]
    assert set(_api().select_images(images, num_images=20)) == set(images)
    assert len(_api().select_images(images, num_images=1)) == 1


@pytest.mark.parametrize("kwargs", [{"num_images": 0}, {"num_images": -1}, {"num_images": True},
                                    {"seed": -1}, {"seed": True}])
def test_selection_rejects_invalid_count_or_seed(kwargs):
    with pytest.raises(ValueError):
        _api().select_images([Path("test.png")], **kwargs)


class FakeResult:
    def __init__(self, path, runtime):
        import numpy as np
        from PIL import Image

        self.runtime, self.path, self.names = runtime, str(path), runtime.names
        self.orig_img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        self.orig_shape = self.orig_img.shape[:2]
        h, w = self.orig_shape
        values = np.empty((0, 6)) if runtime.empty else np.array([[w / 4, h / 4, 3 * w / 4, 3 * h / 4, 0.9, 1]])
        self.boxes = SimpleNamespace(data=values, xyxy=values[:, :4], conf=values[:, 4], cls=values[:, 5])
        masks = np.zeros((len(values), h, w))
        masks[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
        self.masks = None if runtime.empty else SimpleNamespace(data=masks)

    def plot(self, **options):
        self.runtime.plots.append(options)
        image = self.orig_img.copy()
        if not self.runtime.empty:
            image[0, 0] = [10, 20, 30]  # Native plot returns BGR.
        return image


class FakeRuntime:
    def __init__(self, *, empty=False, mutate=lambda result: None, fail_at=None):
        self.empty, self.mutate, self.fail_at = empty, mutate, fail_at
        self.names = {0: "Bat", 1: "Fox"}
        self.loads, self.calls, self.plots, self.seeds = [], [], [], []

    def seed(self, seed, deterministic):
        self.seeds.append((seed, deterministic))
        return []

    def load_inference(self, checkpoint):
        self.loads.append(checkpoint)
        return self

    def predict(self, **options):
        self.calls.append(options)
        if self.fail_at == len(self.calls):
            raise RuntimeError("prediction failed")
        result = FakeResult(Path(options["source"]), self)
        self.mutate(result)
        return [result]


@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
def test_visualize_loads_own_last_and_renders_box_mask_saved_names_and_confidence(tmp_path, method):
    from PIL import Image

    paths, config = _setup(tmp_path, image_size=64)
    root = _run(paths, replace(config, method=method))
    runtime = FakeRuntime()
    before_source, before_run = _snapshot(paths.data_root), _snapshot(root)
    saved = _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=runtime)
    assert runtime.loads == [root / "weights/last.pt"]
    assert len(saved) == 1 and saved[0].is_relative_to(root / "visualizations")
    assert runtime.calls[0]["source"] == str(paths.prepared_root / "test/images/test.png")
    assert runtime.calls[0]["conf"] == 0.25 and runtime.calls[0]["retina_masks"] is True
    assert runtime.calls[0]["save"] is False and runtime.calls[0]["augment"] is False
    assert runtime.plots[0]["boxes"] is True and runtime.plots[0]["masks"] is True
    assert runtime.plots[0]["labels"] is True and runtime.plots[0]["conf"] is True
    assert runtime.seeds == [(2024, True)]
    with Image.open(saved[0]) as rendered:
        assert rendered.size == (64, 64) and rendered.getpixel((0, 0)) == (30, 20, 10)
    report = json.loads((root / "visualizations/manifest.json").read_text())
    assert report["config_hash"] == root.name and report["seed"] == 2024
    assert report["conf"] == 0.25 and report["images"][0]["source"] == "test.png"
    assert report["images"][0]["detections"] == 1
    assert _snapshot(paths.data_root) == before_source
    assert all((root / name).read_bytes() == content for name, content in before_run.items())


def test_empty_prediction_still_saves_the_selected_image_with_explicit_zero_count(tmp_path):
    from PIL import Image

    paths, config = _setup(tmp_path, image_size=64)
    root = _run(paths, config)
    runtime = FakeRuntime(empty=True)
    saved = _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=runtime)
    assert len(saved) == 1
    with Image.open(saved[0]) as rendered:
        assert rendered.getpixel((0, 0)) == (128, 128, 128)
    report = json.loads((root / "visualizations/manifest.json").read_text())
    assert report["images"][0]["detections"] == 0


@pytest.mark.parametrize("failure", ["checkpoint changed", "missing epochs", "wrong epochs", "missing digest",
                                      "test is train", "changed test", "wrong YAML", "wrong run dir"])
def test_visualization_rejects_unbound_checkpoint_data_or_run_before_load(tmp_path, failure):
    paths, config = _setup(tmp_path, shots=(1, 2), image_size=64)
    root = _run(paths, config)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    if failure == "checkpoint changed":
        (root / "weights/last.pt").write_bytes(b"replaced with same taxonomy")
    elif failure in ("missing epochs", "wrong epochs", "missing digest"):
        path = root / "manifest.json"
        document = json.loads(path.read_text())
        if failure == "wrong epochs":
            document["completed_epochs"] = 0
        else:
            document.pop("completed_epochs" if failure == "missing epochs" else "last_checkpoint_sha256")
        path.write_text(json.dumps(document))
    elif failure == "test is train":
        document = json.loads(yaml.read_text())
        document["test"] = document["train"]
        yaml.write_text(json.dumps(document))
    elif failure == "changed test":
        (paths.prepared_root / "test/labels/test.txt").write_text("0 0 0 1 0 1 1\n")
    elif failure == "wrong YAML":
        yaml = paths.prepared_root / "shot_2/data.yaml"
    else:
        copied = paths.runs_root / "impostor"
        copied.mkdir()
        (copied / "manifest.json").write_bytes((root / "manifest.json").read_bytes())
        root = copied
    runtime = FakeRuntime()
    with pytest.raises(ValueError):
        _api().visualize_one(root, yaml, runtime=runtime)
    assert runtime.loads == [] and runtime.calls == [] and runtime.seeds == []
    assert not (root / "visualizations").exists()


@pytest.mark.parametrize("status", ["running", "failed"])
def test_visualization_requires_completed_training(tmp_path, status):
    paths, config = _setup(tmp_path)
    root = _run(paths, config, status=status)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="completed"):
        _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=runtime)
    assert runtime.loads == []


@pytest.mark.parametrize("conf", [-0.1, 1.1, float("nan"), True])
def test_invalid_confidence_rejected_before_model_load(tmp_path, conf):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="conf"):
        _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", conf=conf, runtime=runtime)
    assert runtime.loads == []


@pytest.mark.parametrize("mutation", ["taxonomy", "path", "shape", "unpaired mask", "invalid class"])
def test_prediction_must_match_image_shape_taxonomy_and_instance_masks(tmp_path, mutation):
    paths, config = _setup(tmp_path, image_size=64)
    root = _run(paths, config)
    def mutate(result):
        if mutation == "taxonomy":
            result.names = {0: "Fox", 1: "Bat"}
        elif mutation == "path":
            result.path = str(paths.prepared_root / "shot_1/train/images/bat_1.png")
        elif mutation == "shape":
            result.orig_shape = (32, 64)
        elif mutation == "unpaired mask":
            result.masks = None
        else:
            result.boxes.cls[0] = 99
    with pytest.raises(ValueError):
        _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=FakeRuntime(mutate=mutate))
    assert not (root / "visualizations").exists()


def test_checkpoint_taxonomy_must_match_before_predict(tmp_path):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    runtime = FakeRuntime()
    runtime.names = {0: "Fox", 1: "Bat"}
    with pytest.raises(ValueError, match="taxonomy"):
        _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=runtime)
    assert runtime.calls == []


def _many_setup(tmp_path, names=("a/same.png", "b/same.png", "third.png")):
    from PIL import Image
    from camo_fs.prepare import prepare_selected

    paths, config = _setup(tmp_path, image_size=64)
    document = json.loads(paths.test_json.read_text())
    original = document["annotations"][0]
    document["images"], document["annotations"] = [], []
    for index, name in enumerate(names):
        destination = paths.images_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (128, 128, 128)).save(destination)
        document["images"].append({"id": 900 + index, "file_name": name, "width": 64, "height": 64})
        document["annotations"].append({**original, "id": 900 + index, "image_id": 900 + index})
    paths.test_json.write_text(json.dumps(document))
    assert prepare_selected([1], paths, overwrite=True, continue_on_error=False,
                            val_train_placeholder=True)[0].status == "prepared"
    return paths, replace(config, data_sha256=prepared_data_sha256(paths.prepared_root / "shot_1"))


def test_nested_duplicate_basenames_have_distinct_files_and_seed_does_not_replace_run_identity(tmp_path):
    paths, config = _many_setup(tmp_path)
    root = _run(paths, config)
    saved = _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", seed=7, runtime=FakeRuntime())
    assert len(saved) == 3 and len({path.name for path in saved}) == 3
    report = json.loads((root / "visualizations/manifest.json").read_text())
    assert {record["source"] for record in report["images"]} == {"a/same.png", "b/same.png", "third.png"}
    assert report["seed"] == 2024 and report["selection_seed"] == 7
    assert report["last_checkpoint_sha256"] == json.loads((root / "manifest.json").read_text())["last_checkpoint_sha256"]


def test_successful_rerun_replaces_only_owned_images_and_removes_stale_selection(tmp_path):
    paths, config = _many_setup(tmp_path)
    root = _run(paths, config)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    _api().visualize_one(root, yaml, runtime=FakeRuntime())
    saved = _api().visualize_one(root, yaml, num_images=1, conf=0.1, seed=7, runtime=FakeRuntime())
    assert len(saved) == 1 and list((root / "visualizations").glob("*.png")) == saved
    report = json.loads((root / "visualizations/manifest.json").read_text())
    assert report["conf"] == 0.1 and report["selection_seed"] == 7 and len(report["images"]) == 1


def test_failed_rerun_preserves_every_previous_render_and_report(tmp_path):
    paths, config = _many_setup(tmp_path)
    root = _run(paths, config)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    _api().visualize_one(root, yaml, runtime=FakeRuntime())
    before = _snapshot(root / "visualizations")
    with pytest.raises(RuntimeError, match="prediction failed"):
        _api().visualize_one(root, yaml, runtime=FakeRuntime(fail_at=2))
    assert _snapshot(root / "visualizations") == before
    assert not list(root.glob(".viz-stage-*"))


@pytest.mark.parametrize("restore_fails", [False, True])
def test_failed_publish_preserves_previous_output_or_reports_retained_recovery_backup(tmp_path, monkeypatch, restore_fails):
    paths, config = _many_setup(tmp_path)
    root = _run(paths, config)
    yaml = paths.prepared_root / "shot_1/data.yaml"
    _api().visualize_one(root, yaml, runtime=FakeRuntime())
    target = root / "visualizations"
    before = _snapshot(target)
    original_replace = Path.replace

    def fail_publish(path, destination):
        if Path(destination) == target and path.name in ("fresh", "backup"):
            if path.name == "fresh" or restore_fails:
                raise OSError("injected filesystem failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises((OSError, ValueError)) as error:
        _api().visualize_one(root, yaml, num_images=1, runtime=FakeRuntime())
    if restore_fails:
        backups = list(root.glob(".viz-stage-*/backup"))
        assert len(backups) == 1 and _snapshot(backups[0]) == before
        assert str(backups[0]) in str(error.value) and not target.exists()
    else:
        assert _snapshot(target) == before and not list(root.glob(".viz-stage-*"))


@pytest.mark.parametrize("existing", ["unmanaged", "other identity"])
def test_existing_unowned_output_is_rejected_before_inference(tmp_path, existing):
    paths, config = _setup(tmp_path)
    root = _run(paths, config)
    target = root / "visualizations"
    target.mkdir()
    (target / "user.txt").write_text("preserve me")
    if existing == "other identity":
        (target / "manifest.json").write_text(json.dumps({"config_hash": "a" * 64}))
    before = _snapshot(target)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="owned"):
        _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=runtime)
    assert _snapshot(target) == before and runtime.loads == []


def test_other_run_renders_and_ap_summary_are_unchanged(tmp_path):
    paths, config = _setup(tmp_path)
    roots = [_run(paths, replace(config, epochs=epochs)) for epochs in (2, 3)]
    yaml = paths.prepared_root / "shot_1/data.yaml"
    _api().visualize_one(roots[0], yaml, runtime=FakeRuntime())
    before = _snapshot(roots[0])
    summary = paths.results_root / "summary.csv"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text("preserve evaluation summary\n")
    _api().visualize_one(roots[1], yaml, runtime=FakeRuntime())
    assert _snapshot(roots[0]) == before and summary.read_text() == "preserve evaluation summary\n"


def test_cli_all_methods_and_shots_discovers_every_benchmark_hash_and_skips_smoke(tmp_path, capsys):
    from camo_fs.runs import prepared_data_sha256

    paths, config = _setup(tmp_path, shots=(1, 2, 3, 5))
    roots = []
    for method in ("baseline", "fgbg-triplet"):
        for shot in (1, 2, 3, 5):
            selected = replace(config, method=method, shot=shot,
                                data_sha256=prepared_data_sha256(paths.prepared_root / f"shot_{shot}"))
            roots.append(_run(paths, selected))
    roots.append(_run(paths, replace(selected, triplet_weight=0.2)))
    _run(paths, replace(selected, run_kind="smoke"))
    runtime = FakeRuntime()
    cli = importlib.import_module("scripts.visualize_predictions")
    assert cli.main(["--shot", "all", "--method", "all", "--work-root", str(paths.work_root),
                     "--data-root", str(paths.data_root)], runtime=runtime) == 0
    assert set(runtime.loads) == {root / "weights/last.pt" for root in roots}
    assert capsys.readouterr().out.count("completed") == 9


@pytest.mark.parametrize("include_smoke,expected", [(False, 1), (True, 0)])
def test_cli_single_run_excludes_smoke_unless_opted_in_and_forwards_visual_options(tmp_path, include_smoke, expected):
    paths, config = _setup(tmp_path)
    root = _run(paths, replace(config, run_kind="smoke"))
    runtime = FakeRuntime()
    cli = importlib.import_module("scripts.visualize_predictions")
    args = ["--run-dir", str(root), "--work-root", str(paths.work_root), "--data-root", str(paths.data_root),
            "--num-images", "1", "--conf", "0.1", "--seed", "7"]
    assert cli.main([*args, *(["--include-smoke"] if include_smoke else [])], runtime=runtime) == expected
    if include_smoke:
        assert runtime.calls[0]["conf"] == 0.1 and runtime.seeds == [(7, True)]
    else:
        assert runtime.loads == []


@pytest.mark.parametrize("continue_on_error,expected_count", [(False, 1), (True, 2)])
def test_batch_failed_prediction_is_reported_and_continue_is_explicit(tmp_path, continue_on_error, expected_count):
    paths, config = _setup(tmp_path)
    _run(paths, config)
    _run(paths, replace(config, epochs=3))
    outcomes = _api().visualize_selected((1,), paths, runtime=FakeRuntime(fail_at=1),
                                         continue_on_error=continue_on_error)
    assert len(outcomes) == expected_count and outcomes[0]["status"] == "failed"
    assert "prediction failed" in outcomes[0]["error_message"]
    if continue_on_error:
        assert outcomes[1]["status"] == "completed"


def test_cli_empty_selection_and_failed_run_return_nonzero(tmp_path, capsys):
    paths, config = _setup(tmp_path)
    cli = importlib.import_module("scripts.visualize_predictions")
    args = ["--shot", "1", "--work-root", str(paths.work_root), "--data-root", str(paths.data_root)]
    assert cli.main(args, runtime=FakeRuntime()) == 1
    assert "No completed" in capsys.readouterr().err
    _run(paths, config)
    assert cli.main(args, runtime=FakeRuntime(fail_at=1)) == 1
    assert "prediction failed" in capsys.readouterr().out
    for forbidden in (["--weights", "best.pt"], ["--split", "val"]):
        with pytest.raises(SystemExit) as error:
            cli.main([*args, *forbidden], runtime=FakeRuntime())
        assert error.value.code == 2


def _debug_batch():
    import torch

    image = torch.full((3, 64, 64), 128, dtype=torch.uint8)
    masks = torch.zeros((2, 8, 8), dtype=torch.bool)
    masks[0, 2:6, 2:6] = True
    masks[1, 0, 0] = True
    batch_idx = torch.tensor([0, 0])
    valid = torch.ones((1, 64, 64), dtype=torch.bool)
    valid[:, :8] = False
    positions = torch.tensor([[0, 0, 2, 2, 3, 3, 6, 6]], dtype=torch.int64)
    return image, masks, batch_idx, valid, positions


def test_debug_renders_transformed_gt_and_sampled_points_without_mutating_tensors(tmp_path):
    import torch

    args = _debug_batch()
    before = [value.clone() for value in args]
    rendered = _api().render_sampler_debug(*args, feature_hw=(8, 8))
    assert rendered.getpixel((20, 20)) == (255, 64, 64)  # anchor feature-cell center
    assert rendered.getpixel((28, 28)) == (64, 255, 64)  # positive
    assert rendered.getpixel((52, 52)) == (64, 160, 255)  # negative
    assert rendered.getpixel((63, 0)) != (128, 128, 128)  # visibly marked padding
    assert all(torch.equal(value, saved) for value, saved in zip(args, before))
    destination = tmp_path / "sampler-debug.png"
    rendered.save(destination)
    print(f"DEBUG_FIXTURE={destination}")


@pytest.mark.parametrize("bad", ["negative in GT", "negative in padding", "anchor outside instance",
                                  "identical A/P", "wrong image", "negative in other GT", "test split"])
def test_debug_rejects_points_outside_same_instance_background_or_valid_region(bad):
    args = list(_debug_batch())
    kwargs = {}
    if bad == "negative in GT":
        args[4][0, 6:] = args[4].new_tensor([2, 3])
    elif bad == "negative in padding":
        args[4][0, 6:] = args[4].new_tensor([0, 7])
    elif bad == "anchor outside instance":
        args[4][0, 2:4] = args[4].new_tensor([6, 6])
    elif bad == "identical A/P":
        args[4][0, 4:6] = args[4][0, 2:4]
    elif bad == "wrong image":
        args[4][0, 1] = 1
    elif bad == "negative in other GT":
        args[3][:] = True
        args[4][0, 6:] = args[4].new_tensor([0, 0])
    else:
        kwargs["split"] = "test"
    with pytest.raises(ValueError):
        _api().render_sampler_debug(*args, feature_hw=(8, 8), **kwargs)


def test_debug_accepts_actual_triplet_result_positions_on_fractional_feature_grid():
    import torch
    from camo_fs.triplet import sample_and_loss

    image, masks, batch_idx, valid, _ = _debug_batch()
    features = torch.randn((1, 3, 5, 7), generator=torch.Generator().manual_seed(3))
    result = sample_and_loss(features, masks, batch_idx, valid, 4, 0.3,
                             torch.Generator().manual_seed(2024))
    assert result.sampled_triplets > 0
    rendered = _api().render_sampler_debug(image, masks, batch_idx, valid,
                                           result.sampled_positions, feature_hw=(5, 7))
    assert rendered.width == 128 and rendered.height >= 64


def test_fractional_debug_distinguishes_projected_foreground_from_source_gt(tmp_path):
    import torch

    image = torch.full((3, 64, 64), 128, dtype=torch.uint8)
    masks = torch.zeros((1, 8, 8), dtype=torch.bool)
    masks[0, 3, 1:3] = True
    valid = torch.ones((1, 64, 64), dtype=torch.bool)
    positions = torch.tensor([[0, 0, 2, 1, 2, 2, 4, 6]], dtype=torch.int64)
    rendered = _api().render_sampler_debug(image, masks, torch.tensor([0]), valid,
                                           positions, feature_hw=(5, 7))
    # Actual feature center remains (13,32), outside the source GT's row 24:32.
    assert rendered.getpixel((13, 32)) == (255, 64, 64)
    assert rendered.getpixel((13, 36)) != (128, 128, 128)  # projected cell, beyond point marker
    assert rendered.getpixel((64 + 13, 36)) == (128, 128, 128)  # source GT panel
    assert rendered.getpixel((64 + 13, 28)) != (128, 128, 128)  # actual source foreground
    assert rendered.getpixel((60, 36)) == (128, 128, 128)  # neither projection nor source
    destination = tmp_path / "fractional-sampler-debug.png"
    rendered.save(destination)
    print(f"FRACTIONAL_DEBUG_FIXTURE={destination}")


@pytest.mark.integration
def test_native_results_render_box_mask_class_and_confidence_in_original_image_coordinates(tmp_path):
    ultralytics = pytest.importorskip("ultralytics")
    if ultralytics.__version__ != "8.3.228":
        pytest.skip("Native Results proof requires project pin 8.3.228")
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics.engine.results import Results
    from camo_fs.prepare import prepare_selected

    paths, config = _setup(tmp_path, image_size=128)
    Image.new("RGB", (128, 128), (160, 80, 40)).save(paths.images_dir / "test.png")
    assert prepare_selected([1], paths, True, False, val_train_placeholder=True)[0].status == "prepared"
    config = replace(config, data_sha256=prepared_data_sha256(paths.prepared_root / "shot_1"))
    root = _run(paths, config)

    class NativeResultsRuntime(FakeRuntime):
        def predict(self, **options):
            self.calls.append(options)
            orig_img = np.array(Image.open(options["source"]).convert("RGB"))[:, :, ::-1].copy()
            mask = torch.zeros((1, 128, 128))
            mask[:, 48:96, 32:96] = 1
            return [Results(orig_img, options["source"], self.names,
                            boxes=torch.tensor([[32, 32, 96, 96, 0.91, 1]]), masks=mask)]

    saved = _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=NativeResultsRuntime())
    with Image.open(saved[0]) as rendered:
        assert rendered.getpixel((120, 120)) == (160, 80, 40)
        assert rendered.getpixel((64, 72)) != (160, 80, 40)  # native mask overlay
        assert rendered.getpixel((32, 90)) != (160, 80, 40)  # native box border
    print(f"PREDICTION_FIXTURE={saved[0]}")


@pytest.mark.integration
@pytest.mark.parametrize("method", ["baseline", "fgbg-triplet"])
def test_native_pinned_inference_reloads_verified_checkpoint_and_only_predicts_fixture_test(tmp_path, method):
    ultralytics = pytest.importorskip("ultralytics")
    if ultralytics.__version__ != "8.3.228":
        pytest.skip("Native inference proof requires project pin 8.3.228")
    import torch
    from ultralytics.nn.tasks import SegmentationModel
    from camo_fs.ultralytics_ext import FGSegmentationModel
    from camo_fs.train import UltralyticsRuntime
    from camo_fs.runs import transition_run

    paths, config = _setup(tmp_path, image_size=64)
    config = replace(config, method=method, imgsz=64)
    root = _run(paths, config, status="running")
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
        _record_completion(root, config.epochs)
        transition_run(root, "completed")
        saved = _api().visualize_one(root, paths.prepared_root / "shot_1/data.yaml", runtime=UltralyticsRuntime())
        assert len(saved) == 1 and saved[0].is_file()
        report = json.loads((root / "visualizations/manifest.json").read_text())
        assert [record["source"] for record in report["images"]] == ["test.png"]
        assert report["conf"] == 0.25
    finally:
        torch.set_num_threads(threads)
