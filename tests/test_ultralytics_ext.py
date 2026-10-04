"""Public extension seams on tiny native-like modules, without Ultralytics."""

import pytest
import torch
from torch import nn


class Segment(nn.Module):
    def forward(self, features):
        return features[0].square().mean()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, 3, stride=8, padding=1)
        self.head = Segment()

    def forward(self, images):
        p3 = self.backbone(images)
        return self.head([p3, p3[..., ::2, ::2], p3[..., ::4, ::4]])


def test_capture_is_semantic_keeps_gradient_and_releases_hooks():
    from camo_fs.ultralytics_ext import P3Capture

    model = TinyModel()
    images = torch.randn(2, 3, 32, 64)
    capture = P3Capture(model, Segment, expected_channels=4)
    with capture.scope(images):
        loss = model(images)
        feature = capture.feature
        assert feature.shape == (2, 4, 4, 8)
        feature.retain_grad()
        loss.backward()
        assert feature.grad.abs().sum() > 0
    assert not model._forward_pre_hooks and not model.head._forward_pre_hooks
    with pytest.raises(RuntimeError, match="active"):
        _ = capture.feature


def test_capture_rejects_stale_or_multiple_forward():
    from camo_fs.ultralytics_ext import P3Capture

    model, images = TinyModel(), torch.randn(1, 3, 32, 32)
    capture = P3Capture(model, Segment, expected_channels=4)
    with capture.scope(images):
        model(images)
    with pytest.raises(RuntimeError, match="Missing"):
        with capture.scope(images):
            _ = capture.feature
    with pytest.raises(RuntimeError, match="multiple"):
        with capture.scope(images):
            model(images)
            model(images)
    assert not model.head._forward_pre_hooks


@pytest.mark.parametrize("failure", ["batch", "channels", "stride", "inputs", "detached", "foreign-image"])
def test_capture_rejects_invalid_head_inputs(failure):
    from camo_fs.ultralytics_ext import P3Capture

    class Broken(TinyModel):
        def forward(self, images):
            feature = self.backbone(images)
            if failure == "batch":
                feature = feature[:0]
            elif failure == "channels":
                feature = feature[:, :3]
            elif failure == "stride":
                feature = feature[..., ::2, ::2]
            elif failure == "detached":
                feature = feature.detach()
            inputs = [feature, feature[..., ::2, ::2], feature[..., ::4, ::4]]
            return self.head(inputs[:2] if failure == "inputs" else inputs)

    images = torch.randn(1, 3, 32, 32)
    model = Broken()
    capture = P3Capture(model, Segment, expected_channels=4)
    with pytest.raises((RuntimeError, ValueError), match="P3|three|image"):
        with capture.scope(images):
            model(images.clone() if failure == "foreign-image" else images)
    assert not model._forward_pre_hooks and not model.head._forward_pre_hooks


def test_capture_rejects_multiple_head_calls_and_nested_scope_and_releases_on_error():
    from camo_fs.ultralytics_ext import P3Capture

    model, images = TinyModel(), torch.randn(1, 3, 32, 32)
    capture = P3Capture(model, Segment, expected_channels=4)
    with pytest.raises(RuntimeError, match="multiple"):
        with capture.scope(images):
            features = [torch.randn(1, 4, 4, 4, requires_grad=True),
                        torch.randn(1, 4, 2, 2), torch.randn(1, 4, 1, 1)]
            model.head(features)
            model.head(features)
    with pytest.raises(RuntimeError, match="active"):
        with capture.scope(images):
            with capture.scope(images):
                pass
    with pytest.raises(LookupError):
        with capture.scope(images):
            model(images)
            raise LookupError("native forward failed")
    assert not model._forward_pre_hooks and not model.head._forward_pre_hooks


def test_capture_requires_exactly_one_segment():
    from camo_fs.ultralytics_ext import P3Capture

    with pytest.raises(ValueError, match="exactly one"):
        P3Capture(nn.Sequential(), Segment, expected_channels=4)
    model = TinyModel()
    model.extra = Segment()
    with pytest.raises(ValueError, match="exactly one"):
        P3Capture(model, Segment, expected_channels=4)


class NativeModel(TinyModel):
    def predict(self, images, *args, **kwargs):
        return TinyModel.forward(self, images)

    def forward(self, value):
        return self.loss(value) if isinstance(value, dict) else self.predict(value)

    def loss(self, batch, preds=None):
        prediction = self(batch["img"]) if preds is None else preds
        loss = torch.stack([prediction, prediction * 2, prediction * 3, prediction * 4])
        self.native_items = loss.detach()
        self.native_loss = loss * batch["img"].shape[0]
        return self.native_loss, self.native_items


def enhanced_model():
    from camo_fs.ultralytics_ext import FGModelLossMixin

    class Enhanced(FGModelLossMixin, NativeModel):
        segment_type = Segment

        def p3_channels(self):
            return 4

    model = Enhanced()
    model.configure_triplet(weight=0.2, margin=2.0, count=3, seed=2024)
    return model


def batch(batch_size=1):
    masks = torch.zeros(batch_size, 32, 32, dtype=torch.bool)
    masks[:, :16, :16] = True
    return {"img": torch.randn(batch_size, 3, 32, 32), "masks": masks,
            "batch_idx": torch.arange(batch_size), "camo_valid": torch.ones(batch_size, 32, 32, dtype=torch.bool)}


@pytest.mark.parametrize("batch_size", [1, 3])
def test_loss_preserves_native_return_contract(batch_size):
    model, data = enhanced_model(), batch(batch_size)
    result = model.loss(data)
    assert type(result) is tuple and len(result) == 2
    loss, items = result
    assert items is model.native_items
    assert loss.shape == items.shape == (4,)
    assert torch.equal(loss[1:], model.native_loss[1:])
    metrics = model.triplet_metrics
    assert metrics["sampled_triplets"] == 3 * batch_size
    assert metrics["raw_triplet"] > 0
    assert metrics["weighted_triplet"] == pytest.approx(0.2 * metrics["raw_triplet"] * batch_size)
    assert loss.sum().item() == pytest.approx(model.native_loss.sum().item() + metrics["weighted_triplet"])
    assert all(type(value) in (int, float) for value in metrics.values())


def test_auxiliary_native_relative_scale_is_invariant_to_batch_size(monkeypatch):
    import camo_fs.triplet as triplet
    from camo_fs.triplet import TripletResult

    # Isolate the weighting seam: both objectives use the same feature energy.
    # Native fake components sum to 10 times that mean; weight .2 => ratio .02.
    def sampler(features, *args):
        return TripletResult(features.square().mean(), 1, 0, torch.empty(0, 8, dtype=torch.int64))
    monkeypatch.setattr(triplet, "sample_and_loss", sampler)
    ratios = []
    for size in (1, 4):
        model, data = enhanced_model(), batch(size)
        total = model.loss(data)[0].sum().item()
        native = model.native_loss.sum().item()
        ratios.append((total - native) / native)
    assert ratios == pytest.approx([0.02, 0.02], abs=1e-6)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("auxiliary_inputs", ["present", "missing"])
def test_zero_weight_preserves_native_loss_and_items_without_sampler_or_rng(batch_size, auxiliary_inputs, monkeypatch):
    from copy import deepcopy
    import camo_fs.triplet as triplet

    model, data = enhanced_model(), batch(batch_size)
    baseline = NativeModel()
    baseline.load_state_dict(model.state_dict())
    model.configure_triplet(weight=0, margin=0.3, count=16, seed=2024)
    generator_state = model._fg_generator.get_state().clone()
    def forbidden(*args, **kwargs):
        pytest.fail("zero weight must bypass sampling")
    monkeypatch.setattr(triplet, "sample_and_loss", forbidden)
    expected = baseline.loss(deepcopy(data))
    # These are auxiliary-only inputs; zero weight must not inspect them.
    if auxiliary_inputs == "missing":
        for key in ("masks", "batch_idx", "camo_valid"):
            data.pop(key)
    result = model.loss(data)
    assert type(result) is tuple and len(result) == 2
    assert torch.equal(result[0], expected[0]) and torch.equal(result[1], expected[1])
    assert result[0] is model.native_loss and result[1] is model.native_items
    assert torch.equal(model._fg_generator.get_state(), generator_state)
    assert model.triplet_metrics == {"raw_triplet": 0.0, "weighted_triplet": 0.0,
                                      "sampled_triplets": 0, "skipped_instances": 0}
    assert not model.head._forward_pre_hooks


def test_enhanced_p3_gradient_and_baseline_bypasses_triplet(monkeypatch):
    import camo_fs.triplet as triplet

    real = triplet.sample_and_loss
    captured = []

    def sampler(features, *args, **kwargs):
        features.retain_grad()
        result = real(features, *args, **kwargs)
        captured.append((features, result.loss))
        return result

    monkeypatch.setattr(triplet, "sample_and_loss", sampler)
    model, data = enhanced_model(), batch()
    loss, _ = model.loss(data)
    feature, auxiliary = captured[0]
    # Prove the auxiliary alone reaches P3 and backbone (native loss cannot mask a detach).
    auxiliary.backward(retain_graph=True)
    assert feature.grad.abs().sum() > 0
    assert model.backbone.weight.grad.abs().sum() > 0
    loss.sum().backward()
    assert not model.head._forward_pre_hooks

    def forbidden(*args, **kwargs):
        pytest.fail("baseline constructed or executed triplet sampler")

    monkeypatch.setattr(triplet, "sample_and_loss", forbidden)
    monkeypatch.setattr(torch, "Generator", forbidden)
    baseline = NativeModel()
    baseline.loss(data)[0].sum().backward()
    assert not baseline.head._forward_pre_hooks


@pytest.mark.parametrize("failure", ["stale", "foreign", "no-forward", "bad-native", "missing-valid", "nonfinite"])
def test_extended_loss_fails_closed_and_next_batch_recovers(failure, monkeypatch):
    model, data = enhanced_model(), batch()
    preds = model(data["img"])
    if failure == "stale":
        model(data["img"])
    elif failure == "foreign":
        data["img"] = data["img"].clone()
    elif failure == "no-forward":
        model.loss(data, preds)
    elif failure == "bad-native":
        original = NativeModel.loss
        monkeypatch.setattr(NativeModel, "loss", lambda self, *args: (original(self, *args)[0][:3], torch.zeros(3)))
    elif failure == "missing-valid":
        data.pop("camo_valid")
    elif failure == "nonfinite":
        import camo_fs.triplet as triplet
        from camo_fs.triplet import TripletResult
        monkeypatch.setattr(triplet, "sample_and_loss", lambda f, *args: TripletResult(
            f.sum() * float("nan"), 1, 0, torch.empty(0, 8, dtype=torch.int64)))
    with pytest.raises((RuntimeError, ValueError, FloatingPointError), match="P3|contract|valid|finite"):
        model.loss(data, preds)
    assert model.triplet_metrics == {}
    assert not model.head._forward_pre_hooks
    # Error must consume pending state, including errors after native loss.
    with pytest.raises((RuntimeError, ValueError), match="P3|valid"):
        model.loss(batch(), preds)
    monkeypatch.undo()
    assert torch.isfinite(model.loss(batch())[0]).all()


@pytest.mark.parametrize("setting,value", [("weight", float("nan")), ("weight", -1),
    ("margin", 0), ("count", True), ("count", 0), ("seed", -1)])
def test_triplet_settings_are_validated_before_forward(setting, value):
    model = enhanced_model()
    settings = dict(weight=0.1, margin=0.3, count=16, seed=2024)
    settings[setting] = value
    with pytest.raises(ValueError, match=setting):
        model.configure_triplet(**settings)


def test_eval_preserves_native_loss_and_clears_training_capture():
    model, data = enhanced_model(), batch()
    preds = model(data["img"])
    model.eval()
    loss, items = model.loss(data, preds)
    assert loss is model.native_loss and items is model.native_items
    model.train()
    with pytest.raises(RuntimeError, match="P3"):
        model.loss(data, preds)


def test_version_gate_rejects_unverified_runtime():
    from camo_fs.ultralytics_ext import require_compatible_version

    assert require_compatible_version("8.3.228") == "8.3.228"
    for version in ("8.4.172", "8.3.227", "unknown"):
        with pytest.raises(RuntimeError, match="pip install ultralytics==8.3.228"):
            require_compatible_version(version)


@pytest.mark.parametrize("value", [0.5, float("nan"), float("inf")])
def test_loss_rejects_nonintegral_native_image_indices(value):
    model, data = enhanced_model(), batch()
    data["batch_idx"] = torch.tensor([value])
    with pytest.raises(ValueError, match="integral"):
        model.loss(data)


@pytest.mark.parametrize("contract", ["list", "items-grad", "nonfinite-items"])
def test_loss_rejects_changed_native_container_or_loss_items(contract, monkeypatch):
    model, data = enhanced_model(), batch()
    original = NativeModel.loss
    def changed(self, *args):
        loss, items = original(self, *args)
        if contract == "list":
            return [loss, items]
        if contract == "items-grad":
            return loss, loss
        return loss, items * float("nan")
    monkeypatch.setattr(NativeModel, "loss", changed)
    with pytest.raises((RuntimeError, FloatingPointError), match="contract|finite"):
        model.loss(data)


def test_capture_releases_feature_reference_after_success_or_failure():
    import weakref
    from camo_fs.ultralytics_ext import P3Capture

    model, images = TinyModel(), torch.randn(1, 3, 32, 32)
    capture = P3Capture(model, Segment, expected_channels=4)
    with capture.scope(images):
        model(images)
        feature = weakref.ref(capture.feature)
    assert feature() is None
    with pytest.raises(LookupError):
        with capture.scope(images):
            model(images)
            feature = weakref.ref(capture.feature)
            raise LookupError("failed step")
    assert feature() is None
