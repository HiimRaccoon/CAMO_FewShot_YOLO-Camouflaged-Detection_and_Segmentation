"""T08 behavior at the public sampler/loss seam, without data or YOLO."""

import importlib

import pytest
import torch


def _api():
    return importlib.import_module("camo_fs.triplet")


def _sample(features, masks, batch_idx, valid, *, count=16, margin=0.3, seed=2024, **kwargs):
    return _api().sample_and_loss(
        features, masks, batch_idx, valid, count, margin,
        torch.Generator(device="cpu").manual_seed(seed), **kwargs,
    )


def test_samples_distinct_foreground_pair_and_same_image_background():
    # Break caught: anchor/positive are identical or a negative is inside GT.
    features = torch.tensor([[[[1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]], requires_grad=True)
    masks = torch.tensor([[[True, True, False]]])
    result = _sample(features, masks, torch.tensor([0]), torch.ones((1, 1, 3), dtype=torch.bool), count=4)

    assert result.sampled_triplets == 4
    assert result.skipped_instances == 0
    assert result.sampled_positions.shape == (4, 8)
    assert result.sampled_positions.dtype == torch.int64
    for instance, image, ay, ax, py, px, ny, nx in result.sampled_positions.tolist():
        assert (instance, image, ay, py, ny, nx) == (0, 0, 0, 0, 0, 2)
        assert {ax, px} == {0, 1}
    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss)


def test_negative_is_outside_union_of_all_instances_in_its_own_image():
    # Break caught: another object, or an object in another image, changes BG identity.
    masks = torch.tensor([
        [[True, True, False, False, False]],
        [[False, False, True, True, False]],
        [[False, False, False, True, True]],
    ])
    batch_idx = torch.tensor([0, 0, 1])
    features = torch.arange(20, dtype=torch.float32).reshape(2, 2, 1, 5).requires_grad_()
    result = _sample(features, masks, batch_idx, torch.ones((2, 1, 5), dtype=torch.bool), count=20)

    assert result.sampled_triplets == 60
    assert result.skipped_instances == 0
    for instance, image, ay, ax, py, px, ny, nx in result.sampled_positions.tolist():
        assert image == [0, 0, 1][instance]
        assert masks[instance, ay, ax] and masks[instance, py, px]
        assert (ay, ax) != (py, px)
        assert not masks[batch_idx == image, ny, nx].any()


def test_negative_excludes_padding():
    # Break caught: nearest validity admits a partly padded cell as BG or FG.
    features = torch.ones((1, 2, 1, 4), requires_grad=True)
    masks = torch.tensor([[[True, True, False, False]]])
    valid = torch.tensor([[[True] * 27 + [False] * 5]], dtype=torch.bool)
    result = _sample(features, masks, torch.tensor([0]), valid, count=32)

    assert result.sampled_triplets == 32
    assert result.sampled_positions[:, 7].tolist() == [2] * 32


def test_transformed_masks_project_by_nearest_neighbor_to_actual_feature_shape():
    # Break caught: masks remain at loader size, or bilinear/any-valid creates FG.
    features = torch.ones((1, 2, 2, 3), requires_grad=True)
    masks = torch.tensor([[
        [True, False, True, False, False, False],
        [False] * 6,
        [False] * 6,
        [False] * 6,
    ]], dtype=torch.bool)
    valid = torch.ones((1, 4, 6), dtype=torch.bool)
    result = _sample(features, masks, torch.tensor([0]), valid, count=8)

    assert result.sampled_triplets == 8
    for _, _, ay, ax, py, px, ny, nx in result.sampled_positions.tolist():
        assert {(ay, ax), (py, px)} == {(0, 0), (0, 1)}
        assert (ny, nx) not in {(0, 0), (0, 1)}


@pytest.mark.parametrize(
    "mask,valid",
    [
        ([[True, False, False]], [[True, True, True]]),
        ([[True, True, True]], [[True, True, True]]),
        ([[True, True, False]], [[True, False, True]]),
        ([[True, True, False]], [[True, True, False]]),
    ],
)
def test_tiny_or_no_background_instances_are_skipped_with_differentiable_zero(mask, valid):
    # Break caught: RNG draws from an empty set, identical FG pair, or detached zero.
    features = torch.ones((1, 2, 1, 3), requires_grad=True)
    result = _sample(features, torch.tensor([mask]), torch.tensor([0]), torch.tensor([valid]))

    assert result.skipped_instances == 1
    assert result.sampled_triplets == 0
    assert result.sampled_positions.shape == (0, 8)
    assert result.sampled_positions.dtype == torch.int64
    assert result.loss.item() == 0
    result.loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))


@pytest.mark.parametrize("batch_size", [0, 2])
def test_empty_batch_contributes_finite_differentiable_zero(batch_size):
    features = torch.full((batch_size, 2, 2, 3), 3e38, requires_grad=True)
    result = _sample(features, torch.empty((0, 4, 6), dtype=torch.bool),
                     torch.empty(0, dtype=torch.int64), torch.ones((batch_size, 4, 6), dtype=torch.bool))

    assert result.skipped_instances == result.sampled_triplets == 0
    assert result.loss.item() == 0
    assert result.loss.device == features.device
    result.loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))


def test_tiny_object_lost_by_nearest_projection_still_cannot_be_background():
    # Break caught: downsampling erases a tiny GT object and turns its cell into BG.
    masks = torch.tensor([
        [[True, True, True, True, False, False, False, False]],
        [[False, False, False, False, False, True, False, False]],
    ])
    features = torch.ones((1, 2, 1, 4), requires_grad=True)
    result = _sample(features, masks, torch.tensor([0, 0]), torch.ones((1, 1, 8), dtype=torch.bool), count=32)

    assert result.skipped_instances == 1
    assert result.sampled_triplets == 32
    assert result.sampled_positions[:, 0].tolist() == [0] * 32
    assert result.sampled_positions[:, 7].tolist() == [3] * 32


@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_loss_is_mean_euclidean_margin_on_l2_normalized_features(scale):
    # Both FG vectors normalize to (1,0), BG to (0.8,0.6): distance sqrt(0.4).
    # Break caught: raw feature magnitudes, squared distance, summed loss, or omitted margin.
    features = torch.tensor([[[[3.0, 7.0, 4.0]], [[0.0, 0.0, 3.0]]]], dtype=torch.float64) * scale
    result = _sample(features, torch.tensor([[[True, True, False]]]), torch.tensor([0]),
                     torch.ones((1, 1, 3), dtype=torch.bool), count=5, margin=0.9)
    assert result.loss.item() == pytest.approx(0.2675444679663241, abs=1e-10)


def test_satisfied_triplets_have_zero_loss():
    features = torch.tensor([[[[2.0, 9.0, -4.0]], [[0.0, 0.0, 0.0]]]], requires_grad=True)
    result = _sample(features, torch.tensor([[[True, True, False]]]), torch.tensor([0]),
                     torch.ones((1, 1, 3), dtype=torch.bool))
    assert result.loss.item() == 0


def test_active_triplet_loss_backpropagates_into_features():
    features = torch.tensor([[[[1.0, 0.0, 0.8]], [[0.0, 1.0, 0.6]]]], requires_grad=True)
    result = _sample(features, torch.tensor([[[True, True, False]]]), torch.tensor([0]),
                     torch.ones((1, 1, 3), dtype=torch.bool))
    assert result.loss.item() > 0
    result.loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_vectors_in_low_precision_produce_finite_float32_loss_and_gradients(dtype):
    # Break caught: normalization epsilon underflows in AMP and turns zeros into NaN.
    features = torch.zeros((1, 2, 1, 3), dtype=dtype, requires_grad=True)
    result = _sample(features, torch.tensor([[[True, True, False]]]), torch.tensor([0]),
                     torch.ones((1, 1, 3), dtype=torch.bool))
    assert result.loss.dtype == torch.float32
    assert result.loss.item() == pytest.approx(0.3)
    result.loss.backward()
    assert torch.isfinite(features.grad).all()


def test_fixed_generator_is_bounded_and_independent_of_global_rng_and_input_mutation():
    # Break caught: global RNG dependence, changed inputs, or count scales with area.
    features = torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 8)
    masks = torch.zeros((1, 2, 8), dtype=torch.bool)
    masks[:, :, :4] = True
    batch_idx, valid = torch.tensor([0]), torch.ones((1, 2, 8), dtype=torch.bool)
    snapshots = [tensor.clone() for tensor in (features, masks, batch_idx, valid)]
    rng_before = torch.random.get_rng_state().clone()
    first = _sample(features, masks, batch_idx, valid, count=3)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.rand(100)  # Advance unrelated global state.
    second = _sample(features, masks, batch_idx, valid, count=3)
    third = _sample(features, masks, batch_idx, valid, count=3, seed=7)
    assert first.sampled_triplets == 3
    assert torch.equal(first.sampled_positions, second.sampled_positions)
    assert first.loss.item() == second.loss.item()
    assert not torch.equal(first.sampled_positions, third.sampled_positions)
    for original, snapshot in zip((features, masks, batch_idx, valid), snapshots):
        assert torch.equal(original, snapshot)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("with_instances", [False, True])
def test_nonfinite_features_fail_with_shot_method_and_batch_context(value, with_instances):
    # Break caught: NaN/Inf silently contributes auxiliary loss, including a skipped batch.
    features = torch.ones((1, 2, 1, 3), requires_grad=True)
    with torch.no_grad():
        features[0, 0, 0, 0] = value
    masks = torch.tensor([[[True, True, False]]]) if with_instances else torch.empty((0, 1, 3), dtype=torch.bool)
    batch_idx = torch.tensor([0]) if with_instances else torch.empty(0, dtype=torch.int64)
    with pytest.raises(FloatingPointError) as caught:
        _sample(features, masks, batch_idx, torch.ones((1, 1, 3), dtype=torch.bool),
                context={"shot": 5, "method": "fgbg-triplet", "batch": 7})
    assert "non-finite" in str(caught.value)
    assert "shot=5" in str(caught.value)
    assert "method=fgbg-triplet" in str(caught.value)
    assert "batch=7" in str(caught.value)


def test_nonfinite_loss_fails_with_context_even_when_features_are_finite():
    # Break caught: an unrepresentable margin returns infinite float32 loss.
    features = torch.ones((1, 2, 1, 3))
    with pytest.raises(FloatingPointError, match="shot=1.*batch=3"):
        _sample(features, torch.tensor([[[True, True, False]]]), torch.tensor([0]),
                torch.ones((1, 1, 3), dtype=torch.bool), margin=1e300,
                context={"shot": 1, "batch": 3})


@pytest.mark.parametrize("field,value", [
    ("count", 0), ("count", -1), ("count", 1.5), ("count", True),
    ("margin", 0), ("margin", -0.3), ("margin", float("nan")),
    ("margin", float("inf")), ("margin", True), ("margin", "0.3"),
    ("generator", None), ("context", "shot5"),
])
def test_invalid_sampling_settings_are_rejected_before_sampling(field, value):
    arguments = {"count": 4, "margin": 0.3, "generator": torch.Generator(), field: value}
    with pytest.raises(ValueError, match=field):
        _api().sample_and_loss(torch.ones((1, 2, 1, 3)), torch.tensor([[[True, True, False]]]),
                               torch.tensor([0]), torch.ones((1, 1, 3), dtype=torch.bool), **arguments)


@pytest.mark.parametrize("field,value", [
    ("features", torch.ones((2, 1, 3))),
    ("features", torch.ones((1, 0, 1, 3))),
    ("features", torch.ones((1, 2, 0, 3))),
    ("features", torch.ones((1, 2, 1, 3), dtype=torch.int64)),
    ("masks", torch.ones((1, 1, 1, 3), dtype=torch.bool)),
    ("masks", torch.empty((1, 0, 3), dtype=torch.bool)),
    ("masks", torch.tensor([[[0.0, 0.5, 1.0]]])),
    ("masks", torch.tensor([[[0.0, float("nan"), 1.0]]])),
    ("masks", torch.tensor([[[0, 2, 1]]])),
    ("batch_idx", torch.tensor([0.0])),
    ("batch_idx", torch.tensor([True])),
    ("batch_idx", torch.tensor([[0]])),
    ("batch_idx", torch.empty(0, dtype=torch.int64)),
    ("batch_idx", torch.tensor([-1])),
    ("batch_idx", torch.tensor([1])),
    ("valid", torch.ones((1, 1, 3))),
    ("valid", torch.ones((1, 3), dtype=torch.bool)),
    ("valid", torch.ones((2, 1, 3), dtype=torch.bool)),
    ("valid", torch.ones((1, 1, 2), dtype=torch.bool)),
])
def test_misaligned_or_nonbinary_batch_inputs_are_rejected(field, value):
    arguments = {"features": torch.ones((1, 2, 1, 3)), "masks": torch.tensor([[[True, True, False]]]),
                 "batch_idx": torch.tensor([0]), "valid": torch.ones((1, 1, 3), dtype=torch.bool), field: value}
    with pytest.raises(ValueError, match=field):
        _sample(**arguments)


def test_masks_below_feature_resolution_project_without_losing_background_identity():
    # Break caught: requiring GT masks to have exactly the P3 or input resolution.
    masks = torch.tensor([[[1.0, 0.0]]])
    features = torch.ones((1, 2, 2, 4), requires_grad=True)
    result = _sample(features, masks, torch.tensor([0]), torch.ones((1, 2, 4), dtype=torch.bool))
    assert result.sampled_triplets == 16
    for _, _, ay, ax, py, px, ny, nx in result.sampled_positions.tolist():
        assert ax in (0, 1) and px in (0, 1)
        assert nx in (2, 3)
        assert (ay, ax) != (py, px)


def test_horizontal_flip_keeps_mask_feature_alignment():
    # Break caught: reconstructing centered validity/original COCO mask after a flip.
    from camo_fs.valid_region import valid_letterbox_mask

    valid = valid_letterbox_mask((16, 11), (16, 16))[None]
    masks = torch.zeros((1, 16, 16), dtype=torch.bool)
    masks[0, :, 4:6] = True
    image = torch.zeros((1, 16, 16))
    image[masks] = 5.0
    # A synthetic stride-2 feature represents the already transformed image.
    features = torch.stack((image[:, ::2, ::2], torch.ones((1, 8, 8))), dim=1).requires_grad_()
    original = _sample(features, masks, torch.tensor([0]), valid)
    flipped_features = features.flip(-1)
    flipped = _sample(flipped_features, masks.flip(-1), torch.tensor([0]), valid.flip(-1))

    for result, current_features, current_valid, foreground_x in (
        (original, features, valid, 2), (flipped, flipped_features, valid.flip(-1), 5),
    ):
        assert result.sampled_triplets == 16
        for _, _, ay, ax, py, px, ny, nx in result.sampled_positions.tolist():
            assert ax == px == foreground_x
            assert ay != py
            assert current_features[0, 0, ay, ax] == current_features[0, 0, py, px] == 5
            assert current_features[0, 0, ny, nx] == 0
            assert current_valid[0, ny * 2:ny * 2 + 2, nx * 2:nx * 2 + 2].all()


def test_skipped_instance_does_not_consume_rng_or_block_other_instances():
    features = torch.ones((1, 2, 1, 5), requires_grad=True)
    valid = torch.ones((1, 1, 5), dtype=torch.bool)
    usable = torch.tensor([[[True, True, False, False, False]]])
    # Empty transformed mask represents an instance with no remaining FG.
    masks = torch.cat((torch.zeros_like(usable), usable))
    combined = _sample(features, masks, torch.tensor([0, 0]), valid, count=2)
    alone = _sample(features, usable, torch.tensor([0]), valid, count=2)
    assert combined.skipped_instances == 1
    assert combined.sampled_triplets == 2
    assert combined.sampled_positions[:, 0].tolist() == [1, 1]
    assert torch.equal(combined.sampled_positions[:, 1:], alone.sampled_positions[:, 1:])


def test_sampling_uses_feature_device_under_an_unrelated_default_device_context():
    features = torch.ones((1, 2, 1, 3), requires_grad=True)
    masks = torch.tensor([[[True, True, False]]])
    batch_idx, valid = torch.tensor([0]), torch.ones((1, 1, 3), dtype=torch.bool)
    with torch.device("meta"):
        result = _sample(features, masks, batch_idx, valid)
    assert result.loss.device == features.device
    assert result.sampled_positions.device == features.device
    assert result.sampled_triplets == 16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_empty_auxiliary_loss_also_uses_float32(dtype):
    features = torch.ones((1, 2, 1, 3), dtype=dtype, requires_grad=True)
    result = _sample(features, torch.empty((0, 1, 3), dtype=torch.bool), torch.empty(0, dtype=torch.int64),
                     torch.ones((1, 1, 3), dtype=torch.bool))
    assert result.loss.dtype == torch.float32
    result.loss.backward()
    assert torch.equal(features.grad, torch.zeros_like(features))


def test_large_finite_features_remain_l2_normalized_instead_of_overflowing_norm():
    # Same normalized geometry as the worked margin test, near float32's limit.
    features = torch.tensor([[[[3e38, 2e38, 2.4e38]], [[0.0, 0.0, 1.8e38]]]])
    result = _sample(features, torch.tensor([[[True, True, False]]]), torch.tensor([0]),
                     torch.ones((1, 1, 3), dtype=torch.bool), margin=0.9)
    assert result.loss.item() == pytest.approx(0.2675444679663241, abs=1e-6)


@pytest.mark.parametrize("source_height", [1, 2])
def test_fractional_upsampling_and_mixed_resize_exclude_every_gt_intersection(source_height):
    # Width 3->5: cells 1/2/3 all intersect source GT pixel [1,2).
    # Height 2->1 also includes GT in either source row (mixed down/up axes).
    masks = torch.zeros((1, source_height, 3), dtype=torch.bool)
    masks[0, 0, 1] = True
    result = _sample(torch.ones((1, 2, 1, 5)), masks, torch.tensor([0]),
                     torch.ones((1, 1, 5), dtype=torch.bool), count=32)
    assert result.sampled_triplets == 32
    for _, _, ay, ax, py, px, ny, nx in result.sampled_positions.tolist():
        assert {ax, px} == {2, 3}
        assert nx in (0, 4)


@pytest.mark.parametrize("small_value", [0.0, 1e-7])
def test_float16_mixed_zero_and_nearzero_vectors_backpropagate_finite_gradients(small_value):
    # Break caught: finite float32 loss hides Inf when normalized gradients cast to half.
    features = torch.tensor([[[[small_value, 1.0, 0.0]], [[small_value, 0.0, 1.0]]]],
                            dtype=torch.float16, requires_grad=True)
    result = _sample(features, torch.tensor([[[True, True, False]]]), torch.tensor([0]),
                     torch.ones((1, 1, 3), dtype=torch.bool))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0
