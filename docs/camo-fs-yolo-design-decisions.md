# CAMO-FS YOLO Design Decisions

This document records the decisions made before implementing the CAMO-FS
few-shot object-detection and instance-segmentation pipeline. It is the
implementation contract for this personal GitHub/Kaggle project.

## Goal and scope

Build a clean, rerunnable comparison using COCO-pretrained `yolo11n-seg.pt`
and the official CAMO-FS 1-, 2-, 3-, and 5-shot split files. Both variants use
YOLO11n-Seg for box detection and instance segmentation:

| Method | Training objective |
| --- | --- |
| `baseline` | Native Ultralytics box and segmentation losses |
| `fgbg-triplet` | The same native losses plus FG/BG Triplet Loss |

Each method/shot run starts independently from the original pretrained
checkpoint. The enhanced variant is a custom YOLO extension, not a
reproduction of another architecture. Both methods use the same prepared
training split, preprocessing, training settings, and official test split for
a given shot; the auxiliary loss is the intended methodological difference.

The project is not a research reproduction. It prioritizes correct use of the
official splits, clear scripts and artifacts, Kaggle compatibility, and enough
provenance to rerun an experiment. It does not require multi-seed averages or
research-grade hyperparameter controls by default.

## Non-negotiable dataset rules

- Use only the official class-specific `*_Kshot_split1.json` files for each
  selected training shot, and `camo5_test_split1.json` as the common final
  test set.
- Do not make random train/test splits, modify the official support-set
  membership, or force every class to contain exactly `K` images or instances.
- Every experiment starts independently from the same pretrained checkpoint;
  no shot or method may continue training from another run's weights.
- Never write generated artifacts under `/kaggle/input`; use
  `/kaggle/working`.
- Create segmentation labels, not detection-only labels.

## Observed official-split facts

The local audit of the supplied official annotations found the following.

| Split | Merged files | Unique images | Instances | Multi-polygon instances |
| --- | ---: | ---: | ---: | ---: |
| 1-shot | 47 | 47 | 47 | 5 |
| 2-shot | 47 | 94 | 94 | 10 |
| 3-shot | 47 | 141 | 141 | 16 |
| 5-shot | 47 | 197 | 235 | 38 |
| Official test | 1 | 2,655 | 3,108 | 585 |

There are no train/test overlaps by either `image_id` or file name in these
splits. All training categories match the canonical `(id, name)` pairs, and
all segmentations in the audited official annotations are polygon-based (no
RLE records).

Annotation IDs are not globally unique in the 5-shot files. IDs `826`, `386`,
and `387` are legitimately reused for distinct images/classes. Therefore,
global deduplication by `annotation_id` would incorrectly reduce the official
5-shot split from 235 instances. It is forbidden.

## Taxonomy, merging, and validation

`prepare_dataset.py` must derive the canonical 47-class taxonomy from:

```text
/kaggle/input/datasets/danhnt/camo-fs-dataset/few-shot-annotations/camo5_test_split1.json
```

It sorts `categories` by COCO `category_id`, creates contiguous YOLO indices
`0..46`, and writes the mapping to:

```text
/kaggle/working/camo_fs_yolo/category_mapping.json
```

Every training JSON must match that canonical `(category_id, category_name)`
taxonomy. A mismatch is an error.

When class-specific JSON files are merged, deduplicate images by their official
identity and detect duplicate annotations by their actual content, including
image, category, box, and polygon geometry. An annotation ID reused on a
different image or class is an audit event, not a duplicate. Exact duplicate
geometry in the same image/class must be reported.

Preparation is fail-fast. Before creating a YOLO dataset, it must reject and
report missing images, invalid image dimensions or boxes, unmapped categories,
unsupported segmentations, polygons with odd coordinate counts or too few
points, and invalid/out-of-range geometry. It must also assert that train and
official-test images do not overlap.

The audit report records the shot, counts, errors, error types, and relevant
`image_id`, `annotation_id`, and filenames. A failed integrity gate prevents
dataset creation and training.

## COCO-to-YOLO segmentation conversion

For polygon annotations, output normalized YOLO segmentation labels:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates are normalized by the image width and height. YOLO derives the
box from the mask polygon.

Official CAMO-FS annotations use the exact **-0.5** sentinel at the left/top
pixel boundary. Recognize only that value: bbox x/y may equal -0.5 or be
nonnegative, while positive finite width/height and the original right/bottom
extent checks remain strict. Polygon x/y equal to -0.5 become 0.0 in newly
allocated points. Other negative values (including -0.1 and -0.5001), right/bottom
overflow and non-finite geometry still fail. Check distinct vertices and area
after normalization, because boundary vertices can collapse. Source annotation
dictionaries and files remain unchanged; emitted YOLO coordinates stay in [0,1].

The original marked real-data test checked split counts and metadata audits,
which do not invoke polygon conversion. It therefore could pass while T04's
conversion preflight rejected this official convention. A separate parametrized
read-only conversion gate now covers official test plus every 1/2/3/5-shot split,
verifying source JSON hashes and in-memory records remain unchanged.

Multi-polygon instances must be converted with an Ultralytics-compatible
segment-merging strategy. No polygon may be silently discarded. The converter
logs every multi-polygon instance and provides sample ground-truth
visualizations for sanity checking. Because YOLO segmentation represents each
instance as a single polygon sequence, merging disconnected COCO polygons may
introduce connecting edges. This is an intentional, lossy topology-conversion
compromise and must be documented and visually sanity-checked. Exact COCO
topology preservation is not a reason to replace YOLO for this project.

## Enhanced method: FG/BG Triplet Loss

CAMO-FS objects can resemble their background in color and texture. The
enhanced method adds a class-agnostic feature-space objective to pull features
from the same GT object together and push them away from the image background.
It does not replace Ultralytics' native detection or segmentation losses.

```text
Image + GT instance masks
         |
         v
YOLO11n-Seg backbone and neck
         |
         +----------------------------+
         |                            |
         v                            v
native detection/segment heads   P3/8 neck feature map
         |                            |
         v                            v
native box + mask losses         GT-mask FG/BG sampling
                                      |
                                      v
                               FG/BG Triplet Loss
         |                            |
         +-------------+--------------+
                       |
                       v
       L_total = L_YOLO + triplet_weight * L_triplet
```

### Feature tensor and integration point

The proposed feature tensor is the **P3/8 neck output passed as the first
input to the `Segment` head**. In the currently published YOLO11 segmentation
model definition, the three head inputs are P3/8, P4/16, and P5/32 from the
neck (`[16, 19, 22]`); P3/8 is the highest-resolution head input and retains
spatial detail useful for small or camouflaged objects. At `imgsz=640`, its
expected spatial size is about `80 x 80`. The semantic head-input position is
the feature contract; the numeric layer indices are only a reference for the
inspected architecture. Use one feature level initially. [YOLO11 segmentation
model definition](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-seg.yaml).

The proposed project-owned extension overrides
`SegmentationTrainer.get_model()` to construct an extended
`SegmentationModel`. A forward pre-hook on the verified `Segment` head captures
its first input **without detaching it**. An override of the model's `loss()`
calls the native segmentation loss, then computes and adds the auxiliary loss
from the captured tensor and the training batch. Inference remains the native
YOLO path. This avoids editing installed Ultralytics files or reimplementing
its heads and losses. Before coding, confirm the installed target version's
trainer/model/loss signatures and checkpoint loading behavior; abort clearly
if the expected `Segment` head, three spatial inputs, P3/8 stride, or gradient
path is absent. [SegmentationTrainer source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/segment/train.py),
[SegmentationModel source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py).

The hook must clear its previous capture before every forward pass, capture
exactly one P3 tensor for the current pass, and never detach it. Assert the
expected batch, spatial, and channel dimensions. A missing capture or multiple
unexpected captures is a hard error. Clear the captured reference after the
training step so neither a stale feature nor an old computation graph is
retained for the next batch.

The development machine currently has no Ultralytics or PyTorch installation,
so this is an inspected upstream design, not a runtime-verified integration or
a pinned Ultralytics version. Record the exact installed version in every run
manifest and validate the feature contract with an integration test before
starting a Kaggle experiment.

The first enhanced implementation targets single-GPU training with `device=0`.
Multi-GPU/DDP support, distributed loss reduction, and hook state across
replicas are outside this version's scope. Use the same device setting for
baseline and enhanced runs being compared.

### GT masks, sampling, and loss

Use the GT instance masks from the same YOLO segmentation training batch.
Neither predicted masks nor a second dataset may determine triplets. Preserve
per-instance masks for both methods (proposed `overlap_mask=False`) so the
enhanced sampler can associate two foreground features with the same object;
the same mask setting is required for a fair baseline comparison.
[Ultralytics `overlap_mask` configuration](https://github.com/ultralytics/ultralytics/blob/main/docs/macros/train-args.md).

For the initial comparison, use `mosaic=0.0`, `mixup=0.0`, and
`copy_paste=0.0` for **both** methods. These composition augmentations would
combine multiple source images and complicate the meaning of a background
negative from the same image. Standard single-image flips, scale, translation,
and color augmentation may remain enabled, but their complete configuration
must match between paired baseline and enhanced runs and be recorded in the
manifest.

Project each transformed GT mask to P3/8 resolution and build a union mask of
all GT objects in the image. For each valid instance, draw an anchor and a
different positive feature location inside that instance, and a negative from
the **same image** outside the union of all objects. Background candidates must
also lie inside the valid transformed image region; letterbox padding is never
a negative. The implementation must propagate or reconstruct that region from
training preprocessing metadata, including augmentation, and test it before
training. Skip an instance when fewer than two usable foreground locations or
no valid background location remains; count skips and return a finite zero
auxiliary loss if the batch has no valid triplets. Sampling is bounded and
deterministic under the run seed where practical. A small boundary ignore band
may be added later only with an explicit CLI option.

Default proposals are `--triplets-per-instance 16`, `--triplet-margin 0.3`,
and `--triplet-weight 0.1`. Sample with replacement only when there are at
least two distinct foreground locations. L2-normalize the sampled feature
vectors; use Euclidean distance and PyTorch triplet margin loss:

```text
L_triplet = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
L_total   = L_YOLO + triplet_weight * L_triplet
```

Log native YOLO loss components, raw triplet loss, weighted contribution, and
combined loss where the target Ultralytics API permits. Record the sampled
triplet and skipped-instance counts. A non-finite auxiliary loss is a hard
error with diagnostic context; an empty valid sample set is not.

## Kaggle dataset layout

Copy selected images into `/kaggle/working` rather than rely on symlinks. The
official test split is copied once and shared by every shot:

```text
/kaggle/working/camo_fs_yolo/
├── category_mapping.json
├── test/
│   ├── images/
│   └── labels/
├── shot_1/
│   ├── train/images/
│   ├── train/labels/
│   └── data.yaml
├── shot_2/ ...
├── shot_3/ ...
└── shot_5/ ...
```

Each shot's `data.yaml` contains its own `train` split and the shared official
`test` split. **Never map `val` to the official test set.** Before training,
verify whether the selected Ultralytics version accepts a dataset YAML without
`val` when `val=False`. If it requires `val` for parsing, point `val` to the
same training split as a technical placeholder and retain `val=False`; any
library-triggered validation in that case must not be reported as a held-out
result. Final evaluation must explicitly use `split="test"` on the shared
official test split. Preparation fails if a target exists unless `--overwrite`
is provided; overwrite rebuilds the affected artifacts completely.

## CLI contract

The source of truth is four CLI scripts:

- `prepare_dataset.py`
- `train_yolo.py`
- `evaluate_yolo.py`
- `visualize_predictions.py`

Each supports `--shot {1,2,3,5,all}` where applicable. `--shot all` prepares
the shared test data once, then processes shots sequentially as independent
experiments. By default, the first failure stops the batch. With
`--continue-on-error`, later shots continue and every outcome is recorded.

A thin Kaggle notebook may install dependencies, import or clone the repo,
invoke these scripts, and display results. It must not duplicate business
logic.

Important configurable training arguments are `--weights`, `--epochs`,
`--imgsz`, `--batch`, `--device`, and `--seed`. The default seed is `2024`.
`--method {baseline,fgbg-triplet}` selects the objective. Enhanced runs also
accept `--triplet-weight`, `--triplet-margin`, and
`--triplets-per-instance` with the proposed defaults above; these options do
not activate auxiliary-loss code for `baseline`.
`--weights` defaults to `yolo11n-seg.pt`, but scripts must accept a local
Kaggle Input path such as:

```text
/kaggle/input/yolo11-seg-weights/yolo11n-seg.pt
```

If the path is unavailable and the checkpoint cannot be resolved locally, the
script fails with a clear error. The README must explain the Internet-on and
Internet-off Kaggle cases.

## Training, evaluation, and recovery

Train each shot for a fixed number of epochs with validation disabled where
Ultralytics allows it (`val=False`). Do not use the official test set for early
stopping, hyperparameter selection, or choosing `best.pt`.

For the reported benchmark result, evaluate `last.pt` on the official test
split after training with Ultralytics validation. Persist these values:

- `box.map`, `box.map50`, `box.map75`
- `seg.map`, `seg.map50`, `seg.map75`

All eight method/shot combinations load the same original pretrained
checkpoint. A run identity consists of model, method, shot, seed, and a
deterministic configuration fingerprint.
The fingerprint hashes the training-affecting configuration, including the
base weights, method, shot, seed, epochs, image size, batch size, and
augmentation configuration. For an enhanced run, it also includes triplet
weight, margin, and sampling count. The
manifest stores these fields and `config_hash` explicitly. Readable run paths
include the method and fingerprint, for example:

```text
/kaggle/working/runs/yolo11n-seg/baseline/shot_5/seed_2024/<config_hash>/
/kaggle/working/runs/yolo11n-seg/fgbg-triplet/shot_5/seed_2024/<config_hash>/
```

Existing run paths fail by default. `--overwrite` starts a new run from the
base pretrained checkpoint. Explicit `--resume` may continue only that same
run's checkpoint after validating that its complete run identity, important
configuration, and base weights agree with its manifest. It never resumes from
another shot or method.

## Reproducibility and artifacts

Preparation and training runs write manifests containing CLI arguments, shot,
seed, weights path, category mapping path, source JSON paths and checksums,
merged counts, generated-label counts, output path, timestamp, and actual
versions of Python, Ultralytics, Torch, Torchvision, NumPy, OpenCV, and
PyCOCOTools when present. Hash local weights when available. Enhanced-run
manifests also include the selected feature source, triplet parameters,
sampled-triplet count, and skipped-instance count. Baseline manifests record
`method=baseline` and no active auxiliary loss.

Scripts seed Python `random`, NumPy, PyTorch, and Ultralytics. They request
`deterministic=True` when supported. An inability to guarantee full CUDA
determinism is a warning recorded in the manifest, not a training failure.

After the enhanced integration test and a training run succeed on Kaggle, pin
the exact working Ultralytics version in `requirements.txt` as
`ultralytics==<verified-version>`. Until then, do not present an arbitrary
version as verified. The integration test must fail clearly if the installed
version is incompatible with the custom trainer/model extension. Torch/CUDA
may follow Kaggle's working runtime rather than being aggressively pinned;
record their actual versions in the run manifest.

Evaluation upserts `/kaggle/working/results/summary.csv` by the complete run
identity, including `config_hash`. It contains at least model, method, shot,
seed, epochs, image size, batch,
applicable triplet parameters, all required box/mask AP values, weights path,
`status`, and `error_message`. Failed runs remain visible with
`status=failed`, an error message, and empty/NaN metrics.
The results table compares baseline and enhanced runs at each of the four
shots on the same official test set. The test set is not used to choose
triplet parameters.

## Visualizations and tests

`visualize_predictions.py` saves 20 deterministic test images per run by
default, using `--seed 2024` and `--conf 0.25`. Each rendering shows predicted
box, segmentation mask, class name, and confidence. `--num-images`, `--conf`,
and `--seed` are configurable. Output folders are separated by run identity
under the run directory or `/kaggle/working/results/visualizations/`.

The enhanced method may additionally produce a small training-only debug
visualization of a GT mask and its sampled anchor, positive, and background
negative locations; it need not run for every batch.

The repository includes dataset-independent pytest tests with synthetic COCO
fixtures. They cover taxonomy mapping, coordinate normalization, invalid
polygons and dimensions, multi-polygon conversion, real duplicate detection,
annotation-ID reuse across images/classes, train/test overlap detection, and
fail-fast behavior.

Additional synthetic tests cover feature-to-mask projection, per-instance
anchor/positive selection, same-image background negatives, deterministic and
bounded sampling, padding exclusion, finite and satisfied triplet losses,
gradient propagation into features, empty-batch/skipped-instance behavior,
and method switching (baseline bypasses auxiliary loss; enhanced activates
it). Hook tests must detect stale or multiple captures, validate feature
dimensions, and verify that the captured graph is released after a step. An
integration test checks the selected feature source and native loss interface
against the target Ultralytics package and checkpoint before Kaggle training.
The synthetic tests run locally with:

```bash
pytest -q
```

## T09 provisional native integration note

The local CPU compatibility candidate is **Ultralytics 8.3.228**, inspected and
executed with Python 3.13.14 and PyTorch 2.14.1+cpu. The repository dependencies
remain unpinned until T10's Kaggle smoke gate. The extension rejects other
versions instead of selecting a baseline fallback. Inspection of 8.4.172 found
that its train `RandomPerspective.get_params(labels)` uses a centered affine
warp with an output size; its zero-augmentation pipeline is no longer the
integer `LetterBox` placement assumed by T07. Supporting that release needs a
separate deliberate geometry/interface revision.

The user's first actual Kaggle inspection reported **Torch 2.11.0+cu128,
Tesla T4, Ultralytics 8.4.172**. This is evidence of that environment, not support
for its native interfaces. T09 must explicitly install `ultralytics==8.3.228`
there before attempting the candidate integration. The guarded adapter is still
not Kaggle-certified; changing the version string would not support 8.4.172's
affine training geometry or five-component loss with dictionary loss-items.

Observed 8.3.228 interfaces:

- `SegmentationTrainer.get_model(self, cfg=None, weights=None, verbose=True)`
  creates `SegmentationModel(cfg, nc=data['nc'], ch=data['channels'], ...)` and
  calls `model.load(weights)` when provided. The extension directly constructs
  `FGSegmentationModel` with the same cfg/nc/ch/rank-aware verbosity and loads
  weights in the same way. It does not reassign a live model's `__class__`.
  Focused tests compare constructor invocation, every state tensor, YAML,
  names and strides against the native factory with and without weights.
- `SegmentationModel.loss(self, batch, preds=None)` delegates to the native
  segmentation criterion, returning exactly `(loss_vector, loss_items)` with
  both tensors shaped `[4]` in box/seg/cls/dfl order. The native criterion
  returns `loss * batch_size, loss.detach()`; trainer `loss.sum()` produces
  its scalar. Therefore the contribution added once to vector entry zero is
  `triplet_weight * raw_triplet_mean * actual_batch_image_count`, not the
  unscaled weighted mean. Detached loss-items are returned unchanged.
  `weighted_triplet` reports that actual batch-scaled objective contribution.
  Raw/weighted auxiliary and sample/skip counts are plain
  numbers in `model.triplet_metrics` for a separate logging callback.
  A zero triplet weight delegates directly to native prediction/loss without
  capture, sampling, auxiliary metadata checks or sampler RNG advancement;
  it reports zero auxiliary values/counts. Batch-one and batch-three tests
  establish native parity and nonzero auxiliary P3/backbone gradients; an
  isolated weighting test keeps auxiliary/native ratio fixed at batch four.
- One semantic `Segment` has head strides `[8, 16, 32]`. The first input on
  random YOLO11n-Seg at `[1, 3, 64, 64]` is `[1, 64, 8, 8]`. Channels come
  from the verified head branch, not a hard-coded neck layer index. Capture
  hooks exist only around prediction; pending current-batch features are
  consumed and released by loss, including errors. External/stale predictions
  and duplicate captures fail.
- `BaseDataset` supplies `ori_shape`, preloaded `resized_shape`, and raw
  `ratio_pad=(height_gain,width_gain)`. Native `LetterBox` nests the actual
  integer `(left,top)` padding around those raw gains; `RandomPerspective`
  subsequently removes `ratio_pad`. The adapter therefore observes letterbox
  before that removal. It runs the unchanged native transform on the image and
  instances, plus the same native transform on an all-one/zero-padding probe.
  Measured content extents and verified padding become T07's explicit
  width/height ratios; raw library ratios are never passed directly to T07.
- Zero `RandomPerspective` is checked for an identity matrix and unchanged
  output dimensions. Native `RandomFlip` makes its usual random draw; an
  instance proxy observes the actual `fliplr` call and flips the valid mask
  with it. Mixed images, other geometry and spatial Albumentations are rejected.
  Native `Format` supplies per-instance `masks` and **float32** `batch_idx`;
  finite integral indices are verified before an int64 copy enters the sampler.
  The native batch indices are preserved. Native collation keeps project
  metadata in tuples; trainer preprocessing validates and stacks validity.

CPU integration covers native image/mask/bbox parity, odd padding and flips,
T04 synthetic multi-polygon preparation/loader acceptance, finite native plus
auxiliary loss, auxiliary-only P3/backbone gradients, and random-weight
serialization/reload through `YOLO`. No attached pretrained checkpoint, real
CAMO-FS imagery or training epoch was used. These checks do not certify Kaggle.

After the boundary correction, the separate read-only official count/conversion
suite passed six tests against local source files through a temporary contract
path view. The complete local suite passed 478 tests; only the opt-in Kaggle
attached-checkpoint test skipped. This adds official JSON geometry evidence,
not real-data model inference or CUDA training evidence.

On Kaggle, attach the original `yolo11n-seg.pt`, prepare a shot with T04, install
the **candidate** explicitly in that runtime, then run:

```bash
python -m pip install 'ultralytics==8.3.228'
export CAMO_FS_DATA_ROOT=/kaggle/input/datasets/danhnt/camo-fs-dataset
export CAMO_FS_WORK_ROOT=/kaggle/working
export CAMO_FS_INTEGRATION_WEIGHTS=/kaggle/input/<attached-weights>/yolo11n-seg.pt
export CAMO_FS_INTEGRATION_SHOT=1
python -m pytest -q -m integration -s
```

Without `CAMO_FS_INTEGRATION_WEIGHTS`, the attached-checkpoint gate explicitly
skips. When set, missing CUDA/data/checkpoint, incompatible versions, corrupt
labels, dropped instances, absent geometry or missing gradients fail. The gate
audits T04 provenance, uses a prepared **train** multi-polygon example, and
prints actual runtime signatures/shapes, loss/sample metrics and gradient proof.
Capture the log on Kaggle before checking off T09's target gates. T10 still owns
CLI enhanced dispatch, one-epoch smoke, trained `last.pt` paired predictions and
the final dependency pin.

## Explicitly excluded behavior

- Random 80/20 splitting or test-set reuse during training.
- Hard-coded Windows dataset paths or hard-coded 47-class tables.
- Detection-only labels, silently dropped polygons, or global annotation-ID
  deduplication.
- Cross-shot fine-tuning or automatic resume.
- Initializing the enhanced method from a fine-tuned baseline run, or
  replacing the native YOLO losses with the auxiliary objective.
- Memory banks, hard-negative mining systems, multi-scale triplet losses,
  class-aware metric learning, or additional contrastive objectives.
- Multi-GPU/DDP training for the initial enhanced implementation.
- Silent overwrite, silent skip, or silent omission of failed runs.
