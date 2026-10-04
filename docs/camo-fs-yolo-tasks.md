# CAMO-FS YOLO Implementation Tasks

**Plan:** [camo-fs-yolo-plans.md](camo-fs-yolo-plans.md)  
**Spec:** [camo-fs-yolo-spec.md](camo-fs-yolo-spec.md)  
**Status:** In progress — T01 through T04 synthetic gates complete; real-data audit attempted, blocked by local image layout; Kaggle preparation unverified. Check a step only after its evidence exists.

## Dependency graph

```text
T01 paths/package
  ├─ T02 audit/merge ─ T03 polygon conversion ─ T04 materialization ─┐
  ├─ T05 identity/provenance ───────────────────────────────────────────┤
  └─ T07 valid region ─ T08 triplet sampling/loss ────────────────────┤
                                                                      v
                              T06 baseline training ─────────────── T09 extension
                                     │                                │
                                     └───────────────┬────────────── T10 smoke
                                                     │                │
                                                     v                v
                                                  T11 evaluation ─ T12 visualization
                                                                  │
                                                                  v
                                                              T13 handoff
```

The graph shows prerequisites, not an instruction to run experiments concurrently. Kaggle training stays sequential. Every task below has a focused red/green test or, for Kaggle gates, a recorded runtime proof. After each code task, run `pytest -q` in the supported Python environment and report any failures by name. Make focused commits only when the implementation session is ready to commit; do not stage user-owned `data/`, notebook, `.agents/`, or unrelated files.

## T01 — Package, safe paths, and test harness

**Depends on:** none.  
**Files:** Create `pyproject.toml`, `requirements.txt` (provisional, no fake exact Ultralytics pin), `.gitignore`, `camo_fs/__init__.py`, `camo_fs/paths.py`, `tests/test_paths.py`.  
**Produces:** `DatasetPaths.from_root(data_root: Path, work_root: Path)` and an editable package.  
**Acceptance:** Input paths follow the official Kaggle tree by default; output roots are never under `/kaggle/input`; a caller can override roots for synthetic/local tests.

- [x] Write `test_default_kaggle_paths`, `test_custom_roots_for_fixture`, and `test_rejects_output_inside_input`. Assert every resolved path, not just object construction.
- [x] Run `pytest -q tests/test_paths.py`; confirm the tests fail because the package/API does not exist.
- [x] Implement the immutable `DatasetPaths` interface. Add package/test metadata and generated-artifact ignores; do not ignore source docs or silently discard existing untracked work.
- [x] Run `pytest -q tests/test_paths.py`, then `pytest -q`; record actual pass/fail output.
- [x] Check `git status --short` to ensure `data/` and checkpoint artifacts were not staged or written.

## T02 — Canonical taxonomy and read-only shot audit

**Depends on:** T01.  
**Files:** Create `camo_fs/annotations.py`, `tests/test_annotations.py`, `tests/test_real_data_audit.py` (marked `real_data`, skipped without local CAMO-FS).  
**Produces:** `Taxonomy.from_test_json(path)`, `AuditReport`, `audit_shot(shot, paths, taxonomy)`; report holds deduplicated records, source paths, errors, warnings, and counts without writing data.  
**Acceptance:** Global category mapping is sorted and contiguous; every shot requires exactly the selected official file pattern for all 47 categories; distinct objects with reused IDs survive; exact duplicate content is deduplicated from merged records, reported as an integrity error, and prevents materialization; conflicting geometry/metadata and train/test overlap fail.

- [x] Write synthetic COCO fixture tests for unsorted/noncontiguous category IDs, taxonomy mismatch, missing or extra shot files, duplicated image metadata, exact duplicate content, conflicting same-image geometry, and train/test overlap. Include `test_reused_id_in_different_image_is_retained` with two objects sharing ID but different image/class and expected count `2`.
- [x] Run `pytest -q tests/test_annotations.py`; confirm expected missing-behavior failures.
- [x] Implement taxonomy and audit in one pass over selected files. Key images by `image_id`; require their filename/dimensions to agree across files and flag a filename mapped to different IDs. Use full annotation content/geometry keys; record reused IDs as warnings, never as global dedup keys. Validate file existence and declared versus actual image dimensions.
- [x] Run focused tests and then `pytest -q`; verify no test depends on the real CAMO-FS data.
- [ ] Read-only audit the local official JSON if present through `pytest -q -m real_data`; assert observed 1/2/3/5-shot counts. Do not rewrite any `data/` file. Keep this test separate from the synthetic suite's mandatory gate.

Local read-only evidence (2026-10-04, Python 3.13 / pytest 9.1.1):

- With `CAMO_FS_DATA_ROOT` set to the local `data/` directory,
  `py -3.13 -m pytest -q -m real_data` returned **1 failed, 38 deselected**.
  `test_official_splits_match_audited_counts` stopped at the 1-shot integrity
  assertion with `missing_image`: the contract requires `data/images/images/`,
  while local JPEG files are directly under `data/images/`. No path fallback
  was introduced and no source data was moved or rewritten.
- A separate read-only diagnostic using the unchanged `audit_shot` API
  asserted 47 categories and 47 source files per shot. Observed
  images/annotations/multi-polygon counts were `47/47/5`, `94/94/10`,
  `141/141/16`, and `197/235/38` for 1/2/3/5-shot. It reported respectively
  47/94/141/197 `missing_image` errors and no other error codes; 5-shot reused
  IDs were `386`, `387`, and `826`. These JSON counts do **not** constitute
  a passed image-integrity gate.
- The official test JSON contained 2,655 images, 3,108 annotations, and 585
  multi-polygon instances, with no RLE records. This was a JSON count check,
  not a full shared-test image/geometry audit or prepared-artifact check.
- With `CAMO_FS_DATA_ROOT` unset, the independent mandatory gate
  `py -3.13 -m pytest -q` returned **38 passed, 1 skipped**. Before/after
  source-file inventories (paths, sizes, modification times) and SHA-256
  checksums of every source JSON were unchanged. Keep the real-data item
  unchecked until the contract-layout audit passes.

## T03 — Polygon validation and conversion

**Depends on:** T02.  
**Files:** Create `camo_fs/segments.py`, `tests/test_segments.py`.  
**Produces:** `annotation_to_yolo(annotation, image, category_to_index) -> (line, multi_polygon)` and `DataIntegrityError`.  
**Acceptance:** Every valid instance produces one normalized segmentation line. Multi-polygon components all contribute to the one YOLO sequence; the topology compromise is logged/visualized later. RLE, odd-length polygons, fewer than three points, invalid bbox or image size, non-finite/out-of-range coordinates, and unmapped categories raise a named error.

- [x] Write tests for exact normalization values, contiguous class mapping, simple polygon, two disconnected polygons, and each malformed input. For the two-component case, assert coordinates from **both** components survive and the output is one instance line.
- [x] Run `pytest -q tests/test_segments.py`; verify the expected red state.
- [x] Implement the validated conversion and a deterministic nearest-boundary segment joining policy compatible with a single Ultralytics polygon sequence. Do not discard any component; document connecting-edge topology loss in a code comment.
- [x] Run `pytest -q tests/test_segments.py` and `pytest -q`; check coordinates stay within `[0,1]` and contain an even number of values.
- [x] Add a small synthetic ground-truth render assertion or saved fixture view proving the merged outline can be inspected without real data.

## T04 — Audit-first materialization and preparation CLI

**Depends on:** T02, T03.  
**Files:** Create `camo_fs/prepare.py`, `scripts/prepare_dataset.py`, `tests/test_prepare.py`; extend `camo_fs/annotations.py` with the shared-test audit.

**Consumes:** `DatasetPaths`, `AuditReport`, `annotation_to_yolo`.  
**Produces:** shared `test/images|labels`, `shot_K/train/images|labels`, `shot_K/data.yaml`, `category_mapping.json`, per-shot manifests, `/kaggle/working/results/audit.json`.  
**Acceptance:** Preparation never writes into input; exact shot `all` invokes `1,2,3,5` and copies test once; output collision requires `--overwrite`; no stale label survives overwrite. Without `--continue-on-error`, preflight-audit the shared test and **all selected shots** before any prepared dataset write; if any audit fails, leave `test/`, `category_mapping.json`, and every selected `shot_K/` uncreated or unchanged (audit/error reports may be written). With `--continue-on-error`, audit/materialize shots independently and record every failure, but a failed shared-test audit blocks all materialization.

- [x] Write fixture tests for `--shot 5`, `--shot all`, common test reuse, image copy once per split, expected label count, audit-before-write, fail-fast versus `--continue-on-error`, overwrite cleanup, and `test_val_never_points_to_test` (both YAML-without-val and technical train-placeholder modes). Include `test_all_fail_fast_audits_every_shot_before_materialization`: shot 1 and 2 pass, shot 3 fails, and no shared test or shot artifact is created. Include the complementary continue-on-error case: successful shots materialize once, failed shots do not, and all outcomes are reported. A failed shared-test audit creates no prepared artifact in either mode.
- [x] Run `pytest -q tests/test_prepare.py`; confirm red for missing preparation behavior.
- [x] Implement `prepare_selected(...)` with target-scoped temporary staging and explicit target checks. In fail-fast mode, finish the complete shared-test plus selected-shot audit preflight before creating/staging any prepared artifact; materialize only after every audit passes. In continue-on-error mode, audit and materialize each shot separately after the shared test passes. Write audit/error reports on failure; only promote fully valid staged splits. YAML has absolute `train`/`test` image directories and never maps `val` to test.
- [x] Add CLI flags `--shot {1,2,3,5,all}`, `--data-root`, `--work-root`, `--overwrite`, `--continue-on-error`; use the exact Kaggle roots as defaults.
- [ ] Run focused tests and `pytest -q`. Use T02's optional read-only source audit for local counts. Materialize the real dataset only on Kaggle; compare prepared 5-shot `197/235`, test `2655/3108`, and no train/test overlap there. If Kaggle data is unavailable, leave this runtime gate open rather than claiming preparation passed.

T04 local evidence (2026-10-04, Python 3.13 / pytest 9.1.1):

- Initial focused RED: **35 failed** because the preparation module and CLI
  did not exist. Initial GREEN: **35 passed**, full suite **73 passed, 1 skipped**.
- A fresh read-only review identified promotion/rollback recovery, corrupt
  category-mapping overwrite, and empty shared-test handling issues. Focused
  regression runs reproduced the defects before fixes. Final focused gate
  `py -3.13 -m pytest -q tests/test_prepare.py`: **42 passed**. Independent
  synthetic gate `py -3.13 -m pytest -q`: **80 passed, 1 skipped**.
- Shared-test reuse checks source JSON hashes, taxonomy, every copied image
  and generated label, and extra files. Explicit overwrite rebuilds affected
  targets; retained shots must have compatible taxonomy. Failed promotion
  rolls back replaced targets; if restoration itself fails, original backups
  remain in a reported `.prepare-stage-*` recovery directory.
- `data.yaml` uses JSON syntax (valid YAML), with absolute `train` and `test`
  paths and no `val` by default. `--val-train-placeholder` explicitly points
  `val` to train; T06 must verify whether the target parser needs that option
  and still disable training validation. No target Ultralytics version was
  installed or claimed compatible by T04.
- CLI examples after `pip install -e .`:
  `python scripts/prepare_dataset.py --shot 5` and
  `python scripts/prepare_dataset.py --shot all` use the specified Kaggle
  input/work roots. Per-shot failures under `--continue-on-error` remain
  visible in `results/audit.json`, and the CLI returns a nonzero exit code.
- No real dataset was materialized locally. T02's local image-layout failure
  remains unresolved; the final T04 item stays unchecked for the Kaggle
  prepared-count/overlap runtime gate. T05 has not been started.

## T05 — Stable run identity, manifest, and resume guard

**Depends on:** T01.  
**Files:** Create `camo_fs/runs.py`, `tests/test_runs.py`.  
**Produces:** immutable `RunConfig`, `fingerprint`, `run_path`, manifest serializer, `verify_resume`, and `upsert_summary`.  
**Acceptance:** Method, shot, seed, `run_kind` (`benchmark` or `smoke`), base-weight checksum, prepared-data checksum, epochs, image size, batch, full augmentation preset, and enhanced triplet settings contribute to identity. Same config yields same hash; changed config yields a different hash. Resume checks config/data and own checkpoint; overwrite never resumes. Failed training attempts can be recorded before evaluation exists, while smoke rows remain distinguishable from benchmark rows.

- [ ] Write `test_stable_fingerprint`, `test_training_config_changes_identity`, `test_methods_have_separate_paths`, `test_data_checksum_changes_identity`, `test_smoke_and_benchmark_have_distinct_identity`, `test_resume_rejects_method_or_weight_mismatch`, `test_resume_requires_own_last_checkpoint`, and `test_failed_status_upsert`.
- [ ] Run `pytest -q tests/test_runs.py`; confirm expected red.
- [ ] Implement canonical JSON hashing, path construction `runs/yolo11n-seg/<method>/shot_K/seed_N/<hash>`, explicit state transitions, prepared-data/source/weight SHA-256, actual package versions, UTC timestamp, warning fields, and one fixed-schema `upsert_summary` helper. Hash content rather than the filesystem location of a local checkpoint when possible.
- [ ] Run focused tests and `pytest -q`. Inspect sample manifests to ensure no absolute Windows path is baked into defaults.
- [ ] Test `--overwrite` semantics against a disposable run directory only; never delete broad roots or user-owned files.

## T06 — Baseline training and batch orchestration

**Depends on:** T04, T05.  
**Files:** Create `camo_fs/train.py`, `scripts/train_yolo.py`, `tests/test_train.py`.  
**Produces:** `train_one(config, paths, resume, overwrite) -> last_pt`; shared option builder used by both methods.  
**Acceptance:** Baseline delegates to native YOLO11n-Seg training, uses fixed epochs and `last.pt`, and never executes the custom feature hook or triplet loss. `--shot all` trains sequentially and records failed attempts.

- [ ] Write tests with a narrow injected fake YOLO adapter that records arguments. Assert `val=False`, `overlap_mask=False`, `mosaic=0`, `mixup=0`, `copy_paste=0`, fixed seed/deterministic request, equal train settings, `last.pt`, and no official test as `val`. Assert missing checkpoint and existing run fail before training; test resume/overwrite branches and a failed-row upsert.
- [ ] Run `pytest -q tests/test_train.py`; confirm red from missing orchestration.
- [ ] Implement native baseline dispatch and seed Python/NumPy/Torch/Ultralytics. Load the original checkpoint separately for every method/shot. Keep CLI parsing thin and `--continue-on-error` explicit; record failure status through `upsert_summary` immediately when a run fails.
- [ ] Run focused tests and `pytest -q`. On Kaggle, first confirm a one-epoch baseline run can save `last.pt` without using test for training validation; record installed library version and any automatic final-validation behavior.

## T07 — Valid training-image region at feature resolution

**Depends on:** T01.  
**Files:** Create `camo_fs/valid_region.py`, `tests/test_valid_region.py`.  
**Produces:** Version-independent `valid_letterbox_mask(source_hw, input_hw, ratio_pad=None) -> BoolTensor` and a pure feature-resolution mask reduction.  
**Acceptance:** From explicit source/input dimensions and padding geometry, valid pixels correspond to transformed image content, not letterbox padding. A feature-level cell is eligible for background sampling only when its **entire corresponding input-image spatial cell** is valid; a partially padded cell is invalid. T07 does not inspect or guess Ultralytics-internal metadata. The initial shared augmentation preset allows letterbox, horizontal flip, and color transforms; any future geometric augmentation must extend this logic and its tests before enabling it.

- [ ] Write tests for square, tall, wide, odd-padding, horizontal-flip, and feature-cell-at-boundary cases. Assert an 8-pixel cell with only 3 valid pixels and 5 padding pixels is invalid; all-valid reduction accepts only entirely valid cells. Reject nearest-neighbor and any-valid reductions for the valid-region mask.
- [ ] Run `pytest -q tests/test_valid_region.py`; confirm red.
- [ ] Implement pure geometry from explicit `source_hw`, `input_hw`, and `ratio_pad` inputs; test odd-padding and horizontal-flip cases without importing Ultralytics. Reduce input validity to actual feature-grid cells with an all-valid rule, conservatively excluding every partly padded cell. Do not infer target-version letterbox rounding or internal batch metadata here; T09 provides and verifies those inputs.
- [ ] Run focused tests and `pytest -q`. Leave the comparison against the installed Ultralytics loader's actual image/mask placement and metadata to T09's Kaggle integration gate.

## T08 — GT-mask triplet sampling and differentiable loss

**Depends on:** T07.  
**Files:** Create `camo_fs/triplet.py`, `tests/test_triplet.py`.  
**Produces:** `TripletResult(loss, sampled_triplets, skipped_instances, sampled_positions)` and `sample_and_loss(...)`.  
**Acceptance:** The sampler consumes **batch-transformed** per-instance masks and `batch_idx`, not original COCO polygons. Anchor and positive are distinct positions in one instance; negative is valid background outside the union of all objects in the same image. At most 16 triplets per valid instance by default; feature vectors are L2-normalized and use Euclidean margin `0.3`.

- [ ] Write synthetic feature/mask tests for same-instance positive, same-image background negative, bounded count, fixed-generator determinism, `test_negative_excludes_padding`, mask projection by nearest-neighbor, zero loss for satisfied triplets, skipped tiny/no-background instances, empty-batch finite zero, and gradient backpropagation into the input feature tensor.
- [ ] Run `pytest -q tests/test_triplet.py`; confirm red.
- [ ] Implement mask projection and indexed feature sampling with no detached feature tensors. Use PyTorch's triplet margin implementation. For tiny valid objects, only sample with replacement if at least two distinct foreground cells exist; otherwise skip and count.
- [ ] Run focused tests and `pytest -q`. Add non-finite diagnostic with shot/method/batch context; verify an empty sample set contributes differentiable finite zero without NaN.
- [ ] Add `test_horizontal_flip_keeps_mask_feature_alignment`: horizontally flip a synthetic image/object through the same training transform, then assert the current-batch GT mask, P3 spatial feature position, and valid-region mask align and sampled foreground/background locations remain correct. Also assert alignment after letterbox. Do not recreate the mask by independently resizing its original COCO polygon.

## T09 — Project-owned Ultralytics extension and compatibility test

**Depends on:** T06, T08.  
**Files:** Create `camo_fs/ultralytics_ext.py`, `tests/test_ultralytics_ext.py`, `tests/test_ultralytics_integration.py`.  
**Produces:** A target-version adapter that extracts/verifies actual letterbox geometry for T07, `P3Capture`, `FGSegmentationTrainer`, and a model loss override that adds weighted triplet loss to the native scalar without changing the verified native return schema.  
**Acceptance:** The adapter reads actual preprocessing geometry from the installed loader and validates it against source-image dimensions and batch placement; absent or contradictory metadata fails clearly. First `Segment` head input is captured for the current forward only; it is spatial P3/8 with expected batch/channel dimensions and gradients. Native loss is unchanged and included. The enhanced `loss()` preserves the exact return structure, loss-item count/order/type, and tuple/dict semantics expected by the installed trainer; auxiliary metrics are logged separately. Hook state is cleared before the next forward and after the step. At least one prepared multi-polygon label is accepted by the target segmentation dataloader as one instance without corrupt-label warning or silent drop. Incompatible installed versions fail before a long training run.

- [ ] Inspect `ultralytics.__version__`, actual loader/batch geometry metadata and letterbox rounding, `SegmentationTrainer.get_model`, `SegmentationModel.loss`, the `Segment` head input structure, and native loss return contract **in the actual target environment**. Record concrete signatures, metadata, and observed shapes in a short integration note in the design doc or README. Do not pick a numeric layer index from upstream `main` alone.
- [ ] Write fake-module tests for one capture, missing capture, wrong stride/shape, stale feature, multiple unexpected capture, and hook reference release. Include `test_capture_rejects_stale_or_multiple_forward`, `test_loss_preserves_native_return_contract`, and `test_enhanced_p3_gradient_and_baseline_bypasses_triplet`. The latter asserts a nonzero gradient on the captured P3 tensor and verifies baseline neither instantiates nor calls the triplet sampler.
- [ ] Run `pytest -q tests/test_ultralytics_ext.py`; confirm red.
- [ ] Implement the target-version geometry adapter that extracts/verifies actual loader metadata and passes explicit geometry to T07; fail clearly if the required geometry cannot be established. Implement semantic `Segment` discovery, scoped forward pre-hook, guarded capture lifecycle, and the smallest `SegmentationTrainer.get_model` / `SegmentationModel.loss` extension. Preserve the **observed target-version native return structure exactly**; add weighted triplet loss only to the appropriate native loss scalar and log raw/weighted auxiliary values through a separate side channel or callback. Never append another tuple element, dictionary key, or loss item merely for logging.
- [ ] Run focused and full synthetic tests. Then run `pytest -q -m integration` on Kaggle with attached `yolo11n-seg.pt`; assert the metadata adapter reproduces actual loader image/mask/valid-region placement, checkpoint loads, P3/8 capture matches actual head input, the batch supplies transformed per-instance masks aligned with `batch_idx`, a batch produces finite native+auxiliary loss with the exact native return contract, and auxiliary gradients reach both captured P3 and a neck/backbone parameter. Load at least one **prepared** merged multi-polygon label through this target-version segmentation dataloader and assert it remains exactly one source instance, all normalized coordinates are accepted, and no corrupt-label warning or silent drop occurs. Stop on incompatibility; do not silently train baseline.

## T10 — Enhanced one-epoch smoke gate and version pin

**Depends on:** T09.  
**Files:** Modify `camo_fs/train.py`, `requirements.txt`, `tests/test_train.py`; add smoke-run instructions/results section to `README.md` when evidence exists.  
**Produces:** A recorded enhanced run gate before full experiments.  
**Acceptance:** One official shot (prefer 1-shot) completes a short enhanced run on Kaggle single GPU. Some batches have nonzero raw triplet loss; gradients are nonzero; every logged loss is finite. Enhanced `last.pt` saves, reloads through the same inference-loading path used by evaluation/visualization, and yields at least one paired predicted box and segmentation mask on prepared **training** imagery. An empty result alone does not satisfy this gate. No official test image or metric is used for this smoke check or to tune the loss.

- [ ] Write tests showing `--method fgbg-triplet` routes through the extension while `baseline` bypasses it, enhanced `device` rejects multi-GPU, and both modes receive identical non-method training options.
- [ ] Run the focused test red; implement the dispatch; rerun focused and full suites.
- [ ] Prepare the official 1-shot split on Kaggle and run `--method fgbg-triplet --shot 1 --epochs 1 --device 0 --seed 2024` with `run_kind=smoke` and the base checkpoint. Save console log, manifest, triplet counts, finite-loss proof, and `last.pt` under `/kaggle/working`. Reload that checkpoint through the evaluation/visualization inference-loading path and predict on prepared **train** images until at least one prediction contains both a box and its instance mask. If the first image has no detections, try other prepared training images or a lower smoke-only confidence threshold; record that threshold and leave the gate open if no paired output appears. Do not load an official test image for this proof.
- [ ] If no batch yields valid/nonzero triplets, inspect sampler visualization and shapes; fix with a failing synthetic regression test before retrying. Do not run the eight full experiments yet.
- [ ] Only after the gate succeeds, set `ultralytics==<actual-tested-version>` in `requirements.txt`, record Torch/CUDA versions, rerun the integration test under that pin, and update the README with the verified version. If Kaggle is unavailable, leave this task open and the dependency unpinned.

## T11 — Official-test evaluation, six metrics, and summary upsert

**Depends on:** T05, T06; runtime confirmation after T10 for enhanced.  
**Files:** Create `camo_fs/evaluate.py`, `scripts/evaluate_yolo.py`, `tests/test_evaluate.py`.  
**Produces:** `evaluate_one(run_dir, data_yaml, results_dir) -> dict` and `results/summary.csv`.  
**Acceptance:** Loads only the run's `last.pt`, explicitly evaluates `split="test"`, extracts Box AP/AP50/AP75 and Mask AP/AP50/AP75, and upserts by complete run identity. Failed runs have `status=failed`, error text, and empty metrics. Repeating evaluation updates a row without duplication. Batch selection includes `--method` and `--shot {1,2,3,5,all}`; if multiple configuration hashes match, process every completed benchmark run as distinct. Smoke runs are excluded by default.

- [ ] Write fake-validator tests for exact metric extraction, missing `last.pt`, accidental `val` split rejection, failed row, idempotent rerun, batch discovery across shot `all`, default smoke exclusion, and `test_upsert_preserves_distinct_config_hashes` (same method/shot/seed but different config hashes remain separate).
- [ ] Run `pytest -q tests/test_evaluate.py`; confirm red.
- [ ] Implement test-only evaluation and atomic CSV upsert via the fixed-schema helper from T05. Validate source YAML `test` points to shared official prepared test path, and never choose weights using official test results.
- [ ] Run focused and full tests. Keep evaluation plumbing checks synthetic until the planned final official-test evaluation; do not use the official test set during smoke training or to decide training settings.

## T12 — Deterministic prediction and sampler debug visuals

**Depends on:** T08, T11.  
**Files:** Create `camo_fs/visualize.py`, `scripts/visualize_predictions.py`, `tests/test_visualize.py`.  
**Produces:** `visualize_one(...)` plus optional training-only sampler debug render.  
**Acceptance:** For the same seed and official test image list, select the same 20 images by default; draw predicted boxes, masks, class names, and confidence; separate directories by method/shot/seed/config hash; never overwrite another run's images. Support one run or method/shot `all` benchmark discovery, excluding smoke runs by default.

- [ ] Write tests for deterministic image selection, `num_images` cap, `conf=0.25` forwarding, empty predictions, filename collision avoidance, method/shot `all` run discovery, smoke exclusion, and sampled-point overlay being constrained by GT/valid masks.
- [ ] Run `pytest -q tests/test_visualize.py`; confirm red.
- [ ] Implement rendering using prediction outputs and the saved taxonomy; keep sampler debug rendering optional and outside every-batch hot path.
- [ ] Run focused and full tests; visually inspect a small known fixture and a Kaggle smoke sample to confirm geometry/mask alignment.

## T13 — README, Kaggle entrypoint, full comparison, and release audit

**Depends on:** T04–T12; full-run claim requires T10.  
**Files:** Create/modify `README.md`; optionally create `notebooks/kaggle_entrypoint.ipynb`; update `docs/camo-fs-yolo-design-decisions.md` only when verified implementation details require it.  
**Produces:** Copyable end-to-end commands and a truthfully labeled results table.  
**Acceptance:** README covers official data paths, package install, Internet-on/off checkpoint setup, prepare/train/evaluate/visualize commands for one shot and all shots, smoke gate, resuming, overwriting, manifests, summary schema, multi-polygon topology loss, and exact tested Ultralytics version after proof. It never claims the enhanced method improves AP without measured results.

- [ ] Write a documentation check/test that executes `--help` for all four scripts and asserts documented flags/defaults exist, or review help output against README command blocks if CLI execution needs target dependencies.
- [ ] Run the doc/CLI check red where commands are missing; write README and optional notebook as thin orchestration; rerun check and `pytest -q`.
- [ ] On Kaggle, run 1/2/3/5 for `baseline`, then 1/2/3/5 for `fgbg-triplet`, each independently from original weights. Use the same preset and separate run identity. Stop on failure by default; use `--continue-on-error` only when explicitly desired.
- [ ] Evaluate each completed `last.pt` on official test and create deterministic prediction renders. Verify eight distinct summary identities or explicit failed statuses; compare manifest data/checkpoint/augmentation fields for every paired shot.
- [ ] Run the final synthetic suite, Kaggle integration suite, audit-count checks, smoke gate review, and artifact inspection. Report what was actually run and what remains unverified; do not fill README results with placeholders disguised as measurements.

## Completion checklist

- [ ] All FR-01–FR-21 and QG-01–QG-05 map to an implemented task and verified evidence.
- [ ] `pytest -q` passes in a supported Python/Torch environment; no CAMO-FS data is required for the synthetic suite.
- [ ] Official split counts, including all 235 5-shot instances, have been checked against actual prepared artifacts.
- [ ] Target-version integration and enhanced smoke gates pass on Kaggle before the full eight-run comparison.
- [ ] Both methods' final results use only `last.pt` and the official test split; no official test is used as `val`.
- [ ] README reports measured results or clearly says results have not yet been run.
