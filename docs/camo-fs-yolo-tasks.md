# CAMO-FS YOLO Implementation Tasks

**Plan:** [camo-fs-yolo-plans.md](camo-fs-yolo-plans.md)  
**Spec:** [camo-fs-yolo-spec.md](camo-fs-yolo-spec.md)  
**Status:** T01–T11 are accepted and frozen; their task checklists reflect the user's acceptance. T12 visualization source is implemented for review; its Kaggle visual sample remains unverified. T13 remains.

**Checklist convention:** Checked T01–T11 items record the accepted foundation,
not a fresh rerun of every historical command. Evidence paragraphs below retain
their original observations: statements such as "gate remains open", "not yet
started" or "provisional" describe that earlier stage and are superseded by
the current accepted/frozen state. Do not reopen T01–T11 from those paragraphs.
Conditional actions that were unnecessary are explicitly marked not applicable.

The user confirmed T10's official Kaggle smoke, trained `last.pt` save/reload,
nonzero triplet signal and P3/backbone auxiliary gradients, and paired box/mask
inference on TRAIN imagery. Verified target: Ultralytics **8.3.228**, Torch
**2.11.0+cu128**, CUDA **12.8**, Tesla **T4**. The exact Ultralytics pin is
committed; the user reported post-pin integration **35 passed, 1 skipped,
481 deselected**. The paired prediction used smoke-only confidence **1e-4**;
this is neither a benchmark threshold nor a model-quality claim. These are
user-supplied Kaggle results, not local reruns in the T11 implementation.

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

**Current state: ACCEPTED / FROZEN by the user.**

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

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T01.  
**Files:** Create `camo_fs/annotations.py`, `tests/test_annotations.py`, `tests/test_real_data_audit.py` (marked `real_data`, skipped without local CAMO-FS).  
**Produces:** `Taxonomy.from_test_json(path)`, `AuditReport`, `audit_shot(shot, paths, taxonomy)`; report holds deduplicated records, source paths, errors, warnings, and counts without writing data.  
**Acceptance:** Global category mapping is sorted and contiguous; every shot requires exactly the selected official file pattern for all 47 categories; distinct objects with reused IDs survive; exact duplicate content is deduplicated from merged records, reported as an integrity error, and prevents materialization; conflicting geometry/metadata and train/test overlap fail.

- [x] Write synthetic COCO fixture tests for unsorted/noncontiguous category IDs, taxonomy mismatch, missing or extra shot files, duplicated image metadata, exact duplicate content, conflicting same-image geometry, and train/test overlap. Include `test_reused_id_in_different_image_is_retained` with two objects sharing ID but different image/class and expected count `2`.
- [x] Run `pytest -q tests/test_annotations.py`; confirm expected missing-behavior failures.
- [x] Implement taxonomy and audit in one pass over selected files. Key images by `image_id`; require their filename/dimensions to agree across files and flag a filename mapped to different IDs. Use full annotation content/geometry keys; record reused IDs as warnings, never as global dedup keys. Validate file existence and declared versus actual image dimensions.
- [x] Run focused tests and then `pytest -q`; verify no test depends on the real CAMO-FS data.
- [x] Read-only audit the local official JSON if present through `pytest -q -m real_data`; assert observed 1/2/3/5-shot counts. Do not rewrite any `data/` file. Keep this test separate from the synthetic suite's mandatory gate. Later T09 review correction passed this gate using a local path view; see evidence below.

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

**Current state: ACCEPTED / FROZEN by the user.**

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

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T02, T03.  
**Files:** Create `camo_fs/prepare.py`, `scripts/prepare_dataset.py`, `tests/test_prepare.py`; extend `camo_fs/annotations.py` with the shared-test audit.

**Consumes:** `DatasetPaths`, `AuditReport`, `annotation_to_yolo`.  
**Produces:** shared `test/images|labels`, `shot_K/train/images|labels`, `shot_K/data.yaml`, `category_mapping.json`, per-shot manifests, `/kaggle/working/results/audit.json`.  
**Acceptance:** Preparation never writes into input; exact shot `all` invokes `1,2,3,5` and copies test once; output collision requires `--overwrite`; no stale label survives overwrite. Without `--continue-on-error`, preflight-audit the shared test and **all selected shots** before any prepared dataset write; if any audit fails, leave `test/`, `category_mapping.json`, and every selected `shot_K/` uncreated or unchanged (audit/error reports may be written). With `--continue-on-error`, audit/materialize shots independently and record every failure, but a failed shared-test audit blocks all materialization. When overwrite changes shared-test provenance used by existing shots, preflight every selected shot and replace all existing dependents in one transaction; any existing dependent's audit or staging failure must preserve the previous prepared artifacts.

- [x] Write fixture tests for `--shot 5`, `--shot all`, common test reuse, image copy once per split, expected label count, audit-before-write, fail-fast versus `--continue-on-error`, overwrite cleanup, and `test_val_never_points_to_test` (both YAML-without-val and technical train-placeholder modes). Include `test_all_fail_fast_audits_every_shot_before_materialization`: shot 1 and 2 pass, shot 3 fails, and no shared test or shot artifact is created. Include the complementary continue-on-error case: successful shots materialize once, failed shots do not, and all outcomes are reported. A failed shared-test audit creates no prepared artifact in either mode.
- [x] Run `pytest -q tests/test_prepare.py`; confirm red for missing preparation behavior.
- [x] Implement `prepare_selected(...)` with target-scoped temporary staging and explicit target checks. In fail-fast mode, finish the complete shared-test plus selected-shot audit preflight before creating/staging any prepared artifact; materialize only after every audit passes. In continue-on-error mode, audit and materialize each shot separately after the shared test passes, except when changed shared-test provenance requires replacing all existing dependents together. Write audit/error reports on failure; only promote fully valid staged splits. YAML has absolute `train`/`test` image directories and never maps `val` to test.
- [x] Add CLI flags `--shot {1,2,3,5,all}`, `--data-root`, `--work-root`, `--overwrite`, `--continue-on-error`; use the exact Kaggle roots as defaults.
- [x] Run focused tests and `pytest -q`. Use T02's optional read-only source audit for local counts. Materialize the real dataset only on Kaggle; compare prepared 5-shot `197/235`, test `2655/3108`, and no train/test overlap there. If Kaggle data is unavailable, leave this runtime gate open rather than claiming preparation passed.

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
- Follow-up review correction (2026-10-04): partial overwrite now checks the
  canonical test JSON SHA-256 recorded in every retained shot's manifest,
  in addition to taxonomy. Changed or unverifiable shared-test provenance
  blocks all prepared writes in either execution mode and asks the caller to
  rebuild all retained shots together; retained manifests are never silently
  updated. Metadata-only and valid polygon changes both have regressions, and
  rebuilding all retained shots updates every test provenance entry correctly.
  Shared-test reuse also validates deterministic counts and multi-polygon
  metadata; the audit comment now describes atomic replacement of the current
  report, including failures. Focused regression RED: **12 failed, 2 passed**.
  Latest focused GREEN: **55 passed**; full synthetic: **93 passed, 1 skipped**.
  The Kaggle preparation gate remains open.
- Second follow-up correction (2026-10-04): changed shared-test provenance
  now requires all existing selected shots to pass preflight and stage before
  any replacement, including with `--continue-on-error`. An existing dependent's
  audit or copy failure leaves the previous shared test, mapping and all shots
  unchanged. Absent shots that fail audit can still be reported and skipped
  while existing dependents rebuild together. Regression fixtures cover both
  metadata-only and polygon changes, copy failure, and a successful rebuild
  with a failing new shot. Combined T04/T05 regression RED: **23 failed,
  6 passed**; focused GREEN: **126 passed**; full synthetic gate: **164 passed,
  1 skipped**. Real preparation remains unverified on Kaggle.

## T05 — Stable run identity, manifest, and resume guard

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T01.  
**Files:** Create `camo_fs/runs.py`, `tests/test_runs.py`.  
**Produces:** immutable `RunConfig`, `fingerprint`, `run_path`, manifest serializer, `verify_resume`, and `upsert_summary`.  
**Acceptance:** Method, shot, seed, `run_kind` (`benchmark` or `smoke`), base-weight checksum, prepared-data checksum, epochs, image size, batch, full augmentation preset, and enhanced triplet settings contribute to identity. Same config yields same hash; changed config yields a different hash. Resume checks config/data and own checkpoint; overwrite never resumes. Failed training attempts can be recorded before evaluation exists, while smoke rows remain distinguishable from benchmark rows.

- [x] Write `test_stable_fingerprint`, `test_training_config_changes_identity`, `test_methods_have_separate_paths`, `test_data_checksum_changes_identity`, `test_smoke_and_benchmark_have_distinct_identity`, `test_resume_rejects_method_or_weight_mismatch`, `test_resume_requires_own_last_checkpoint`, and `test_failed_status_upsert`.
- [x] Run `pytest -q tests/test_runs.py`; confirm expected red.
- [x] Implement canonical JSON hashing, path construction `runs/yolo11n-seg/<method>/shot_K/seed_N/<hash>`, explicit state transitions, prepared-data/source/weight SHA-256, actual package versions, UTC timestamp, warning fields, and one fixed-schema `upsert_summary` helper. Hash content rather than the filesystem location of a local checkpoint when possible.
- [x] Run focused tests and `pytest -q`. Inspect sample manifests to ensure no absolute Windows path is baked into defaults.
- [x] Test `--overwrite` semantics against a disposable run directory only; never delete broad roots or user-owned files.

T05 local evidence (2026-10-04, Python 3.13 / pytest 9.1.1):

- Focused initial RED: **37 failed**, because `camo_fs.runs` did not exist.
  Initial GREEN: **37 passed**; full suite **130 passed, 1 skipped**.
- Fresh read-only review found that preparation provenance needed validation
  in new and existing run manifests. Regression RED: **9 failed** for missing
  fields, wrong shot, invalid counts/paths/checksums and corrupted saved
  provenance. Final `py -3.13 -m pytest -q tests/test_runs.py`: **46 passed**;
  independent `py -3.13 -m pytest -q`: **139 passed, 1 skipped**.
- `RunConfig` binds weights/data SHA-256 to immutable effective settings;
  `fingerprint(config, hashes)` rejects contradictory digests. `run_path`
  uses the same canonical identity. Weight paths are recorded for provenance
  but identical weight content at another location keeps the fingerprint.
  Smoke/benchmark, method, shot, seed, augmentation, extra scalar training
  options and applicable triplet parameters remain distinct identities.
- `prepared_data_sha256` hashes actual selected train/shared test file bytes,
  mapping, semantic JSON-compatible YAML, and recorded source JSON hashes.
  Generation timestamps and relocatable roots do not alter the fingerprint.
  This helper fingerprints content; it does not replace T04's integrity audit.
- `build_manifest` requires valid preparation provenance. `initialize_run`
  refuses existing targets by default; explicit resume verifies identity,
  interrupted state, and the run's own `weights/last.pt`. Explicit overwrite
  replaces only a verified managed target with a fresh initialized manifest,
  removes old checkpoints, preserves other runs, and retains recovery backup
  if restoring an old target fails. `transition_run` writes state changes
  atomically; `summary_row`/`upsert_summary` use one fixed CSV schema and complete
  identity key. Failed rows always have blank AP metrics.
- All filesystem fixtures were disposable synthetic data. No real dataset
  was prepared or trained. T06 must resolve actual hashes, verify checkpoint
  authenticity/compatibility, forward and record effective target-library
  training options, and require `last.pt` before declaring training complete.
  Augmentation/epoch/batch defaults here are project presets, not a claim
  about verified Ultralytics defaults.
- Deferred review minors: AP-value range/type validation belongs to final
  evaluation plumbing in T11; completion checkpoint enforcement remains the
  T06 caller's responsibility; dedicated overwrite promotion/rollback
  failure-injection tests are not yet present (successful scoped overwrite
  and atomic summary-write failure are covered). Summary writes assume the
  specified sequential single-process orchestration. T06 has not started.
- Follow-up hardening (2026-10-04): preparation provenance now requires the
  supported schema, matching shot, positive coherent instance/image/label
  counts, canonical contiguous taxonomy, output paths, source checksums,
  an aware UTC timestamp and dependency version entries. Arguments and file
  checksums remain snapshots; actual bytes are covered by the resolved data
  fingerprint and T04 integrity checks. A genuine synthetic T04 manifest is
  accepted by T05. Augmentation probability/fraction settings reject values
  outside `[0,1]`. Combined regression and full-suite evidence is recorded
  above; runtime training and the deferred review items remain open.

## T06 — Baseline training and batch orchestration

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T04, T05.  
**Files:** Create `camo_fs/train.py`, `scripts/train_yolo.py`, `tests/test_train.py`.  
**Produces:** `train_one(config, paths, resume, overwrite) -> last_pt`; shared option builder used by both methods.  
**Acceptance:** Baseline delegates to native YOLO11n-Seg training, uses fixed epochs and `last.pt`, and never executes the custom feature hook or triplet loss. `--shot all` trains sequentially and records failed attempts.

- [x] Write tests with a narrow injected fake YOLO adapter that records arguments. Assert `val=False`, `overlap_mask=False`, `mosaic=0`, `mixup=0`, `copy_paste=0`, fixed seed/deterministic request, equal train settings, `last.pt`, and no official test as `val`. Assert missing checkpoint and existing run fail before training; test resume/overwrite branches and a failed-row upsert.
- [x] Run `pytest -q tests/test_train.py`; confirm red from missing orchestration.
- [x] Implement native baseline dispatch and seed Python/NumPy/Torch/Ultralytics. Load the original checkpoint separately for every method/shot. Keep CLI parsing thin and `--continue-on-error` explicit; record failure status through `upsert_summary` immediately when a run fails.
- [x] Run focused tests and `pytest -q`. On Kaggle, first confirm a one-epoch baseline run can save `last.pt` without using test for training validation; record installed library version and any automatic final-validation behavior.

T06 local evidence (2026-10-04, Python 3.13 / pytest 9.1.1):

- Tracer RED: **1 failed**, because `camo_fs.train` was absent. Each subsequent
  behavioral slice was run RED before implementation: failure states, native
  protocol guards, read-only integrity, caches, sequential batch behavior,
  runtime compatibility/seeding, CLI, interrupted-running resume and stale
  checkpoint rejection. Final focused `py -3.13 -m pytest -q tests/test_train.py`:
  **49 passed**; full synthetic `py -3.13 -m pytest -q`: **213 passed, 1 skipped**.
  Optional `real_data` remained unset. No real dataset or model was trained.
- `resolve_config` binds installed supported native training/segmentation
  defaults and explicit overrides before fingerprinting. `training_options`
  builds the common protocol for either method. The baseline calls native
  `YOLO.train`; enhanced dispatch explicitly fails until T09/T10, with no
  auxiliary imports or feature capture. The CLI keeps all business logic in
  `train_selected`, processes shots sequentially, and returns nonzero on failure.
- `verify_prepared` re-audits official sources read-only and verifies generated
  image/label bytes, source hashes, counts, mapping and YAML taxonomy. Native
  sibling `labels.cache` files do not change data identity; shared-test reuse
  ignores only its recognized cache and continues rejecting extra data files.
- Native callbacks verify requested settings, train/validation data and exact
  run/checkpoint paths before epochs and again on return. Manifests record
  requested and actual trainer args, seed warnings, completed epochs and final
  checkpoint SHA-256. Completion requires all requested epochs and a nonempty
  own `last.pt`; resume must produce a new checkpoint, uses only its own
  interrupted optimizer checkpoint, and rejects changed effective defaults.
  Both `running` and `failed` interruption states are supported. Overwrite
  starts fresh from the same bound base checkpoint through T05's scoped guard.
- Fresh review findings were reproduced and fixed: defaults missing from
  identity, shared-test cache rejection, and already-failed resume masking a
  new load error (**4 regression RED failures**). Native optimizer/CPU worker
  normalization also had **3 RED failures** before its fix. Focused and full
  final GREEN results above cover the fixes; review found no other blocker.
- Ruling: the project resolves native `optimizer=auto` to explicit **SGD**
  before hashing, preserving recorded native LR/warmup values. Explicit `auto`
  overrides are rejected because native heuristics rewrite those settings.
  CPU workers resolve to **0** before hashing. This avoids false compatibility
  failures and gives paired methods the same explicit preset; changing the
  optimizer or resolved hyperparameters creates another run identity.
- Ruling: failures before a weight/data identity can be resolved are recorded
  with a null `config_hash` in the current invocation's atomic
  `results/training_attempts.json`; no checksum is invented. Resolved failures
  immediately upsert the fixed-schema CSV with blank AP fields. Rejected
  existing run attempts preserve that run's manifest and prior summary row.
- Base loading checks the installed packaged YOLO11n segmentation architecture,
  Segment head and COCO class names; it binds actual local bytes by SHA-256.
  This is compatibility checking, not cryptographic upstream-release attestation.
  Use the original official pretrained checkpoint. Install the package and
  target runtime on Kaggle (`pip install -e . ultralytics`), attach/download
  `yolo11n-seg.pt` locally, then run e.g.
  `python scripts/train_yolo.py --method baseline --shot 1 --epochs 1 --run-kind smoke --weights /kaggle/input/yolo11-seg-weights/yolo11n-seg.pt`.
  If the installed parser requires `val`, prepare with
  `--val-train-placeholder` so it refers to train, never official test.
- Kaggle baseline training, real checkpoint loading, loader/parser compatibility
  and library-triggered final validation remain **unverified**; the final T06
  item stays unchecked. Training validation/final native validation may only
  read the train placeholder; no training AP is used as official test results.
  No Ultralytics version was pinned. T07 has not started.
- Follow-up hardening (2026-10-04): one shared reader now enforces T04's
  exact prepared YAML keys (`train`, `test`, `names`, optional `val`) in both
  fingerprinting and read-only auditing. Extra semantic fields such as
  `channels`, `nc` and `path` are rejected before run initialization or model
  loading. Native trainer guards require RGB and canonical parsed names/class
  count against audited taxonomy; class count is derived, never hardcoded to
  47. Valid canonical list/dictionary names and optional `val=train` remain
  accepted. TDD evidence: **9 YAML rejection RED failures**, then **6 native
  metadata RED failures** before their fixes; all 19 added cases are GREEN.
  Related `tests/test_train.py tests/test_prepare.py tests/test_runs.py`:
  **194 passed**; full `py -3.13 -m pytest -q --tb=short` with optional
  real-data root unset: **232 passed, 1 skipped**. Fresh scoped code review
  found no Critical/Important issues. Real Kaggle gates remain open.

## T07 — Valid training-image region at feature resolution

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T01.  
**Files:** Create `camo_fs/valid_region.py`, `tests/test_valid_region.py`.  
**Produces:** Version-independent `valid_letterbox_mask(source_hw, input_hw, ratio_pad=None) -> BoolTensor` and a pure feature-resolution mask reduction.  
**Acceptance:** From explicit source/input dimensions and padding geometry, valid pixels correspond to transformed image content, not letterbox padding. A feature-level cell is eligible for background sampling only when its **entire corresponding input-image spatial cell** is valid; a partially padded cell is invalid. T07 does not inspect or guess Ultralytics-internal metadata. The initial shared augmentation preset allows letterbox, horizontal flip, and color transforms; any future geometric augmentation must extend this logic and its tests before enabling it.

- [x] Write tests for square, tall, wide, odd-padding, horizontal-flip, and feature-cell-at-boundary cases. Assert an 8-pixel cell with only 3 valid pixels and 5 padding pixels is invalid; all-valid reduction accepts only entirely valid cells. Reject nearest-neighbor and any-valid reductions for the valid-region mask.
- [x] Run `pytest -q tests/test_valid_region.py`; confirm red.
- [x] Implement pure geometry from explicit `source_hw`, `input_hw`, and `ratio_pad` inputs; test odd-padding and horizontal-flip cases without importing Ultralytics. Reduce input validity to actual feature-grid cells with an all-valid rule, conservatively excluding every partly padded cell. Do not infer target-version letterbox rounding or internal batch metadata here; T09 provides and verifies those inputs.
- [x] Run focused tests and `pytest -q`. Leave the comparison against the installed Ultralytics loader's actual image/mask placement and metadata to T09's Kaggle integration gate.

T07 local evidence (2026-10-04, Python 3.13 / pytest 9.1.1 / Torch 2.14.1+cpu):

- Baseline before changes: **232 passed, 1 skipped**. Vertical TDD slices:
  initial RED **1 failed** for the missing module, GREEN **1 passed**;
  centered-fit RED **6 failed, 1 passed**, GREEN **7 passed**;
  explicit-geometry RED **3 failed, 7 passed**, GREEN **10 passed**;
  all-valid reduction RED **1 failed, 10 passed**, GREEN **11 passed**;
  nondivisible-grid RED **4 failed, 11 passed**, GREEN **15 passed**.
  Input validation also ran RED before implementation: dimension cases
  **14 failed, 18 passed**, explicit placement **22 failed, 32 passed**,
  reduction inputs **12 failed, 54 passed**. Corresponding GREEN runs were
  **32**, **54**, and **66 passed**.
- Final focused command
  `.superpowers/t07-venv/Scripts/python.exe -m pytest -q tests/test_valid_region.py --tb=short`:
  **73 passed**. Full command with `CAMO_FS_DATA_ROOT` unset,
  `.superpowers/t07-venv/Scripts/python.exe -m pytest -q --tb=short`:
  **305 passed, 1 skipped**. PyTorch CPU was installed in this ignored local
  test environment; `requirements.txt` and the package's `test` extra now
  declare unpinned `torch`. No Ultralytics version was pinned or installed.
- Public geometry contract: `ratio_pad=((scale_x, scale_y), (left, top))`
  contains actual post-rounding resize ratios and integer leading offsets,
  rather than half the total padding. Scales must recover integer resized
  dimensions within `1e-6` input pixels; invalid, ambiguous, cropped, or
  out-of-bounds placement fails explicitly. Without `ratio_pad`, the helper
  uses a documented project-owned centered aspect fit (nearest integer,
  ties to even, minimum one pixel, odd extra padding on bottom/right).
  This default does **not** certify an Ultralytics preprocessing policy.
- `reduce_valid_mask(valid, feature_hw)` returns bool validity for actual
  feature dimensions, preserving leading batch axes and device without
  mutating input. Integer floor/ceil footprints conservatively include every
  intersecting pixel for nondivisible grids. Tests reject a cell containing
  three valid pixels and five padding pixels, the equivalent stride-8 cell,
  and a cell with just one invalid corner pixel. Odd padding is flipped with
  image content before reduction; batches remain independent.
- T09 must adapt and compare these explicit inputs against the installed
  loader's actual transformed images/masks on Kaggle. That integration gate
  remains **not verified**. No source dataset was modified or materialized,
  no real training was run, and T08 was not started.
- Independent read-only review checked the geometry/reduction against a
  direct-slicing oracle for all input sizes `1..8` in both axes, every smaller
  feature grid, and multiple leading dimensions. It found one Important
  issue: implicit mask allocation inherited Torch's default device despite
  the documented CPU contract. Regression
  `test_letterbox_output_stays_on_cpu_under_a_non_cpu_device_context` ran
  RED (**1 failed, 72 deselected**) under a `meta` device context before
  explicitly allocating on CPU. The final focused/full commands above
  include the fix; no Critical or Minor findings were reported. Actual
  Ultralytics geometry/CUDA integration and runtime dependency pinning remain
  deferred to the existing T09/T10 gates.

T07 review follow-up (2026-10-04): the supplied review found no
Critical/Important issue and confirmed the synthetic gate. Docstrings now
explicitly distinguish project-normalized `ratio_pad` placement from raw
Ultralytics metadata and describe CPU output, explicit sampler-device transfer,
and the integral table's input-sized memory cost. The public API and geometry
algorithm remain unchanged; optimize only after measuring runtime overhead.
The T08 sampler reduces validity on its supplied device, then explicitly moves
the reduced mask to the feature device. T07 focused tests remain **73 passed**.

## T08 — GT-mask triplet sampling and differentiable loss

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T07.  
**Files:** Create `camo_fs/triplet.py`, `tests/test_triplet.py`.  
**Produces:** `TripletResult(loss, sampled_triplets, skipped_instances, sampled_positions)` and `sample_and_loss(...)`.  
**Acceptance:** The sampler consumes **batch-transformed** per-instance masks and `batch_idx`, not original COCO polygons. Anchor and positive are distinct positions in one instance; negative is valid background outside the union of all objects in the same image. At most 16 triplets per valid instance by default; feature vectors are L2-normalized and use Euclidean margin `0.3`.

- [x] Write synthetic feature/mask tests for same-instance positive, same-image background negative, bounded count, fixed-generator determinism, `test_negative_excludes_padding`, mask projection by nearest-neighbor, zero loss for satisfied triplets, skipped tiny/no-background instances, empty-batch finite zero, and gradient backpropagation into the input feature tensor.
- [x] Run `pytest -q tests/test_triplet.py`; confirm red.
- [x] Implement mask projection and indexed feature sampling with no detached feature tensors. Use PyTorch's triplet margin implementation. For tiny valid objects, only sample with replacement if at least two distinct foreground cells exist; otherwise skip and count.
- [x] Run focused tests and `pytest -q`. Add non-finite diagnostic with shot/method/batch context; verify an empty sample set contributes differentiable finite zero without NaN.
- [x] Add `test_horizontal_flip_keeps_mask_feature_alignment`: horizontally flip a synthetic image/object through the same training transform, then assert the current-batch GT mask, P3 spatial feature position, and valid-region mask align and sampled foreground/background locations remain correct. Also assert alignment after letterbox. Do not recreate the mask by independently resizing its original COCO polygon.

T08 local evidence (2026-10-04, Python 3.13.14 / pytest 9.1.1 / Torch 2.14.1+cpu):

- Baseline: **305 passed, 1 skipped**. Vertical TDD slices ran RED before
  production changes: missing sampler **1 failed**; same-image union
  **1 failed, 1 passed**; mask/validity projection **2 failed, 2 passed**;
  skipped/empty instances **6 failed, 4 passed**; tiny-object BG exclusion
  **1 failed, 10 passed**; normalized Euclidean loss **3 failed, 12 passed**;
  low-precision arithmetic **2 failed, 16 passed**; non-finite context
  **7 failed, 18 passed**; validation **27 failed, 29 passed**; GT upsampling
  **1 failed, 57 passed**; empty-loss precision **2 failed, 60 passed**;
  large-finite-vector normalization **1 failed, 62 passed**. Each slice then
  ran GREEN before the next behavior was implemented.
- Final focused command
  `.superpowers/t07-venv/Scripts/python.exe -m pytest -q tests/test_valid_region.py tests/test_triplet.py --tb=short`:
  **140 passed** (**73 T07 + 67 T08**). Full command with
  `CAMO_FS_DATA_ROOT` unset,
  `.superpowers/t07-venv/Scripts/python.exe -m pytest -q --tb=short`:
  **372 passed, 1 skipped**. No dataset or Ultralytics runtime is required.
- `sample_and_loss(features, masks, batch_idx, valid, count, margin, generator,
  *, context=None)` consumes current transformed binary per-instance masks
  and integer image indices. Foreground uses nearest-neighbor projection;
  input validity uses T07 all-valid reduction. Background excludes any GT
  union pixel intersecting the original source footprint of a feature cell,
  using adaptive max occupancy with floor/ceil bounds. This also excludes
  tiny objects lost by nearest foreground projection, fractional upsampling,
  and mixed up/down axes. Background candidates are cached per image.
- Every usable instance contributes exactly the requested bounded count,
  with replacement across draws but distinct anchor/positive positions.
  CLI/config defaults remain 16 triplets and margin 0.3. All random draws use
  the supplied generator; unrelated global RNG changes do not change results.
  Inputs are not mutated. Masks/indices explicitly move to the feature device,
  and a CPU generator is supported independently of that feature device.
  Synthetic non-CPU default-device tests verify explicit allocation; actual
  CUDA execution is **not verified**.
- `TripletResult` contains a raw unweighted mean Euclidean margin loss,
  sampled/skipped counts, and feature-device int64 positions `[T,8]` in the
  order `[instance,image,ay,ax,py,px,ny,nx]`. Skipped instances do not consume
  RNG. Empty sampling returns differentiable finite zero without summing large
  input values. Non-finite features/loss raise `FloatingPointError` with
  method, shot, batch, and feature-shape context; malformed batch inputs fail
  explicitly before sampling.
- Loss arithmetic promotes half/bfloat16 samples to float32 and preserves
  float32/float64 precision otherwise. L2 normalization first scales large
  finite vectors to prevent norm overflow. The denominator floor is `1e-4`
  for float16 inputs to prevent cast-back gradient overflow at zero/near-zero
  vectors, and `1e-12` otherwise; this numerical policy is documented in the
  public seam. Triplet distance uses `p=2`, `eps=0`, mean reduction, no swap.
  Literal analytic loss cases, satisfied zero loss, active nonzero feature
  gradients, and finite float16/bfloat16 backward cases pass.
- Independent read-only review found two Important issues: nearest-expanding
  the BG union could miss a GT intersection for width `3 -> 5`, and finite
  promoted loss could still backpropagate Inf into mixed zero/nonzero float16
  features. Four regression cases ran RED (**4 failed, 63 deselected**) before
  fixing original-source occupancy and normalization stability. The final
  focused/full GREEN results above include both fixes. No Critical or Minor
  findings were reported.
- `test_horizontal_flip_keeps_mask_feature_alignment` transforms a synthetic
  current image, instance mask, P3-like feature grid, and odd letterbox validity
  together; sampled FG stays on object features and BG stays entirely inside
  unpadded source content before and after the flip. No original COCO polygon
  is reconstructed. Actual loader/P3 alignment, CUDA/AMP training and memory
  measurements remain T09/T10 gates; native-loss weighting/logging, CLI
  activation, checkpoint behavior and runtime dependency pinning remain later
  tasks. T09 was not started; no real preparation/training/source writes ran.

## T09 — Project-owned Ultralytics extension and compatibility test

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T06, T08.  
**Files:** Create `camo_fs/ultralytics_ext.py`, `tests/test_ultralytics_ext.py`, `tests/test_ultralytics_integration.py`.  
**Produces:** A target-version adapter that extracts/verifies actual letterbox geometry for T07, `P3Capture`, `FGSegmentationTrainer`, and a model loss override that adds weighted triplet loss to the native scalar without changing the verified native return schema.  
**Acceptance:** The adapter reads actual preprocessing geometry from the installed loader and validates it against source-image dimensions and batch placement; absent or contradictory metadata fails clearly. First `Segment` head input is captured for the current forward only; it is spatial P3/8 with expected batch/channel dimensions and gradients. Native loss is unchanged and included. The enhanced `loss()` preserves the exact return structure, loss-item count/order/type, and tuple/dict semantics expected by the installed trainer; auxiliary metrics are logged separately. Hook state is cleared before the next forward and after the step. At least one prepared multi-polygon label is accepted by the target segmentation dataloader as one instance without corrupt-label warning or silent drop. Incompatible installed versions fail before a long training run.

- [x] Inspect `ultralytics.__version__`, actual loader/batch geometry metadata and letterbox rounding, `SegmentationTrainer.get_model`, `SegmentationModel.loss`, the `Segment` head input structure, and native loss return contract **in the actual target environment**. Record concrete signatures, metadata, and observed shapes in a short integration note in the design doc or README. Do not pick a numeric layer index from upstream `main` alone.
- [x] Write fake-module tests for one capture, missing capture, wrong stride/shape, stale feature, multiple unexpected capture, and hook reference release. Include `test_capture_rejects_stale_or_multiple_forward`, `test_loss_preserves_native_return_contract`, and `test_enhanced_p3_gradient_and_baseline_bypasses_triplet`. The latter asserts a nonzero gradient on the captured P3 tensor and verifies baseline neither instantiates nor calls the triplet sampler.
- [x] Run `pytest -q tests/test_ultralytics_ext.py`; confirm red.
- [x] Implement the target-version geometry adapter that extracts/verifies actual loader metadata and passes explicit geometry to T07; fail clearly if the required geometry cannot be established. Implement semantic `Segment` discovery, scoped forward pre-hook, guarded capture lifecycle, and the smallest `SegmentationTrainer.get_model` / `SegmentationModel.loss` extension. Preserve the **observed target-version native return structure exactly**; add weighted triplet loss only to the appropriate native loss scalar and log raw/weighted auxiliary values through a separate side channel or callback. Never append another tuple element, dictionary key, or loss item merely for logging.
- [x] Run focused and full synthetic tests. Then run `pytest -q -m integration` on Kaggle with attached `yolo11n-seg.pt`; assert the metadata adapter reproduces actual loader image/mask/valid-region placement, checkpoint loads, P3/8 capture matches actual head input, the batch supplies transformed per-instance masks aligned with `batch_idx`, a batch produces finite native+auxiliary loss with the exact native return contract, and auxiliary gradients reach both captured P3 and a neck/backbone parameter. Load at least one **prepared** merged multi-polygon label through this target-version segmentation dataloader and assert it remains exactly one source instance, all normalized coordinates are accepted, and no corrupt-label warning or silent drop occurs. Stop on incompatibility; do not silently train baseline.

**Initial local implementation evidence at 3366318 (T09 target gate remains open):**

- Added `ultralytics_ext.py` for guarded capture, loss, metadata adapters and
  lazy native bindings; `_ultralytics_83228.py` owns the concrete native classes.
  The baseline training path still uses its native trainer; enhanced CLI dispatch
  remains T10. No real local CAMO-FS preparation/training ran.
- Source inspection of candidate 8.4.172 found training centered affine warp
  instead of the integer letterbox contract. Explicitly selected **8.3.228 as a
  provisional CPU compatibility candidate**. Other versions fail closed.
  This is not a certified Kaggle target or the T10 dependency pin.
- TDD observed RED for missing capture/loss/geometry/trainer APIs, followed by
  behavioral RED/GREEN cases for duplicate/stale capture, wrong head inputs,
  native loss schema/items and incomplete collated geometry.
- Native 8.3.228 CPU proof uses the actual `SegmentationTrainer.get_model`,
  `YOLODataset`, `DataLoader`, `Segment`, and segmentation criterion with
  random YAML weights and temporary synthetic imagery. One T04-prepared merged
  multi-polygon label stays one instance, with zero corrupt-label messages;
  image/mask/bbox parity is exact against the native pipeline, including an
  asymmetric horizontal flip. Auxiliary-only gradients reach P3 and backbone.
- Observed native return: `tuple(Tensor[4], detached Tensor[4])`, ordered
  box/seg/cls/dfl. The trainer sums the first vector. Add the weighted mean
  auxiliary once to its first entry and preserve the original detached items
  object; metrics use `model.triplet_metrics`, never extra return items. The
  review correction below supersedes the original unscaled auxiliary addition.
- Checkpoint serialization/reload through native `YOLO` passes for random
  synthetic weights. This does not prove the attached COCO checkpoint, a
  completed training epoch, or paired predicted boxes/masks; those runtime
  gates remain open.
- Final focused suite: **59 passed, 1 skipped** (3.92s). Final full suite:
  **431 passed, 2 skipped** (19.37s), using the workspace-local venv and
  `python -m pytest -q --tb=short`. Skips are explicit real-data and
  attached-checkpoint gates. Graph-reference release passes on success/errors.
- Independent reviewer found no Critical/Important issue and separately ran
  **33 unit tests**. Minor run/shot/batch context in auxiliary error diagnostics
  is deferred to T10's dispatch/logging plumbing. Target CUDA/AMP acceptance,
  persistent logging, real smoke predictions and resume RNG continuity were
  explicitly outside this local review; their runtime gates remain open.
- Actual Kaggle inspection, target adapter certification, attached-checkpoint
  GPU integration and its prepared multi-polygon proof remain unchecked above.
  See the integration note in `camo-fs-yolo-design-decisions.md` for opt-in
  commands; T10 smoke/version-pin gates are not satisfied by CPU evidence.

**Review corrections to 3366318 (local evidence, runtime gates still open):**

- Exact official CAMO-FS `-0.5` left/top sentinels are recognized in bbox
  validation and normalized to zero in newly allocated polygon points. All
  other negative coordinates, right/bottom overflow, non-finite/nonpositive
  bbox sizes, and polygons with fewer than three distinct non-collinear points
  after normalization remain errors. Source dictionaries/files are unchanged.
- The former real-data marked test checked counts and metadata, not converter
  geometry, so it missed the shared-test/train conversion blocker. A dedicated
  parametrized conversion preflight now checks official test and all selected
  1/2/3/5-shot annotations, normalized bounds and source hashes, without creating
  prepared data. It no longer relies on hard-coded official annotation IDs.
- A direct audit against `data/` first failed because its images are extracted
  at `data/images`, while the contract expects `images/images`. A junction view
  under ignored `.superpowers/t09-review-input` maps those existing sources to
  the contract layout without copying/moving/changing source files. The
  strengthened real-data suite then returned **6 passed** (25.35s), including
  count audit and all five test/train geometry cases. This is local read-only
  data evidence, not a Kaggle training/integration result.
- Verified the installed 8.3.228 criterion returns native `loss * batch_size`
  and unchanged `loss.detach()`. Add `weight * raw_triplet_mean * actual B`
  exactly once to native vector entry zero. `weighted_triplet` reports that
  actual objective contribution; native items/order/type remain unchanged.
  Batch-one and batch-three native tests check this scale and auxiliary-only
  gradients; the batch-four unit regression prevents relative-weight shrinkage.
- Weight zero directly delegates to native prediction/loss. Sampling, capture,
  auxiliary geometry checks and sampler RNG advancement are bypassed; native
  objective/items match baseline exactly and auxiliary metrics/counts are zero.
- `get_model` now constructs the project subclass directly with native
  cfg/nc/ch/rank-aware verbosity and native weight loading. Focused parity
  tests check constructor execution and all state tensors with/without weights.
- RED evidence: **3 bbox** cases rejected legal sentinels; **6 polygon/preparation**
  cases rejected legal boundaries or failed before post-normalization topology;
  **3 batch-scale** cases lost the B factor; **6 zero-weight** cases invoked
  sampling or demanded unrelated auxiliary inputs; **3 factory/version** cases
  skipped subclass initialization or lacked an explicit install instruction.
- The user-provided first Kaggle inspection was Torch **2.11.0+cu128**, GPU
  **Tesla T4**, Ultralytics **8.4.172**. The extension still supports only the
  explicit **8.3.228 provisional candidate** and errors instruct
  `pip install ultralytics==8.3.228` before integration. `requirements.txt` is
  unchanged; no 8.4.172 support, Kaggle certification, T10 work or experiment
  result is claimed.
- Final required focused run (`tests/test_segments.py tests/test_prepare.py
  tests/test_valid_region.py tests/test_triplet.py tests/test_ultralytics_ext.py
  tests/test_ultralytics_integration.py`): **321 passed, 1 skipped** (11.25s).
  The extension/native-only focused run: **69 passed, 1 skipped** (7.06s).
- Full `python -m pytest -q --tb=short` with `CAMO_FS_DATA_ROOT` set to the
  read-only view: **478 passed, 1 skipped** (33.93s). All six real-data tests
  ran and passed. The single skipped test is the opt-in attached-checkpoint
  Kaggle CUDA gate because `CAMO_FS_INTEGRATION_WEIGHTS` is unset; it is not
  counted as a pass. The independent reviewer found no Critical/Important/Minor
  issue in these targeted corrections.
- T09 is **local/synthetic-complete for the guarded candidate, runtime-incomplete**.
  On Kaggle, explicitly install 8.3.228, prepare T04 under `/kaggle/working`,
  export the source/weights paths shown in the design note, and run
  `python -m pytest -q -m integration -s` with the original attached checkpoint.
  CUDA/AMP compatibility and the target-version prepared multi-polygon proof
  remain open. T10's epoch smoke, trained checkpoint predictions and final
  dependency pin are untouched; the eight full experiments were not run.

## T10 — Enhanced one-epoch smoke gate and version pin

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T09.  
**Files:** Modify `camo_fs/train.py`, `requirements.txt`, `tests/test_train.py`; add smoke-run instructions/results section to `README.md` when evidence exists.  
**Produces:** A recorded enhanced run gate before full experiments.  
**Acceptance:** One official shot (prefer 1-shot) completes a short enhanced run on Kaggle single GPU. Some batches have nonzero raw triplet loss; gradients are nonzero; every logged loss is finite. Enhanced `last.pt` saves, reloads through the same inference-loading path used by evaluation/visualization, and yields at least one paired predicted box and segmentation mask on prepared **training** imagery. An empty result alone does not satisfy this gate. No official test image or metric is used for this smoke check or to tune the loss.

- [x] Write tests showing `--method fgbg-triplet` routes through the extension while `baseline` bypasses it, enhanced `device` rejects multi-GPU, and both modes receive identical non-method training options.
- [x] Run the focused test red; implement the dispatch; rerun focused and full suites.
- [x] Prepare the official 1-shot split on Kaggle and run `--method fgbg-triplet --shot 1 --epochs 1 --device 0 --seed 2024` with `run_kind=smoke` and the base checkpoint. Save console log, manifest, triplet counts, finite-loss proof, and `last.pt` under `/kaggle/working`. Reload that checkpoint through the evaluation/visualization inference-loading path and predict on prepared **train** images until at least one prediction contains both a box and its instance mask. If the first image has no detections, try other prepared training images or a lower smoke-only confidence threshold; record that threshold and leave the gate open if no paired output appears. Do not load an official test image for this proof.
- [x] Conditional zero-triplet remediation is **not applicable**: the accepted Kaggle smoke produced finite nonzero triplet signal and nonzero P3/backbone gradients. No remediation run is claimed; the eight full experiments remain under T13.
- [x] After the accepted smoke gate, pin `ultralytics==8.3.228` in `requirements.txt`, record Torch/CUDA versions, and confirm post-pin integration. User-provided evidence is recorded above. Final README consolidation remains under T13.

T10 local implementation evidence (official Kaggle gate remains open):

- Enhanced CLI binds the guarded project trainer through a lazy callable;
  triplet arguments stay outside native overrides. Baseline keeps its native
  path. Single GPU 0, common option parity and own-checkpoint resume are tested.
- Enhanced batch logs preserve native loss items and separately record finite
  raw/weighted/combined loss and sample/skip counts in `triplet_batches.jsonl`.
  Both manifest count fields and `triplet_training` totals agree; CUDA runtime
  is recorded alongside Torch/library versions.
- `UltralyticsRuntime.load_inference` accepts completed stripped checkpoints
  without resume state. A real synthetic CPU epoch through the native enhanced
  factory saved `last.pt`, reloaded it and predicted on fixture TRAIN imagery.
- `tests/test_training_smoke.py` is the opt-in official 1-shot CUDA gate. It
  invokes the CLI, observes real auxiliary-only P3/backbone gradients, then
  requires paired box/nonempty-mask prediction on audited TRAIN images via
  the shared inference loader. It saves `smoke_gate.json` and a prediction
  image under the run directory. README provides the runnable gate command.
- TDD observed RED then GREEN for dispatch, runtime binding, logging, completed
  checkpoint inference and a review regression for inconsistent manifest
  counts. Initial T10 full verification with the existing read-only official input
  view: **495 passed, 2 skipped**. Skips are the opt-in attached-checkpoint
  T09 gate and the new T10 epoch gate without configured weights/CUDA; the
  local T09 skip does not change the user's prior T01–T09 acceptance.
- Actual local runtime: Python 3.13.14, Ultralytics 8.3.228,
  Torch 2.14.1+cpu, no CUDA. Requirements remain unpinned. No official T10
  Kaggle epoch or eight full experiments were executed in this implementation.

T10 official-test isolation review correction:

- Confirmed the installed 8.3.228 trainer constructs its validation loader
  from `data.get("val") or data.get("test")` and validates at the final epoch
  even with `val=False`. Missing `val` therefore cannot be allowed in training.
- A shared training-data guard now runs before the CLI's initial data hash,
  before `train_one` initializes/overwrites a run or loads a model, and against
  parsed native trainer data. Both methods require own TRAIN paths for `train`
  and nonempty `val`, plus the shared official path for `test`.
- Missing/null/empty `val`, `val=test`, or misrouted `test` is rejected. Failure
  reports remain visible; rejected YAML is preserved and no run is created.
  T04's optional-placeholder output contract and T05 hashing remain unchanged.
- TDD reproduced missing YAML `val` and missing/null parsed native `val` as
  failures to raise, then verified GREEN. A CLI regression separately caught
  hash-layer errors preceding the actionable preflight and verified the fix.
  Training doubles now preserve actual YAML fields; success fixtures explicitly
  prepare the TRAIN placeholder instead of silently inserting it in the double.
- Fresh review and final full verification: **513 passed, 2 skipped**, including
  the official read-only audits and native CPU epoch/reload probe. The two
  opt-in CUDA/checkpoint gates remain unexecuted locally; prior T01–T09
  acceptance stands. README uses `--no-deps` for the candidate on an already
  compatible Kaggle runtime. T10's actual Kaggle smoke and final pin remain open.

## T11 — Official-test evaluation, six metrics, and summary upsert

**Current state: ACCEPTED / FROZEN by the user.**

**Depends on:** T05, T06; runtime confirmation after T10 for enhanced.  
**Files:** Create `camo_fs/evaluate.py`, `scripts/evaluate_yolo.py`, `tests/test_evaluate.py`.  
**Produces:** `evaluate_one(run_dir, data_yaml, results_dir) -> dict` and `results/summary.csv`.  
**Acceptance:** Loads only the run's `last.pt`, explicitly evaluates `split="test"`, extracts Box AP/AP50/AP75 and Mask AP/AP50/AP75, and upserts by complete run identity. Failed runs have `status=failed`, error text, and empty metrics. Repeating evaluation updates a row without duplication. Batch selection includes `--method` and `--shot {1,2,3,5,all}`; if multiple configuration hashes match, process every completed benchmark run as distinct. Smoke runs are excluded by default.

- [x] Write fake-validator tests for exact metric extraction, missing `last.pt`, accidental `val` split rejection, failed row, idempotent rerun, batch discovery across shot `all`, default smoke exclusion, and `test_upsert_preserves_distinct_config_hashes` (same method/shot/seed but different config hashes remain separate).
- [x] Run `pytest -q tests/test_evaluate.py`; confirm red.
- [x] Implement test-only evaluation and atomic CSV upsert via the fixed-schema helper from T05. Validate source YAML `test` points to shared official prepared test path, and never choose weights using official test results.
- [x] Run focused and full tests. Keep evaluation plumbing checks synthetic until the planned final official-test evaluation; do not use the official test set during smoke training or to decide training settings.

T11 local implementation and verification:

- `evaluate_one` verifies manifest identity/directory ownership, the run's own
  completed `last.pt`, own-shot YAML, shared prepared test path, saved data
  checksum, and checkpoint taxonomy before inference. It reuses T10's
  `load_inference`, never the resume loader or `best.pt`.
- Native validation explicitly uses `split="test"`, no TTA or class filtering,
  nonoverlapping instance masks, and fixed native evaluation settings
  `conf=0.001`, `iou=0.7`, `max_det=300`. Image size, batch, device, seed and
  mask ratio come from the run config. Smoke confidence `1e-4` is not reused.
  Six finite AP fractions come directly from `box/seg.map/map50/map75`.
- T05's fixed-schema atomic summary helper is reused unchanged. Evaluation
  failures write a failed row with error and empty scores, preserving training
  status/checkpoints. A later success replaces that row; every configuration
  hash remains distinct. The summary's `weights_path` is the evaluated `last.pt`;
  `weights_sha256` remains the original base-weight identity checksum.
- Discovery includes every completed matching benchmark identity and excludes
  smoke by default. CLI supports a single `--run-dir` or `--shot` selection,
  both methods, explicit `--include-smoke`, and `--continue-on-error`. There is
  no CLI override for checkpoint, split or confidence. Invalid manifests and
  unsafe output roots fail rather than fabricating identity/report rows.
- TDD observed RED then GREEN for missing evaluation APIs, failure/metric
  guards, dataset/identity/output isolation, discovery, checkpoint-path reporting
  and batch/CLI behavior. Two native **8.3.228** CPU probes save/reload stripped
  random synthetic baseline/enhanced checkpoints and observe only fixture TEST
  images in the actual segmentation validator; they are not benchmark results.
- Focused T11 verification: **37 passed**. Independent read-only review found
  no Critical, Important or Minor issue. Local runtime: Python **3.13.14**,
  Ultralytics **8.3.228**, Torch **2.14.1+cpu**, no CUDA.
- Full local verification: **552 passed, 2 skipped**, including the read-only
  official data audits. The two opt-in CUDA/checkpoint gates are skipped locally
  and do not replace or revoke the user's accepted T09/T10 Kaggle evidence.
  No official-test inference, full benchmark training or T12/T13 work ran.

T11 checkpoint-identity review correction:

- Before constructing/seeding the runtime or loading a model, evaluation now
  requires integer `completed_epochs` equal to the requested fixed epochs and
  verifies own `last.pt` against the recorded `last_checkpoint_sha256`. Missing
  completion evidence or changed checkpoint bytes produces a failed row with
  empty metrics; training manifests/checkpoints remain unchanged.
- Synthetic completion fixtures now record the actual checkpoint checksum and
  epoch count before transitioning to completed, including both native probes.
  Regression RED: **10 failed, 37 deselected**. Focused GREEN: **47 passed**.
  Full GREEN: **562 passed, 2 skipped**, with official read-only audits included.
  T01–T10 source is unchanged. The user subsequently accepted and froze T11;
  no official-test evaluation or new Kaggle gate ran for this correction.

Commands for the planned final evaluation, **after fixed-epoch benchmark runs**:

```bash
python scripts/evaluate_yolo.py --method baseline --shot all
python scripts/evaluate_yolo.py --method fgbg-triplet --shot all
python scripts/evaluate_yolo.py --run-dir "/kaggle/working/runs/yolo11n-seg/baseline/shot_1/seed_2024/<config_hash>"
```

Replace `<config_hash>` with the actual run's fingerprint. Custom local roots
require `--work-root` and `--data-root`. The CSV is at
`<work-root>/results/summary.csv`; native evaluation artifacts stay inside
`<run-dir>/evaluation`. Batch processing stops at the first failure unless
`--continue-on-error` is explicitly requested. All eight benchmark runs and
their official-test AP results remain the later full-comparison gate.

## T12 — Deterministic prediction and sampler debug visuals

**Depends on:** T08, T11.  
**Files:** Create `camo_fs/visualize.py`, `scripts/visualize_predictions.py`, `tests/test_visualize.py`.  
**Produces:** `visualize_one(...)` plus optional training-only sampler debug render.  
**Acceptance:** For the same seed and official test image list, select the same 20 images by default; draw predicted boxes, masks, class names, and confidence; separate directories by method/shot/seed/config hash; never overwrite another run's images. Support one run or method/shot `all` benchmark discovery, excluding smoke runs by default.

- [x] Write tests for deterministic image selection, `num_images` cap, `conf=0.25` forwarding, empty predictions, filename collision avoidance, method/shot `all` run discovery, smoke exclusion, and sampled-point overlay being constrained by GT/valid masks.
- [x] Run `pytest -q tests/test_visualize.py`; confirm red.
- [x] Implement rendering using prediction outputs and the saved taxonomy; keep sampler debug rendering optional and outside every-batch hot path.
- [x] Run focused and full tests; visually inspect known synthetic prediction and sampler fixtures to confirm geometry/mask alignment.
- [ ] Visually inspect a Kaggle sample. Local synthetic evidence does not satisfy this target gate. Do not use official-test predictions to tune training or choose checkpoints.

T12 local implementation and verification:

- Deterministic sorted-list selection uses an isolated RNG, default seed 2024
  and 20 images, capped at the available count. Rendering loads only the run's
  checksum-verified completed `last.pt`, checks fixed-epoch evidence, own-shot
  YAML/shared-test paths, saved data checksum and taxonomy. T01–T11 source is
  unchanged; public T11 discovery and T10 inference loading are reused.
- Native box/mask/class/confidence plotting requests original-resolution
  instance masks and exports BGR output to RGB PNG. Confidence defaults to
  0.25 for visualization only; AP evaluation settings and CSV are untouched.
  Empty detections still save the image and record zero detections.
- Output is `<run-dir>/visualizations/`, already separated by complete run
  identity. Filenames hash the full relative image path, preserving distinct
  nested basenames. `manifest.json` records identity, checkpoint checksum,
  selection seed, confidence and selected source/output/detection records.
  Selection seed is separate from the run's original training seed.
- Reruns replace only verified owned output, after every new image succeeds.
  Render/publish failures preserve previous output. If filesystem rollback
  also fails, a named recovery backup is retained. Synthetic fault tests check
  both outcomes. Unmanaged output or another run's identity is rejected.
- CLI selects one run or every matching completed configuration across methods
  and shots. Smoke is excluded unless explicitly opted in; failure stops the
  batch unless `--continue-on-error` is supplied. Each attempt reports full
  identity and error/count, with a nonzero exit if any attempt fails.
- Optional `render_sampler_debug` is a pure, TRAIN-only PIL renderer taking
  the current transformed RGB image, per-instance GT masks, batch indices,
  input-validity mask and T08 `sampled_positions`/feature shape. It does not
  sample, add hooks or enter training's hot path. It verifies nearest-projected
  same-instance A/P, conservative all-GT-free N and all-valid feature cells.
  The left `FG grid` panel shows projected foreground and actual cell centers;
  the right `GT source` panel shows transformed source GT without points.
  Red/green/blue indicate A/P/N; padding is dark in both panels.
- TDD slices observed RED/GREEN: selection **7**, native render/empty **3**,
  completion/data/result guards **20**, ownership/report transaction **4**,
  batch/CLI **6**, sampler debug **9**. Independent review identified one
  fractional-grid debug mismatch: its pixel regression failed before the
  two-panel fix, then passed. No Critical finding was reported.
- Focused final verification: **57 passed**. Native pinned **8.3.228** probes
  render real `Results` and reload synthetic stripped checkpoints for both
  methods; all inference is on fixture images. Actual PNGs were inspected for
  box/mask alignment, readable class/confidence, RGB export and both debug
  geometries. Local runtime: Python **3.13.14**, Torch **2.14.1+cpu**, no CUDA.
- Full final verification: **619 passed, 2 skipped**, including all six official
  read-only audit/conversion tests. The skips are the existing opt-in T09
  attached-checkpoint CUDA and T10 official smoke gates; neither is counted
  as a pass or changes the accepted foundation evidence.
- No Kaggle T12 visual sample, official-test inference or full benchmark ran.
  T12 awaits the user's review; T13 and the eight-run comparison remain open.

T12 native batch-index review correction:

- Confirmed the frozen T09 extension accepts native floating-point `batch_idx`
  only when finite and integral, then normalizes it to `int64`. T12's debug
  renderer now follows this rule on a detached CPU tensor, preserving input
  dtype and values. Integer input remains supported; fractional/nonfinite,
  out-of-range, count-mismatched and unsupported index types are rejected.
- Regression RED: **2 failed, 57 deselected** for native `float32` indices in
  images 0 and 1. Both render A/P/N correctly after the fix and assert every
  input unchanged. Additional rejection cases cover `0.5`, NaN, positive and
  negative infinity, negative/out-of-batch indices, missing indices and
  bool/complex types. Focused GREEN: **68 passed**.
- Full GREEN: **630 passed, 2 skipped**, including the six official read-only
  audits. The two existing opt-in CUDA/checkpoint skips are not counted as
  passes and do not change the foundation's accepted evidence.
- T01–T11 source and prediction visualization are unchanged. This correction
  does not verify the Kaggle visual sample; T12 remains **not frozen**.

Commands for prediction renders from completed runs:

```bash
python scripts/visualize_predictions.py --method baseline --shot all
python scripts/visualize_predictions.py --method fgbg-triplet --shot all
python scripts/visualize_predictions.py --method all --shot all --num-images 20 --conf 0.25 --seed 2024
python scripts/visualize_predictions.py --run-dir "/kaggle/working/runs/yolo11n-seg/baseline/shot_1/seed_2024/<config_hash>"
```

Replace `<config_hash>` with the actual fingerprint; custom local roots require
`--work-root` and `--data-root`. `--include-smoke` only changes run discovery;
this prediction CLI always uses the prepared shared TEST list. T10's existing
TRAIN-only smoke proof stays separate. Optional TRAIN debug use outside the
hot path (save under the managed work root):

```python
from camo_fs.visualize import render_sampler_debug

debug = render_sampler_debug(
    batch["img"][0], batch["masks"], batch["batch_idx"].reshape(-1),
    valid, triplet_result.sampled_positions, tuple(p3.shape[-2:]),
    image_index=0, split="train",
)
debug.save("/kaggle/working/sampler-debug.png")
```

Here `valid`, `triplet_result` and `p3` must belong to the same transformed
TRAIN batch; never substitute predicted masks or test imagery.

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
