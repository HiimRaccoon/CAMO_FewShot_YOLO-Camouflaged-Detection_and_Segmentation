# CAMO-FS YOLO

Compare COCO-pretrained YOLO11n-Seg with the same model plus a class-agnostic
foreground/background triplet objective on official CAMO-FS shots. T01–T09
are accepted. T10's enhanced dispatch and smoke gate are implemented; its
official Kaggle epoch gate remains pending. Evaluation, visualization CLIs,
and the eight full experiments belong to T11–T13.

## T10 setup on Kaggle

Enable one GPU and run commands from this repository's root after cloning
or attaching it. Install the guarded **candidate** explicitly:

```bash
python -m pip install -e .
python -m pip install -q --no-deps 'ultralytics==8.3.228'
```

The adapter supports only 8.3.228 and fails on other versions. This installation
command is separate from the final dependency pin: `requirements.txt` remains
unpinned until the T10 gate actually succeeds on Kaggle. Retain Kaggle's working
Torch/CUDA runtime; the gate records actual versions and GPU.
The `--no-deps` installation assumes Kaggle's compatible Torch, Torchvision,
NumPy and other required packages are already available, along with pytest.
It replaces Ultralytics without re-resolving the verified runtime dependencies.

Attach the original COCO `yolo11n-seg.pt` as Kaggle Input and use its local
path. Internet-off training requires that checkpoint and compatible packages
to be available locally. With Internet enabled, download the original weights
before invoking training; the training CLI requires an existing local `.pt`.

```bash
export CAMO_FS_DATA_ROOT=/kaggle/input/datasets/danhnt/camo-fs-dataset
export CAMO_FS_WORK_ROOT=/kaggle/working
export CAMO_FS_SMOKE_WEIGHTS=/kaggle/input/<attached-weights>/yolo11n-seg.pt

python scripts/prepare_dataset.py --shot 1 --val-train-placeholder \
  --data-root "$CAMO_FS_DATA_ROOT" --work-root "$CAMO_FS_WORK_ROOT"
```

Replace `<attached-weights>` with the attached dataset path. Source images must
be at `<data-root>/images/images`; source files are read only. Preparation
creates 47 training images/instances with the canonical 47-class taxonomy and
the shared official test split. Multi-polygon instances retain all components
in one label; connecting components may compromise disconnected COCO topology.
If preparation already exists and passes the audit, proceed to the gate.
Training additionally requires `val` to point to that shot's TRAIN images and
`test` to the shared official test directory. Missing, null or empty `val`
fails before a run is initialized or a model is loaded, because 8.3.228 would
otherwise build its validation loader from `test`. Both training methods and
the parsed native trainer enforce this check. Preparation's optional-placeholder
contract is unchanged; do not edit an existing YAML in place, since it enters
the data fingerprint. Use preparation's explicit `--overwrite` and
`--val-train-placeholder` flags to rebuild a target that lacks the placeholder.

## Run the official T10 gate

```bash
set -o pipefail
python -m pytest -q -s tests/test_training_smoke.py \
  2>&1 | tee /kaggle/working/t10-smoke-console.log
```

This invokes the real training CLI with `--method fgbg-triplet --shot 1
--epochs 1 --device 0 --seed 2024 --run-kind smoke`, the original weights,
and default triplet settings (weight 0.1, margin 0.3, count 16). It checks
finite native/auxiliary/combined losses, valid samples, nonzero raw loss in
some batch, and nonzero auxiliary-only P3/backbone gradients. Gradient
observations do not change the loss, sampler RNG, or accumulated gradients.

The gate reloads that run's trained `weights/last.pt` through
`UltralyticsRuntime.load_inference`, the shared loading seam for T11/T12.
It predicts only on prepared **TRAIN** images, trying confidence 0.25, 0.01,
then 0.001 until at least one paired box and nonempty instance mask appears.
It records the chosen smoke-only threshold. No official test image or metric
is used for this proof or to tune training. Native final validation may still
use the TRAIN placeholder despite `val=False`; its metrics are not held-out
results and do not satisfy official-test evaluation.

Artifacts are under `/kaggle/working/runs/yolo11n-seg/fgbg-triplet/shot_1/
seed_2024/<config_hash>/`:

- `manifest.json`: identity, effective options, versions including CUDA,
  completed epochs, checkpoint checksum and accumulated sample/skip counts.
- `triplet_batches.jsonl`: native loss items in box/seg/cls/dfl order,
  raw triplet mean, batch-scaled weighted contribution and combined objective.
- `weights/last.pt`, `smoke_gate.json`, and, on paired output,
  `smoke_prediction.jpg`.

`smoke_gate.json` must report `status=passed`. Training completion alone does
not pass this gate. Missing CUDA, incompatible version, absent signal or no
paired prediction fails. With no `CAMO_FS_SMOKE_WEIGHTS`, the test skips; a skip
is not a pass. Existing runs fail by default. To deliberately restart the same
smoke run from the original checkpoint, set `CAMO_FS_SMOKE_OVERWRITE=1` before
rerunning; it replaces that managed run's artifacts.

After a successful Kaggle gate, preserve the console log and run artifacts,
set the exact tested Ultralytics version in `requirements.txt`, rerun the
integration suite under that pin, and record the actual Torch/CUDA versions.
Until those steps have evidence, T10 stays open and full experiments wait.

## Training CLI and local verification

The CLI also accepts `--method baseline`, `--shot {1,2,3,5,all}`, `--epochs`,
`--imgsz`, `--batch`, `--seed`, and the three triplet options. Both methods
receive the same non-method options; baseline bypasses the extension.
Enhanced requires `--device 0`; baseline also permits `--device cpu`.
`--shot all` runs independent shots sequentially; `--continue-on-error`
records failures and continues. `--resume` uses only a matching interrupted
run's own checkpoint; `--overwrite` restarts from the original weights.

```bash
python -m pytest -q
```

Local verification includes a synthetic CPU epoch and native reload/inference.
The final full local suite with official read-only audits had 513 passing
tests and two explicit opt-in CUDA/checkpoint skips.
These checks do not certify the official Kaggle smoke. Local candidate runtime:
Python 3.13.14, Ultralytics 8.3.228, Torch 2.14.1+cpu, CUDA unavailable. Smoke
identities differ from benchmark identities; no AP improvement is claimed.
