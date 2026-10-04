"""Thin CLI for independent CAMO-FS baseline/enhanced training attempts."""

import argparse
from pathlib import Path
import sys

from camo_fs.paths import DatasetPaths
from camo_fs.train import train_selected


def main(argv: list[str] | None = None, *, runtime=None) -> int:
    parser = argparse.ArgumentParser(description="Train CAMO-FS from a local COCO yolo11n-seg.pt checkpoint.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--shot", choices=["1", "2", "3", "5", "all"], required=True)
    parser.add_argument("--method", choices=["baseline", "fgbg-triplet"], default="baseline",
                        help="Native baseline or the version-guarded project triplet trainer")
    parser.add_argument("--weights", default="yolo11n-seg.pt", help="Local checkpoint; attach/download it before training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="Single GPU 0, or cpu for baseline")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--run-kind", choices=["benchmark", "smoke"], default="benchmark")
    parser.add_argument("--triplet-weight", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplets-per-instance", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=DatasetPaths.DEFAULT_DATA_ROOT.as_posix())
    parser.add_argument("--work-root", type=Path, default=DatasetPaths.DEFAULT_WORK_ROOT.as_posix())
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true", help="Continue only the matching interrupted run's own last.pt")
    recovery.add_argument("--overwrite", action="store_true", help="Restart the matching managed run from the original checkpoint")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failures and attempt later shots")
    args = parser.parse_args(argv)
    settings = {key: getattr(args, key) for key in ("method", "epochs", "imgsz", "batch", "device", "seed", "run_kind",
                                                  "triplet_weight", "triplet_margin", "triplets_per_instance")}
    try:
        paths = DatasetPaths.from_root(args.data_root, args.work_root)
        outcomes = train_selected([1, 2, 3, 5] if args.shot == "all" else [int(args.shot)], paths,
                                  weights=args.weights, settings=settings, resume=args.resume, overwrite=args.overwrite,
                                  continue_on_error=args.continue_on_error, runtime=runtime)
    except (OSError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    for item in outcomes:
        print(f"shot {item.shot}: {item.status}: {item.last_pt or item.error_message}")
    return int(any(item.status != "completed" for item in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
