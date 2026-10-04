"""Thin CLI for audited official CAMO-FS segmentation preparation."""

import argparse
from pathlib import Path
import sys

from camo_fs.paths import DatasetPaths
from camo_fs.prepare import prepare_selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit official CAMO-FS JSON/images before copying YOLO segmentation splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shot", choices=["1", "2", "3", "5", "all"], required=True,
                        help="Official support shot, or all shots in order")
    parser.add_argument("--data-root", type=Path, default=DatasetPaths.DEFAULT_DATA_ROOT.as_posix(),
                        help="Read-only official dataset root")
    parser.add_argument("--work-root", type=Path, default=DatasetPaths.DEFAULT_WORK_ROOT.as_posix(),
                        help="Writable output root")
    parser.add_argument("--overwrite", action="store_true", help="Completely rebuild affected targets")
    parser.add_argument("--continue-on-error", action="store_true", help="Record shot failures and attempt later shots")
    parser.add_argument("--val-train-placeholder", action="store_true",
                        help="Set val to train only for parser compatibility; training still requires val=False")
    args = parser.parse_args(argv)
    try:
        paths = DatasetPaths.from_root(args.data_root, args.work_root)
        shots = [1, 2, 3, 5] if args.shot == "all" else [int(args.shot)]
        outcomes = prepare_selected(shots, paths, args.overwrite, args.continue_on_error,
                                    val_train_placeholder=args.val_train_placeholder)
    except (OSError, ValueError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    for outcome in outcomes:
        print(f"shot {outcome.shot}: {outcome.status}: {outcome.data_yaml or outcome.error_message}")
    return int(any(outcome.status != "prepared" for outcome in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
