"""Thin CLI for deterministic run-owned CAMO-FS test prediction renders."""

import argparse
from pathlib import Path
import sys

from camo_fs.paths import DatasetPaths
from camo_fs.visualize import visualize_selected


def main(argv: list[str] | None = None, *, runtime=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render boxes, instance masks, class names and confidence from each run's verified last.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path, help="One run; infer shot/method from its manifest")
    selection.add_argument("--shot", choices=["1", "2", "3", "5", "all"], help="Every completed matching config hash")
    parser.add_argument("--method", choices=["baseline", "fgbg-triplet", "all"],
                        help="Batch default: baseline; single-run default: the manifest's method")
    parser.add_argument("--num-images", type=int, default=20, help="Cap at available test images")
    parser.add_argument("--conf", type=float, default=0.25, help="Visualization confidence, independent of AP evaluation")
    parser.add_argument("--seed", type=int, default=2024, help="Deterministic image-selection seed")
    parser.add_argument("--include-smoke", action="store_true", help="Explicitly include smoke identities")
    parser.add_argument("--continue-on-error", action="store_true", help="Report failure and attempt later runs")
    parser.add_argument("--data-root", type=Path, default=DatasetPaths.DEFAULT_DATA_ROOT.as_posix())
    parser.add_argument("--work-root", type=Path, default=DatasetPaths.DEFAULT_WORK_ROOT.as_posix())
    args = parser.parse_args(argv)
    shots = None if args.run_dir else [1, 2, 3, 5] if args.shot == "all" else [int(args.shot)]
    try:
        paths = DatasetPaths.from_root(args.data_root, args.work_root)
        outcomes = visualize_selected(shots, paths, method=args.method, run_dir=args.run_dir,
                                       num_images=args.num_images, conf=args.conf, seed=args.seed,
                                       include_smoke=args.include_smoke,
                                       continue_on_error=args.continue_on_error, runtime=runtime)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Visualization failed: {error}", file=sys.stderr)
        return 1
    for row in outcomes:
        print(f"{row['method']} shot {row['shot']} {row['run_kind']} {row['config_hash']}: "
              f"{row['status']}: {row['error_message'] or str(len(row['images'])) + ' images'}")
    return int(any(row["status"] != "completed" for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
