"""Thin CLI for final official-test evaluation of completed last.pt runs."""

import argparse
from pathlib import Path
import sys

from camo_fs.evaluate import evaluate_selected
from camo_fs.paths import DatasetPaths


def main(argv: list[str] | None = None, *, runtime=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate only each run's last.pt on the shared official test; upsert six AP metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path, help="One run directory; infer shot/method from its manifest")
    selection.add_argument("--shot", choices=["1", "2", "3", "5", "all"],
                           help="Discover every completed matching config hash, in shot order")
    parser.add_argument("--method", choices=["baseline", "fgbg-triplet"],
                        help="Batch default: baseline; single-run default: the manifest's method")
    parser.add_argument("--include-smoke", action="store_true",
                        help="Explicitly include smoke runs, separately labeled in the summary")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failure and attempt later runs")
    parser.add_argument("--data-root", type=Path, default=DatasetPaths.DEFAULT_DATA_ROOT.as_posix())
    parser.add_argument("--work-root", type=Path, default=DatasetPaths.DEFAULT_WORK_ROOT.as_posix())
    args = parser.parse_args(argv)
    shots = None if args.run_dir else [1, 2, 3, 5] if args.shot == "all" else [int(args.shot)]
    try:
        paths = DatasetPaths.from_root(args.data_root, args.work_root)
        outcomes = evaluate_selected(shots, paths, method=args.method, run_dir=args.run_dir,
                                     include_smoke=args.include_smoke,
                                     continue_on_error=args.continue_on_error, runtime=runtime)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    for row in outcomes:
        print(f"{row['method']} shot {row['shot']} {row['run_kind']} {row['config_hash']}: "
              f"{row['status']}: {row['error_message'] or row['weights_path']}")
    return int(any(row["status"] != "completed" for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
