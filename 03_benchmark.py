"""Step 3 — Benchmark a trained model and log the iteration.

Run:  python 03_benchmark.py --tag v1 --seeds 42

What it does:
  1. Calls drive2win.benchmark.run_benchmark for each --seeds value (5 runs each).
  2. Saves benchmarks/<tag>.json with summary numbers AND the run tracks. This
     is what the instructor grades for "process". Commit it after every
     iteration.
  3. Saves PNGs:
       benchmarks/<tag>_paths.png         all run paths overlaid
       benchmarks/<tag>_progress.png      checkpoints per run
       benchmarks/<tag>_overlay.png       your drive (gray) vs NN test (blue)
                                          (only if you pass --data data_<tag>.npz)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from drive2win.benchmark import run_benchmark
from drive2win import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="Iteration tag, e.g. v1, v2-recovery, v3-deepnet.")
    ap.add_argument("--weights", default=None,
                    help="Path to weights file. Defaults to nav_<tag>.npz.")
    ap.add_argument("--module", default=None,
                    help="Optional adapter module (drive2win.cnn etc).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Map seed(s) to test on. Use one for fast iteration; "
                         "use several to test generalisation across terrains.")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--data", default=None,
                    help="Optional data_<tag>.npz for the training-vs-test path "
                         "overlay.")
    ap.add_argument("--no-obstacles", action="store_true",
                    help="Disable arena obstacles (match --no-obstacles in collect).")
    args = ap.parse_args()

    weights = args.weights or f"nav_{args.tag}.npz"
    out_dir = Path("benchmarks"); out_dir.mkdir(exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        result = run_benchmark(
            weights=weights, runs=args.runs, seed=seed,
            duration=args.duration, module=args.module,
            obstacles=not args.no_obstacles,
        )
        all_results.append({"seed": seed, **result})

    # ---- print headline numbers ----
    print("\n" + "=" * 56)
    print(f"  iteration: {args.tag}    weights: {weights}")
    for r in all_results:
        s = r["summary"]
        print(f"  seed {r['seed']:>4}  "
              f"complete={int(s['completion_rate'] * s['n_runs'])}/{s['n_runs']}  "
              f"median_lap={s['median_lap_time']:.1f}s  "
              f"crashes={s['mean_crashes']:.1f}  "
              f"max_cp={s['max_checkpoints']}")
    print("=" * 56)

    # ---- write JSON log ----
    log_path = out_dir / f"{args.tag}.json"
    log = {
        "tag": args.tag, "weights": weights, "module": args.module,
        "runs_per_seed": args.runs, "duration_s": args.duration,
        "seeds": [
            {"seed": r["seed"], "summary": r["summary"], "runs": r["runs"]}
            for r in all_results
        ],
    }
    log_path.write_text(json.dumps(log, indent=2, default=float))
    print(f"\nwrote {log_path}")

    # ---- visuals ----
    flat_runs = [run for r in all_results for run in r["runs"]]
    viz.plot_multi_run_paths(flat_runs,
                             out=str(out_dir / f"{args.tag}_paths.png"),
                             title=f"All paths — {args.tag}")
    viz.plot_checkpoint_progress(flat_runs,
                                 out=str(out_dir / f"{args.tag}_progress.png"))

    if args.data:
        d = np.load(args.data, allow_pickle=False)
        train_xz = d["positions"] if "positions" in d.files else None
        first_track = flat_runs[0].get("track") or []
        viz.plot_path_overlay(train_xz, first_track,
                              out=str(out_dir / f"{args.tag}_overlay.png"),
                              title=f"{args.tag} — your drive (gray) vs NN (blue)")


if __name__ == "__main__":
    main()
