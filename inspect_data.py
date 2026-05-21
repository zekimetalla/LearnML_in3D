"""Convert a data_*.npz file to CSV and print a detailed per-parameter report.

Usage:
    python inspect_data.py --data data_v11-clean.npz
    python inspect_data.py --data data_v11-clean.npz --param heading_error
    python inspect_data.py --data data_v11-clean.npz --param ray_0_front --samples 20

Outputs:
    data_v11-clean.csv        — full dataset, one row per frame
    data_v11-clean_stats.csv  — min/max/mean/std/percentiles per column
"""
from __future__ import annotations
import argparse
import numpy as np

FEATURE_NAMES = [
    "speed",
    "heading_error",
    "checkpoint_distance",
    "ray_0_front",
    "ray_1_+45",
    "ray_2_+90",
    "ray_3_+135",
    "ray_4_back",
    "ray_5_-135",
    "ray_6_-90",
    "ray_7_-45",
    "ground_friction",
]
ACTION_NAMES = ["throttle", "steering"]
ALL_COLS = FEATURE_NAMES + ACTION_NAMES

DESCRIPTIONS = {
    "speed":               "car speed (m/s)",
    "heading_error":       "angle to next checkpoint (rad, neg=right, pos=left)",
    "checkpoint_distance": "distance to next checkpoint (m)",
    "ray_0_front":         "front raycast distance (m)",
    "ray_1_+45":           "front-right raycast (m)",
    "ray_2_+90":           "right raycast (m)",
    "ray_3_+135":          "rear-right raycast (m)",
    "ray_4_back":          "rear raycast (m)",
    "ray_5_-135":          "rear-left raycast (m)",
    "ray_6_-90":           "left raycast (m)",
    "ray_7_-45":           "front-left raycast (m)",
    "ground_friction":     "surface friction coefficient (0=ice, 1=normal)",
    "throttle":            "recorded throttle action (-1=reverse, +1=forward)",
    "steering":            "recorded steering action (-1=right, +1=left)",
}


def ascii_hist(values, bins=20, width=40):
    """One-line ASCII histogram."""
    counts, edges = np.histogram(values, bins=bins)
    max_c = max(counts)
    bar = ""
    for c in counts:
        h = int(round(c / max_c * width)) if max_c > 0 else 0
        bar += "█" * h + " " * (1)
    return bar.rstrip()


def print_param(name, values):
    desc = DESCRIPTIONS.get(name, "")
    print(f"\n{'─'*60}")
    print(f"  {name}   ({desc})")
    print(f"{'─'*60}")
    print(f"  count : {len(values):,}")
    print(f"  min   : {values.min():.4f}")
    print(f"  max   : {values.max():.4f}")
    print(f"  mean  : {values.mean():.4f}")
    print(f"  std   : {values.std():.4f}")
    p = np.percentile(values, [5, 25, 50, 75, 95])
    print(f"  p5/p25/p50/p75/p95: {p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f} / {p[3]:.3f} / {p[4]:.3f}")
    print(f"  distribution:")
    print(f"    {values.min():.2f} {'':2} {ascii_hist(values)} {'':2} {values.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data_*.npz")
    ap.add_argument("--param", default=None,
                    help="Show detailed info for one parameter. "
                         "Options: " + ", ".join(ALL_COLS))
    ap.add_argument("--samples", type=int, default=10,
                    help="Number of raw sample values to print for --param")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states  = d["states"]
    actions = d["actions"]
    data = np.concatenate([states, actions], axis=1)

    base = args.data.replace(".npz", "")

    # --- save full CSV ---
    csv_path = base + ".csv"
    np.savetxt(csv_path, data, delimiter=",",
               header=",".join(ALL_COLS), comments="", fmt="%.4f")
    print(f"Saved {csv_path}  ({data.shape[0]:,} rows x {data.shape[1]} cols)")

    # --- save stats CSV ---
    stats_path = base + "_stats.csv"
    with open(stats_path, "w") as f:
        f.write("feature,min,max,mean,std,p5,p25,p50,p75,p95\n")
        for i, col in enumerate(ALL_COLS):
            v = data[:, i]
            p = np.percentile(v, [5, 25, 50, 75, 95])
            f.write(f"{col},{v.min():.4f},{v.max():.4f},{v.mean():.4f},{v.std():.4f},"
                    f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{p[3]:.4f},{p[4]:.4f}\n")
    print(f"Saved {stats_path}")

    # --- single param deep-dive ---
    if args.param:
        if args.param not in ALL_COLS:
            print(f"\nUnknown param '{args.param}'. Choose from: {', '.join(ALL_COLS)}")
            return
        idx = ALL_COLS.index(args.param)
        values = data[:, idx]
        print_param(args.param, values)
        print(f"\n  First {args.samples} raw values:")
        for i, v in enumerate(values[:args.samples]):
            print(f"    [{i:4d}]  {v:.4f}")
        return

    # --- summary table for all params ---
    print(f"\n{'feature':>22}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>7}  distribution")
    print("─" * 90)
    for i, col in enumerate(ALL_COLS):
        v = data[:, i]
        bar = ascii_hist(v, bins=15, width=25)
        print(f"{col:>22}  {v.min():>8.3f}  {v.max():>8.3f}  {v.mean():>8.3f}  {v.std():>7.3f}  {bar}")


if __name__ == "__main__":
    main()
