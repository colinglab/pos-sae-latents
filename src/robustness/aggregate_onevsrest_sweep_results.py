"""
Scans a results directory for *_metrics.json files written by
probe_onevsrest_torch.py and combines them into one long-format table (one
row per PoS tag x coverage-threshold-file x C) plus, for each
important_act_pkl, a wide pivot table (PoS tag x C, values = CV macro-F1) for
quick visual comparison - the one-vs-rest analogue of
robustness/aggregate_sweep_results.py.

Usage:
    python aggregate_onevsrest_sweep_results.py --results_dir onevsrest_sweep_results/2026-08-26_12-00-00
"""
import os
import glob
import json
import argparse
import pandas as pd


def load_runs(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.json"))):
        with open(path, "r") as f:
            run = json.load(f)
        for pos_row in run["per_pos"]:
            rows.append({
                "important_act_pkl": os.path.basename(run["important_act_pkl"]),
                "C": run["C"],
                "backend": run["backend"],
                "pos": pos_row["pos"],
                "n_positive": pos_row["n_positive"],
                "cv_f1_mean": pos_row["cv_f1_mean"],
                "cv_f1_std": pos_row["cv_f1_std"],
                "n_nonzero_coefs": pos_row["n_nonzero_coefs"],
                "fit_seconds": pos_row["fit_seconds"],
                "n_features": run["n_features"],
                "macro_f1_mean": run["macro_f1_mean"],
                "total_seconds": run["total_seconds"],
                "metrics_path": path,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                         help="Long-format output CSV path. Default: <results_dir>/onevsrest_summary_long.csv")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.results_dir, "onevsrest_summary_long.csv")

    df = load_runs(args.results_dir)
    if df.empty:
        print(f"No *_metrics.json files found in {args.results_dir}")
        return

    df = df.sort_values(["important_act_pkl", "C", "pos"])
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")

    # One run-level (important_act_pkl x C) macro-F1 summary, sorted best-first.
    run_summary = (
        df[["important_act_pkl", "C", "macro_f1_mean", "n_features", "total_seconds"]]
        .drop_duplicates()
        .sort_values("macro_f1_mean", ascending=False)
    )
    run_summary_path = os.path.join(args.results_dir, "onevsrest_run_summary.csv")
    run_summary.to_csv(run_summary_path, index=False)
    print(f"Wrote {len(run_summary)} rows to {run_summary_path}\n")
    print(run_summary.to_string(index=False))

    # A wide pivot (PoS x C) per important_act_pkl - the "full results table"
    # laid out for direct eyeballing, one per coverage threshold.
    for iap, sub in df.groupby("important_act_pkl"):
        pivot = sub.pivot(index="pos", columns="C", values="cv_f1_mean").sort_index()
        pivot_path = os.path.join(
            args.results_dir, f"onevsrest_pivot_{os.path.splitext(iap)[0]}.csv"
        )
        pivot.to_csv(pivot_path)
        print(f"\n=== {iap} (CV macro-F1 by PoS x C) ===")
        print(pivot.round(4).to_string())
        print(f"Wrote pivot -> {pivot_path}")


if __name__ == "__main__":
    main()
