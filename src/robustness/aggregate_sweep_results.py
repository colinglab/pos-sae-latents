"""
Scans a results directory for *_metrics.json files written by
probe_multiclass_torch.py and/or probe_multiclass_train_test_torch.py,
and combines them into a single summary.csv sorted by the best available macro-F1
(test macro-F1 for train-test runs, CV macro-F1 for single-file CV runs).

Usage:
    python aggregate_sweep_results.py --results_dir sweep_results/2026-08-26_12-00-00 --output summary.csv
"""
import os
import glob
import json
import argparse
import pandas as pd


def load_metrics(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.json"))):
        with open(path, "r") as f:
            row = json.load(f)
        row["metrics_path"] = path
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate probe sweep *_metrics.json files into a summary CSV.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                         help="Output CSV path. Default: <results_dir>/summary.csv")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.results_dir, "summary.csv")

    rows = load_metrics(args.results_dir)
    if not rows:
        print(f"No *_metrics.json files found in {args.results_dir}")
        return

    df = pd.DataFrame(rows)

    # Single ranking column regardless of which script produced the row:
    # test-set macro-F1 for train-test runs, CV macro-F1 for single-file CV runs.
    test_f1 = df["test_f1_macro"] if "test_f1_macro" in df.columns else None
    cv_f1 = df["cv_f1_macro_mean"] if "cv_f1_macro_mean" in df.columns else None
    if test_f1 is not None and cv_f1 is not None:
        df["f1_macro"] = test_f1.fillna(cv_f1)
    elif test_f1 is not None:
        df["f1_macro"] = test_f1
    elif cv_f1 is not None:
        df["f1_macro"] = cv_f1

    sort_col = "f1_macro" if "f1_macro" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False)

    preferred_cols = [
        "script", "backend", "important_act_pkl", "target_col", "C",
        "n_features", "n_classes", "f1_macro", "test_f1_macro", "test_accuracy",
        "cv_f1_macro_mean", "cv_f1_macro_std", "train_f1_macro", "train_accuracy",
        "n_iter", "fit_seconds",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + \
           [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    print()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
