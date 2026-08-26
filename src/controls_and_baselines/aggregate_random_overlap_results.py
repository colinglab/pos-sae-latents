#!/usr/bin/env python3
"""
Aggregates the random-latent overlap baseline runs (Table 3: accuracy/macro-F1
across overlap levels) produced by probe_multiclass_random_cv.py and
probe_multiclass_random_train_test.py (driven by
scripts/run_random_overlap_baseline.sh) into two summary tables.

Usage:
    python aggregate_random_overlap_results.py \\
        --cv_dir probing-results-random-overlap \\
        --train_test_dir probing-results-train-test-random-overlap
"""
import os
import glob
import pickle
import argparse
import pandas as pd


def load_cv_results(results_dir):
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/*_results.pkl")):
        with open(path, "rb") as f:
            r = pickle.load(f)
        rows.append({
            "overlap_pct": r["overlap_pct"],
            "seed": r["random_seed"],
            "accuracy_in_sample": r["classification_report"]["accuracy"],  # NOT held-out
            "f1_macro_cv": r["cv_f1_macro"],                               # held-out
            "f1_macro_cv_std": r["cv_f1_macro_std"],
            "file": path,
        })
    return pd.DataFrame(rows).sort_values("overlap_pct")


def load_train_test_results(results_dir):
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/*_results.pkl")):
        with open(path, "rb") as f:
            r = pickle.load(f)
        report = r["test_classification_report"]
        rows.append({
            "overlap_pct": r["overlap_pct"],
            "seed": r["random_seed"],
            "accuracy_test": report["accuracy"],
            "f1_macro_test": report["macro avg"]["f1-score"],
            "file": path,
        })
    return pd.DataFrame(rows).sort_values("overlap_pct")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv_dir", type=str, default="probing-results-random-overlap")
    parser.add_argument("--train_test_dir", type=str, default="probing-results-train-test-random-overlap")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Cross-validation ===")
    cv_df = load_cv_results(args.cv_dir)
    print(cv_df.to_string(index=False))

    print("\n=== Train/test ===")
    tt_df = load_train_test_results(args.train_test_dir)
    print(tt_df.to_string(index=False))

    cv_df.to_csv(os.path.join(args.output_dir, "random_overlap_cv_summary.csv"), index=False)
    tt_df.to_csv(os.path.join(args.output_dir, "random_overlap_train_test_summary.csv"), index=False)
    print(f"\nSaved summaries to {args.output_dir}/")


if __name__ == "__main__":
    main()
