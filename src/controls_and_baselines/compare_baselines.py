import os
import argparse
import pickle
import pandas as pd
import warnings

from probe_bucket_eval import compute_type_stats, bucketed_report, word_type_series, BUCKET_ORDER

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Combine the SAE-latent probe, dense-layer probe, static-embedding "
                    "probe, and majority/lexicon baseline into one bucketed comparison "
                    "table (accuracy and macro-F1 per bucket per model). All four are "
                    "re-scored here from their raw predictions against the same "
                    "train-derived type statistics, so bucket definitions can't drift "
                    "between runs."
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="Training UD+SAE-latent parquet, used to (re)compute consistent "
             "ambiguity/OOV type statistics for all four models.",
    )
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument(
        "--sae_test_predictions", type=str, required=True,
        help="_test_predictions.parquet produced by probing/probe_multiclass_train_test.py.",
    )
    parser.add_argument("--sae_predicted_col", type=str, default="predicted")
    parser.add_argument(
        "--dense_results", type=str, required=True,
        help="_results.pkl from probe_dense_train_test.py run on layer-30 dense activations.",
    )
    parser.add_argument(
        "--static_results", type=str, required=True,
        help="_results.pkl from probe_dense_train_test.py run on layer-0 static activations.",
    )
    parser.add_argument(
        "--majority_results", type=str, required=True,
        help="_results.pkl from majority_baseline_train_test.py.",
    )
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Loading training data from {args.train_file} to compute type statistics...")
    df_train = pd.read_parquet(args.train_file)
    type_stats = compute_type_stats(df_train, args.target_col)

    models = {}

    print(f"Loading SAE-latent probe test predictions from {args.sae_test_predictions}...")
    df_sae = pd.read_parquet(args.sae_test_predictions)
    y_true_sae = df_sae[args.target_col].to_numpy()
    y_pred_sae = df_sae[args.sae_predicted_col].to_numpy()
    word_type_sae = word_type_series(df_sae).to_numpy()
    models["sae_latents"] = bucketed_report(y_true_sae, y_pred_sae, word_type_sae, type_stats)

    for label, path in [
        ("dense_layer30", args.dense_results),
        ("static_layer0", args.static_results),
        ("majority_baseline", args.majority_results),
    ]:
        print(f"Loading {label} results from {path}...")
        with open(path, "rb") as f:
            r = pickle.load(f)
        models[label] = bucketed_report(r["y_true"], r["y_pred"], r["word_type_test"], type_stats)

    overall_ns = {name: rep["overall"]["n"] for name, rep in models.items()}
    if len(set(overall_ns.values())) > 1:
        print(
            f"  Warning: models were evaluated on different numbers of test tokens "
            f"{overall_ns} - are all four pointing at the same test split?"
        )

    rows = []
    for bucket in BUCKET_ORDER:
        row = {"bucket": bucket, "n": models["sae_latents"][bucket]["n"]}
        for model_name, rep in models.items():
            row[f"{model_name}_accuracy"] = rep[bucket]["accuracy"]
            row[f"{model_name}_f1_macro"] = rep[bucket]["f1_macro"]
        rows.append(row)
    table = pd.DataFrame(rows)

    acc_cols = ["bucket", "n"] + [f"{m}_accuracy" for m in models]
    f1_cols = ["bucket", "n"] + [f"{m}_f1_macro" for m in models]

    acc_text = "=== Bucketed comparison (accuracy) ===\n" + table[acc_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    f1_text = "=== Bucketed comparison (macro-F1) ===\n" + table[f1_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    print("\n" + acc_text)
    print("\n" + f1_text)

    train_basename = os.path.splitext(os.path.basename(args.train_file))[0]
    base_name = f"{train_basename}_{args.target_col}_baseline_comparison"

    summary_path = os.path.join(args.output_dir, f"{base_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(acc_text + "\n\n" + f1_text + "\n")
    print(f"\nSaved text summary to {summary_path}")

    csv_path = os.path.join(args.output_dir, f"{base_name}.csv")
    table.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")

    results_path = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({"table": table, "models": models, "target_col": args.target_col}, f)
    print(f"Saved full comparison results to {results_path}")


if __name__ == "__main__":
    main()
