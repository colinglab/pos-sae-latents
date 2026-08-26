import os
import argparse
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from probe_bucket_eval import (
    compute_type_stats, bucketed_report, print_bucketed_report,
    lexicon_predict, word_type_series,
)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Majority/lexicon baseline: predicts each test token's most frequent "
                    "training-set tag for its word type, falling back to the overall "
                    "training-set majority tag for word types never seen in training (OOV). "
                    "No activations or model needed - just the UD columns already present "
                    "in the same parquet files used by the SAE-latent pipeline."
    )
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Loading train data from {args.train_file}...")
    df_train = pd.read_parquet(args.train_file)
    print(f"Loading test data from {args.test_file}...")
    df_test = pd.read_parquet(args.test_file)

    print("Building per-type majority-tag lexicon from training data...")
    type_stats = compute_type_stats(df_train, args.target_col)

    word_type_test = word_type_series(df_test).to_numpy()
    y_test = df_test[args.target_col].to_numpy()
    y_pred = lexicon_predict(word_type_test, type_stats)

    classes = sorted(set(y_test) | set(y_pred))
    report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    print(classification_report(y_test, y_pred, labels=classes, zero_division=0))

    bucket_rep = bucketed_report(y_test, y_pred, word_type_test, type_stats)
    print_bucketed_report("majority_baseline", bucket_rep)

    n_oov = int(sum(t not in type_stats["mode_tag"] for t in word_type_test))
    print(
        f"\nOOV test tokens: {n_oov} / {len(word_type_test)} "
        f"({100 * n_oov / len(word_type_test):.1f}%)"
    )

    results = {
        "label": "majority_baseline",
        "target_col": args.target_col,
        "test_classification_report": report,
        "test_confusion_matrix": cm_df,
        "bucketed_report": bucket_rep,
        "y_true": y_test,
        "y_pred": y_pred,
        "word_type_test": word_type_test,
        "type_stats": type_stats,
    }

    train_basename = os.path.splitext(os.path.basename(args.train_file))[0]
    base_name = f"{train_basename}_{args.target_col}_majority_baseline"
    results_path = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
