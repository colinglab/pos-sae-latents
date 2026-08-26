import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from probe_bucket_eval import compute_type_stats, bucketed_report, print_bucketed_report, word_type_series

warnings.filterwarnings("ignore")


def load_split(meta_path, activations_path):
    meta = pd.read_parquet(meta_path)
    X = np.load(activations_path)
    if len(meta) != X.shape[0]:
        raise ValueError(
            f"Metadata rows ({len(meta)}) != activation rows ({X.shape[0]}) - "
            f"{meta_path} and {activations_path} are not row-aligned."
        )
    return meta, X


def main():
    parser = argparse.ArgumentParser(
        description="Train/test L1 multiclass probe on raw dense LLM activations (no SAE). "
                    "Used for both the layer-30 dense residual-stream baseline and the "
                    "layer-0 static-embedding baseline: point --*_activations at whichever "
                    "layer's .npy file produced by extract_dense_activations.py. Same probe "
                    "architecture as probing/probe_multiclass_train_test.py, just with "
                    "different features - not tuned to make either baseline look better."
    )
    parser.add_argument("--train_meta", type=str, required=True)
    parser.add_argument("--train_activations", type=str, required=True)
    parser.add_argument("--test_meta", type=str, required=True)
    parser.add_argument("--test_activations", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument(
        "--label", type=str, required=True,
        help="Short tag identifying this run, e.g. 'dense_layer30' or 'static_layer0'. "
             "Used in output filenames only.",
    )
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Loading train activations from {args.train_activations}...")
    df_train, X_train = load_split(args.train_meta, args.train_activations)
    print(f"Loading test activations from {args.test_activations}...")
    df_test, X_test = load_split(args.test_meta, args.test_activations)

    y_train = df_train[args.target_col].to_numpy()
    y_test = df_test[args.target_col].to_numpy()

    print("Computing train-set type statistics for bucketed evaluation...")
    type_stats = compute_type_stats(df_train, args.target_col)

    # Dense/static residual-stream vectors are not sparse and not
    # non-negative like SAE activations, so (unlike the SAE probe scripts)
    # we mean-center: StandardScaler default (with_mean=True) is the
    # standard choice for continuous dense features.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train.astype(np.float32))
    X_test_sc = scaler.transform(X_test.astype(np.float32))

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=args.C,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=42,
        n_jobs=-1,
    )

    print(f"Training L1 probe on {X_train_sc.shape[1]}-dim features ({args.label})...")
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    report = classification_report(y_test, y_pred, target_names=clf.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    print(classification_report(y_test, y_pred, target_names=clf.classes_))

    word_type_test = word_type_series(df_test).to_numpy()
    bucket_rep = bucketed_report(y_test, y_pred, word_type_test, type_stats)
    print_bucketed_report(args.label, bucket_rep)

    results = {
        "label": args.label,
        "target_col": args.target_col,
        "n_features": X_train_sc.shape[1],
        "classes": clf.classes_.tolist(),
        "test_classification_report": report,
        "test_confusion_matrix": cm_df,
        "bucketed_report": bucket_rep,
        "y_true": y_test,
        "y_pred": y_pred,
        "word_type_test": word_type_test,
        "clf": clf,
    }

    base_name = f"{args.label}_{args.target_col}_probe"
    results_path = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
