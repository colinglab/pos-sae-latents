import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Word type = lowercased surface form. Every occurrence of a type gets the
# same fixed control label, regardless of context.
WORD_TYPE_SOURCE_COL = "form"


def build_control_labels(df_train, df_test, target_col, seed, word_type_col=WORD_TYPE_SOURCE_COL):
    """
    Hewitt & Liang (2019) control task: assign each word type (seen in train
    or test) one fixed label, drawn i.i.d. from the empirical (token-level)
    distribution of `target_col` *in the training set*. Every occurrence of
    that type - train or test - gets that label, regardless of context.
    """
    word_type_train = df_train[word_type_col].astype(str).str.lower()
    word_type_test = df_test[word_type_col].astype(str).str.lower()

    counts = df_train[target_col].value_counts(normalize=True)
    classes = counts.index.to_numpy()
    probs = counts.to_numpy()
    probs = probs / probs.sum()  # guard against float rounding for np.random.choice

    unique_types = np.sort(pd.unique(pd.concat([word_type_train, word_type_test], ignore_index=True)))
    rng = np.random.RandomState(seed)
    drawn = rng.choice(classes, size=len(unique_types), p=probs)
    type_to_control_label = dict(zip(unique_types, drawn))

    control_y_train = word_type_train.map(type_to_control_label).to_numpy()
    control_y_test = word_type_test.map(type_to_control_label).to_numpy()
    return control_y_train, control_y_test, type_to_control_label


def fit_and_eval(clf_template, X_train, y_train, X_test, y_test):
    scaler = StandardScaler(with_mean=False)
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = clone(clf_template)
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    report = classification_report(y_test, y_pred, target_names=clf.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
    return clf, scaler, report, cm_df


def write_control_task_summary(output_dir, base_name, info):
    lines = [
        "Hewitt & Liang (2019) control task - selectivity report (train/test)",
        f"Target column: {info['target_col']}",
        f"Word-type key: lowercased '{info['word_type_col']}'",
        f"Unique word types (train + test): {info['n_word_types']}",
        f"Control-label sampling: empirical marginal distribution of {info['target_col']} "
        f"in the training set (i.i.d. draw per type)",
        f"Random seed: {info['seed']}",
        "",
        f"Real probe    - test accuracy: {info['real_test_accuracy']:.4f}, "
        f"macro-F1: {info['real_test_f1_macro']:.4f}",
        f"Control probe - test accuracy: {info['control_test_accuracy']:.4f}, "
        f"macro-F1: {info['control_test_f1_macro']:.4f}",
        "",
        f"Selectivity (accuracy):  {info['selectivity_accuracy']:.4f}",
        f"Selectivity (macro-F1):  {info['selectivity_f1_macro']:.4f}",
    ]
    text = "\n".join(lines)
    print(text)
    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(text + "\n")
    print(f"Control task summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hewitt & Liang control task for the multiclass SAE-latent probe "
                    "(train/test variant): same features/hyperparameters as an existing "
                    "real probe, retrained to predict a fixed random per-word-type label. "
                    "Reports selectivity = real accuracy/F1 - control accuracy/F1 on the "
                    "held-out test set."
    )
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument(
        "--real_results_path", type=str, required=True,
        help="Path to the _results.pkl produced by probing/probe_multiclass_train_test.py "
             "(or its random-latent counterpart) for this exact train/test files & target column.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Loading train data from {args.train_file}...")
    df_train = pd.read_parquet(args.train_file)
    print(f"Loading test data from {args.test_file}...")
    df_test = pd.read_parquet(args.test_file)

    print(f"Loading real-probe artifact from {args.real_results_path}...")
    with open(args.real_results_path, "rb") as f:
        real_results = pickle.load(f)

    kept_act_cols = real_results["feature_names"]
    missing_train = [c for c in kept_act_cols if c not in df_train.columns]
    missing_test = [c for c in kept_act_cols if c not in df_test.columns]
    if missing_train or missing_test:
        raise ValueError(
            "Feature mismatch between --real_results_path and the given train/test files - "
            "is --real_results_path for the wrong files/target column? "
            f"Missing in train: {missing_train[:5]}, missing in test: {missing_test[:5]}"
        )
    if WORD_TYPE_SOURCE_COL not in df_train.columns or WORD_TYPE_SOURCE_COL not in df_test.columns:
        raise ValueError(f"Expected a '{WORD_TYPE_SOURCE_COL}' column in both train and test files.")

    real_report = real_results["test_classification_report"]
    real_test_accuracy = real_report["accuracy"]
    real_test_f1_macro = real_report["macro avg"]["f1-score"]

    print("Building control task labels (fixed random label per word type, drawn from train distribution)...")
    control_y_train, control_y_test, type_to_control_label = build_control_labels(
        df_train, df_test, args.target_col, seed=args.seed
    )

    X_train = df_train[kept_act_cols].values.astype(np.float32)
    X_test = df_test.reindex(columns=kept_act_cols, fill_value=0).values.astype(np.float32)

    print("Training control probe on train set, evaluating on held-out test set...")
    clf_control, scaler_control, control_report, control_cm = fit_and_eval(
        real_results["clf"], X_train, control_y_train, X_test, control_y_test
    )

    control_test_accuracy = control_report["accuracy"]
    control_test_f1_macro = control_report["macro avg"]["f1-score"]

    selectivity_accuracy = real_test_accuracy - control_test_accuracy
    selectivity_f1_macro = real_test_f1_macro - control_test_f1_macro

    results = {
        "target_col": args.target_col,
        "word_type_col": WORD_TYPE_SOURCE_COL,
        "seed": args.seed,
        "n_word_types": len(type_to_control_label),
        "feature_names": kept_act_cols,
        "real_test_accuracy": real_test_accuracy,
        "real_test_f1_macro": real_test_f1_macro,
        "control_test_classification_report": control_report,
        "control_test_confusion_matrix": control_cm,
        "control_test_accuracy": control_test_accuracy,
        "control_test_f1_macro": control_test_f1_macro,
        "selectivity_accuracy": selectivity_accuracy,
        "selectivity_f1_macro": selectivity_f1_macro,
        "type_to_control_label": type_to_control_label,
        "clf_control": clf_control,
    }

    train_basename = os.path.splitext(os.path.basename(args.train_file))[0]
    base_name = f"{train_basename}_{args.target_col}_control_task_seed{args.seed}"

    write_control_task_summary(args.output_dir, base_name, results)

    results_path = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    mapping_path = os.path.join(args.output_dir, f"{base_name}_type_to_control_label.pkl")

    print(f"Saving results to {results_path}...")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    with open(mapping_path, "wb") as f:
        pickle.dump(type_to_control_label, f)
    print("Done.")


if __name__ == "__main__":
    main()
