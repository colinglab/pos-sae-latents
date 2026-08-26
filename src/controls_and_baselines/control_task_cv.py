import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Word type = lowercased surface form. Every occurrence of a type gets the
# same fixed control label, regardless of context.
WORD_TYPE_SOURCE_COL = "form"


def build_control_labels(df, target_col, seed, word_type_col=WORD_TYPE_SOURCE_COL):
    """
    Hewitt & Liang (2019) control task: assign each word type one fixed
    label, drawn i.i.d. from the empirical (token-level) distribution of
    `target_col`. A probe that predicts this behavior well is recovering
    type-level identity, not linguistic structure.
    """
    word_type = df[word_type_col].astype(str).str.lower()

    counts = df[target_col].value_counts(normalize=True)
    classes = counts.index.to_numpy()
    probs = counts.to_numpy()
    probs = probs / probs.sum()  # guard against float rounding for np.random.choice

    unique_types = np.sort(word_type.unique())
    rng = np.random.RandomState(seed)
    drawn = rng.choice(classes, size=len(unique_types), p=probs)
    type_to_control_label = dict(zip(unique_types, drawn))

    control_y = word_type.map(type_to_control_label).to_numpy()
    return control_y, type_to_control_label


def held_out_scores(clf_template, X, y, cv, seed=42):
    min_class_count = int(pd.Series(y).value_counts().min())
    cv_eff = int(min(cv, min_class_count))
    if cv_eff < cv:
        print(
            f"  Warning: reducing CV folds from {cv} to {cv_eff} "
            f"(smallest class has only {min_class_count} examples)."
        )
    skf = StratifiedKFold(n_splits=cv_eff, shuffle=True, random_state=seed)
    scores = cross_validate(
        clone(clf_template), X, y,
        cv=skf,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1,
    )
    return {
        "cv_folds": cv_eff,
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "f1_macro_std": float(scores["test_f1_macro"].std()),
    }


def write_control_task_summary(output_dir, base_name, info):
    lines = [
        "Hewitt & Liang (2019) control task - selectivity report",
        f"Target column: {info['target_col']}",
        f"Word-type key: lowercased '{info['word_type_col']}'",
        f"Unique word types: {info['n_word_types']}",
        f"Tokens probed: {info['n_tokens']}",
        f"Control-label sampling: empirical marginal distribution of {info['target_col']} "
        f"(i.i.d. draw per type)",
        f"Random seed: {info['seed']}",
        "",
        f"Real probe    - held-out accuracy: {info['real']['accuracy_mean']:.4f} "
        f"± {info['real']['accuracy_std']:.4f} (cv={info['real']['cv_folds']})",
        f"Real probe    - held-out macro-F1: {info['real']['f1_macro_mean']:.4f} "
        f"± {info['real']['f1_macro_std']:.4f}",
        f"Control probe - held-out accuracy: {info['control']['accuracy_mean']:.4f} "
        f"± {info['control']['accuracy_std']:.4f} (cv={info['control']['cv_folds']})",
        f"Control probe - held-out macro-F1: {info['control']['f1_macro_mean']:.4f} "
        f"± {info['control']['f1_macro_std']:.4f}",
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
                    "(CV variant): same features/hyperparameters as an existing real "
                    "probe, retrained to predict a fixed random per-word-type label. "
                    "Reports selectivity = real accuracy/F1 - control accuracy/F1."
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument(
        "--real_results_path", type=str, required=True,
        help="Path to the _results.pkl produced by probing/probe_multiclass_cv.py "
             "(or its random-latent counterpart) for this exact input_file/target_col.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")
    df = pd.read_parquet(args.input_file)

    print(f"Loading real-probe artifact from {args.real_results_path}...")
    with open(args.real_results_path, "rb") as f:
        real_results = pickle.load(f)

    kept_act_cols = real_results["feature_names"]
    missing = [c for c in kept_act_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} feature(s) from the real-probe artifact are missing in "
            f"{args.input_file} (e.g. {missing[:5]}); is --real_results_path for the "
            f"wrong input file/target column?"
        )
    if WORD_TYPE_SOURCE_COL not in df.columns:
        raise ValueError(f"Expected a '{WORD_TYPE_SOURCE_COL}' column in {args.input_file}.")

    X = df[kept_act_cols].values.astype(np.float32)
    # .to_numpy() (not .values) avoids an Arrow-backed string array that
    # breaks joblib's multiprocess row indexing under pandas' string dtype.
    y_real = df[args.target_col].to_numpy()

    scaler = StandardScaler(with_mean=False)
    X_sc = scaler.fit_transform(X)

    print("Recomputing held-out CV scores for the real probe (same features/hyperparameters)...")
    real_scores = held_out_scores(real_results["clf"], X_sc, y_real, args.cv, seed=args.seed)
    stored_f1 = real_results.get("cv_f1_macro")
    if stored_f1 is not None and abs(stored_f1 - real_scores["f1_macro_mean"]) > 0.02:
        print(
            f"  Warning: recomputed real macro-F1 ({real_scores['f1_macro_mean']:.3f}) differs "
            f"from the value stored in the artifact ({stored_f1:.3f}) by more than 0.02 - "
            f"double-check --real_results_path matches --input_file/--target_col."
        )

    print("Building control task labels (fixed random label per word type)...")
    control_y, type_to_control_label = build_control_labels(df, args.target_col, seed=args.seed)

    print("Training control probe (same features/hyperparameters as the real probe)...")
    control_scores = held_out_scores(real_results["clf"], X_sc, control_y, args.cv, seed=args.seed)

    selectivity_accuracy = real_scores["accuracy_mean"] - control_scores["accuracy_mean"]
    selectivity_f1_macro = real_scores["f1_macro_mean"] - control_scores["f1_macro_mean"]

    # Full-data fits, kept for coefficients / reuse (not used for the reported metrics above).
    clf_real = clone(real_results["clf"]).fit(X_sc, y_real)
    clf_control = clone(real_results["clf"]).fit(X_sc, control_y)
    coef_df_real = pd.DataFrame(clf_real.coef_, index=clf_real.classes_, columns=kept_act_cols)
    coef_df_control = pd.DataFrame(clf_control.coef_, index=clf_control.classes_, columns=kept_act_cols)

    results = {
        "target_col": args.target_col,
        "word_type_col": WORD_TYPE_SOURCE_COL,
        "seed": args.seed,
        "n_word_types": len(type_to_control_label),
        "n_tokens": len(df),
        "feature_names": kept_act_cols,
        "real": real_scores,
        "control": control_scores,
        "selectivity_accuracy": selectivity_accuracy,
        "selectivity_f1_macro": selectivity_f1_macro,
        "type_to_control_label": type_to_control_label,
        "clf_real_refit": clf_real,
        "clf_control": clf_control,
        "coefficients_real": coef_df_real,
        "coefficients_control": coef_df_control,
    }

    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    base_name = f"{input_basename}_{args.target_col}_control_task_seed{args.seed}"

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
