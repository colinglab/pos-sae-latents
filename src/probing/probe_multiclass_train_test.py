import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


def load_important_acts(layer_n):
    with open(f"important_act_per_pos_layer{layer_n}.pkl", "rb") as file:
        important_act_per_pos = pickle.load(file)

    important_acts = set([e for l in important_act_per_pos.values() for e in l])
    per_pos_latent_counts = {pos: len(v) for pos, v in important_act_per_pos.items()}
    sum_per_pos_latents = sum(per_pos_latent_counts.values())
    union_latent_count = len(important_acts)

    return important_acts, per_pos_latent_counts, sum_per_pos_latents, union_latent_count


def write_latent_union_stats(output_dir, base_name, per_pos_latent_counts, sum_per_pos_latents, union_latent_count):
    lines = ["Important latents per POS:"]
    for pos, n in sorted(per_pos_latent_counts.items()):
        lines.append(f"  {pos}: {n}")
    lines.append(f"Sum of per-POS counts (with overlap across POS): {sum_per_pos_latents}")
    lines.append(f"Size of the union (unique latents used for filtering): {union_latent_count}")
    lines.append(f"Overlap removed by union (sum - union): {sum_per_pos_latents - union_latent_count}")
    text = "\n".join(lines)
    print(text)
    stats_path = os.path.join(output_dir, f"{base_name}_latent_union_stats.txt")
    with open(stats_path, "w") as f:
        f.write(text + "\n")
    print(f"Latent union stats written to {stats_path}")


def select_features(df_train, important_acts, min_latent_freq):
    act_cols_train = [c for c in df_train.columns if c.startswith("act_") and c in important_acts]
    mask = (df_train[act_cols_train] != 0).sum(axis=0) >= min_latent_freq
    kept_act_cols = [c for c in act_cols_train if mask[c]]
    return kept_act_cols


def train_classifier(df_train, target_col, kept_act_cols, C, max_iter, top_k):
    ud_cols = [c for c in df_train.columns if not c.startswith("act_")]
    df_filtered = df_train[ud_cols + kept_act_cols]

    X_train = df_filtered[kept_act_cols].values.astype(np.float32)
    y_train = df_filtered[target_col].values

    scaler = StandardScaler(with_mean=False)
    X_sc = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_sc, y_train)

    coef_df = pd.DataFrame(clf.coef_, index=clf.classes_, columns=kept_act_cols)
    top_latents = {
        cat: coef_df.loc[cat][coef_df.loc[cat] > 0]
                    .sort_values(ascending=False)
                    .head(top_k)
        for cat in clf.classes_
    }

    return clf, scaler, coef_df, top_latents, df_filtered


def evaluate_on_test(clf, scaler, df_test, target_col, kept_act_cols):
    ud_cols = [c for c in df_test.columns if not c.startswith("act_")]
    # Align test to training features; missing columns are filled with zero
    df_test_acts = df_test.reindex(columns=kept_act_cols, fill_value=0)
    df_test_filtered = pd.concat([df_test[ud_cols], df_test_acts], axis=1)

    X_test = df_test_filtered[kept_act_cols].values.astype(np.float32)
    y_test = df_test_filtered[target_col].values

    X_sc = scaler.transform(X_test)
    y_pred = clf.predict(X_sc)

    report = classification_report(y_test, y_pred, target_names=clf.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    print(classification_report(y_test, y_pred, target_names=clf.classes_))

    df_test_filtered["predicted"] = y_pred
    return report, cm_df, df_test_filtered


def main():
    parser = argparse.ArgumentParser(description="Train multiclass probe on train set, evaluate on test set.")
    parser.add_argument("--train_file",     type=str, required=True)
    parser.add_argument("--test_file",      type=str, required=True)
    parser.add_argument("--target_col",     type=str, required=True)
    parser.add_argument("--min_latent_freq", type=int, default=100)
    parser.add_argument("--C",              type=float, default=0.1)
    parser.add_argument("--max_iter",       type=int, default=1000)
    parser.add_argument("--top_k",          type=int, default=20)
    parser.add_argument("--layer_n",        type=int, default=30, help="Layer used to pick important_act_per_pos_layer{N}.pkl")
    parser.add_argument("--output_dir",     type=str, default=".")
    args = parser.parse_args()

    important_acts, per_pos_latent_counts, sum_per_pos_latents, union_latent_count = load_important_acts(args.layer_n)

    print(f"Loading train data from {args.train_file}...")
    df_train = pd.read_parquet(args.train_file)
    print(f"Loading test data from {args.test_file}...")
    df_test = pd.read_parquet(args.test_file)

    print("Selecting features...")
    kept_act_cols = select_features(df_train, important_acts, args.min_latent_freq)
    print(f"  {len(kept_act_cols)} features retained after filtering")

    print(f"Training classifier on full training set (target: {args.target_col})...")
    clf, scaler, coef_df, top_latents, df_train_filtered = train_classifier(
        df_train, args.target_col, kept_act_cols, args.C, args.max_iter, args.top_k
    )

    print("Evaluating on test set...")
    test_report, test_cm, df_test_predictions = evaluate_on_test(
        clf, scaler, df_test, args.target_col, kept_act_cols
    )

    train_basename = os.path.splitext(os.path.basename(args.train_file))[0]
    base_name = f"{train_basename}_{args.target_col}_multiclass"

    write_latent_union_stats(args.output_dir, base_name, per_pos_latent_counts, sum_per_pos_latents, union_latent_count)

    results = {
        "clf":                   clf,
        "coefficients":          coef_df,
        "top_latents":           top_latents,
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm,
        "classes":               clf.classes_.tolist(),
        "feature_names":         kept_act_cols,
    }

    results_path      = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    scaler_path       = os.path.join(args.output_dir, f"{base_name}_scaler.pkl")
    train_df_path     = os.path.join(args.output_dir, f"{base_name}_train_filtered.parquet")
    test_preds_path   = os.path.join(args.output_dir, f"{base_name}_test_predictions.parquet")

    print(f"Saving results to {results_path}...")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    df_train_filtered.to_parquet(train_df_path, compression="gzip")
    df_test_predictions.to_parquet(test_preds_path, compression="gzip")
    print("Done.")


if __name__ == "__main__":
    main()
