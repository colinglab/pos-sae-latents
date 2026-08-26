import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


def load_important_acts(layer_n):
    with open(f"important_act_per_pos_layer{layer_n}.pkl", "rb") as file:
        important_act_per_pos = pickle.load(file)

    important_acts = list(set([e for l in list(important_act_per_pos.values()) for e in l]))
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


def probe_column_multiclass(
    df,
    target_col,
    important_acts,
    min_token_freq=50,
    min_latent_freq=100,
    C=0.1,
    max_iter=1000,
    top_k=20,
    cv=5,
):
    #act_cols = [c for c in df.columns if c.startswith("act_")]
    act_cols = [c for c in df.columns if c.startswith("act_") and c in important_acts]
    ud_cols = [c for c in df.columns if c not in act_cols]

    mask = (df[act_cols] != 0).sum(axis=0) >= min_latent_freq
    kept_act_cols = df[act_cols].columns[mask].tolist()
    df_filtered = df[ud_cols + kept_act_cols]

    #counts = df_filtered[target_col].value_counts()
    #valid_categories = counts[counts >= min_token_freq].index.tolist()
    #df_filtered = df_filtered[df_filtered[target_col].isin(valid_categories)]

    X = df_filtered[kept_act_cols].values.astype(np.float32)
    y = df_filtered[target_col].to_numpy()

    scaler = StandardScaler(with_mean=False)
    X_sc = scaler.fit_transform(X)

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",          # saga supports L1 + multinomial (liblinear does OVR only)
        multi_class="multinomial",
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(
        clf, X_sc, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
    )

    clf.fit(X_sc, y)

    # coef_ shape: (n_classes, n_features)
    coef_df = pd.DataFrame(clf.coef_, index=clf.classes_, columns=kept_act_cols)

    top_latents = {
        cat: coef_df.loc[cat][coef_df.loc[cat] > 0]
                    .sort_values(ascending=False)
                    .head(top_k)
        for cat in clf.classes_
    }

    y_pred = clf.predict(X_sc)
    report = classification_report(y, y_pred, target_names=clf.classes_, output_dict=True)
    cm = confusion_matrix(y, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    # Console summary
    print(f"CV macro-F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(classification_report(y, y_pred, target_names=clf.classes_))

    results = {
        "clf":                    clf,
        "coefficients":           coef_df,          # DataFrame (n_classes × n_features)
        "top_latents":            top_latents,       # dict[class -> Series]
        "cv_f1_macro":            cv_scores.mean(),
        "cv_f1_macro_std":        cv_scores.std(),
        "classification_report":  report,
        "confusion_matrix":       cm_df,
        "classes":                clf.classes_.tolist(),
        "feature_names":          kept_act_cols,
    }

    return results, scaler, df_filtered


def main():
    parser = argparse.ArgumentParser(description="Multiclass probe of SAE latents for a UD column.")
    parser.add_argument("--input_file",      type=str, required=True)
    parser.add_argument("--target_col",      type=str, required=True)
    parser.add_argument("--min_freq",        type=int, default=50)
    parser.add_argument("--min_latent_freq", type=int, default=100)
    parser.add_argument("--layer_n",         type=int, default=30, help="Layer used to pick important_act_per_pos_layer{N}.pkl")
    parser.add_argument("--output_dir",      type=str, default=".")
    args = parser.parse_args()

    important_acts, per_pos_latent_counts, sum_per_pos_latents, union_latent_count = load_important_acts(args.layer_n)

    print(f"Loading data from {args.input_file}...")
    df = pd.read_parquet(args.input_file)

    print(f"Probing target column (multiclass): {args.target_col}...")
    results, scaler, df_filtered = probe_column_multiclass(
        df,
        target_col=args.target_col,
        important_acts=important_acts,
        min_token_freq=args.min_freq,
        min_latent_freq=args.min_latent_freq,
    )

    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    base_name = f"{input_basename}_{args.target_col}_multiclass"

    write_latent_union_stats(args.output_dir, base_name, per_pos_latent_counts, sum_per_pos_latents, union_latent_count)

    results_path    = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    scaler_path     = os.path.join(args.output_dir, f"{base_name}_scaler.pkl")
    df_filtered_path = os.path.join(args.output_dir, f"{base_name}_df_filtered.parquet")

    print(f"Saving results to {results_path}...")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    df_filtered.to_parquet(df_filtered_path, compression="gzip")
    print("Done.")


if __name__ == "__main__":
    main()