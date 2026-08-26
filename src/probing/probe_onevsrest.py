import os
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

def probe_column_classifiers(
    df,
    target_col,
    min_token_freq=50,
    min_latent_freq = 100,
    C=0.1,
    max_iter=1000,
    top_k=20,
    cv=5,
    n_jobs=-1,
):
    """
    Generalised version of POS probing: trains one classifier per unique value in target_col.
    """
    act_cols = [c for c in df.columns if c.startswith("act_")]
    ud_cols = [c for c in df.columns if c not in act_cols]



    mask = (df[act_cols] != 0).sum(axis=0) >= min_latent_freq
    kept_act_cols = df[act_cols].columns[mask].tolist()
    df_filtered = df[ud_cols + kept_act_cols]

    # Drop rare categories in target column
    counts = df_filtered[target_col].value_counts()
    valid_categories = counts[counts >= min_token_freq].index.tolist()
    df_filtered = df_filtered[df_filtered[target_col].isin(valid_categories)]

    X = df_filtered[kept_act_cols].values.astype(np.float32)
    scaler = StandardScaler(with_mean=False)
    X_sc = scaler.fit_transform(X)

    def fit_one_category(cat):
        y = (df_filtered[target_col] == cat).astype(int).values

        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=42,
        )

        cv_scores = cross_val_score(
            clf, X_sc, y, cv=cv, scoring="f1", n_jobs=1
        )

        clf.fit(X_sc, y)
        coefs = pd.Series(clf.coef_[0], index=kept_act_cols)
        top_latents = coefs[coefs > 0].sort_values(ascending=False).head(top_k)

        result = {
            "clf":          clf,
            "coefficients": coefs,
            "top_latents":  top_latents,
            "cv_f1":        cv_scores.mean(),
            "n_positive":   int(y.sum()),
        }

        log = f"{cat:15s}  n={y.sum():5d}  F1={cv_scores.mean():.3f}  non-zero coefs={(coefs != 0).sum():4d}"
        return cat, result, log

    outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_one_category)(cat) for cat in sorted(valid_categories)
    )

    results = {}
    for cat, res, log in outputs:
        results[cat] = res
        print(log)

    return results, scaler, df_filtered

def main():
    parser = argparse.ArgumentParser(description="Probe SAE latents for a specific UD column.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the latent parquet file")
    parser.add_argument("--target_col", type=str, required=True, help="UD column to probe (e.g., upos, deprel, feats)")
    parser.add_argument("--min_freq", type=int, default=50, help="Minimum frequency for categories")
    parser.add_argument("--min_latent_freq", type=int, default=100, help="Minimum frequency for latent activations")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results")

    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")
    df = pd.read_parquet(args.input_file)

    print(f"Probing target column: {args.target_col}...")
    results, scaler, df_filtered = probe_column_classifiers(
        df,
        target_col=args.target_col,
        min_token_freq=args.min_freq,
        min_latent_freq = args.min_latent_freq
    )

    # Clear naming convention: {input_basename}_{target_col}_{type}.ext
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    base_name = f"{input_basename}_{args.target_col}"

    results_path = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    scaler_path = os.path.join(args.output_dir, f"{base_name}_scaler.pkl")
    df_filtered_path = os.path.join(args.output_dir, f"{base_name}_df_filtered.parquet")

    print(f"Saving results to {results_path}...")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saving scaler to {scaler_path}...")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saving filtered dataframe to {df_filtered_path}...")
    df_filtered.to_parquet(df_filtered_path, compression="gzip")

    print("Done.")

if __name__ == "__main__":
    main()
