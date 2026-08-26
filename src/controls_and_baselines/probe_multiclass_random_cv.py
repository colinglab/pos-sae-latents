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

# Load the dictionary from the pickle file to determine (a) how many latents
# were used in the "important activations" selection, and (b) which latents
# those are, so we can control how much the random sample overlaps with them.
# The reviewer asked for a controlled overlap sweep (0/25/50/75%) rather than
# uncontrolled random sampling, since a purely random draw's overlap with the
# important set is not directly comparable across runs.
layer_n = 30
with open(f"important_act_per_pos_layer{layer_n}.pkl", "rb") as file:
    important_act_per_pos = pickle.load(file)

important_acts = list(set([e for l in list(important_act_per_pos.values()) for e in l]))
important_acts_set = set(important_acts)
N_RANDOM_LATENTS = len(important_acts)

per_pos_latent_counts = {pos: len(v) for pos, v in important_act_per_pos.items()}
sum_per_pos_latents = sum(per_pos_latent_counts.values())
union_latent_count = len(important_acts)


def sample_latents_with_overlap(all_act_cols, overlap_pct, n_total, seed):
    """Sample n_total latent columns from all_act_cols such that overlap_pct%
    of them are drawn from the important-latent union (important_acts_set)
    and the remainder are drawn from the non-important pool. Total sample
    size is always n_total (the important-latent baseline's latent count),
    matching the original uncontrolled-random scripts' sample size."""
    overlap_pool = [c for c in all_act_cols if c in important_acts_set]
    non_overlap_pool = [c for c in all_act_cols if c not in important_acts_set]

    n_overlap_target = int(round(overlap_pct / 100.0 * n_total))
    n_overlap = min(n_overlap_target, len(overlap_pool))
    if n_overlap < n_overlap_target:
        print(
            f"WARNING: requested {n_overlap_target} overlap latents but only "
            f"{len(overlap_pool)} important latents are present as columns; using {n_overlap}."
        )

    n_non_overlap_target = n_total - n_overlap
    n_non_overlap = min(n_non_overlap_target, len(non_overlap_pool))
    if n_non_overlap < n_non_overlap_target:
        print(
            f"WARNING: requested {n_non_overlap_target} non-overlap latents but only "
            f"{len(non_overlap_pool)} non-important latents are available; using {n_non_overlap}."
        )

    rng = np.random.RandomState(seed)
    sampled_overlap = list(rng.choice(overlap_pool, size=n_overlap, replace=False)) if n_overlap > 0 else []
    sampled_non_overlap = list(rng.choice(non_overlap_pool, size=n_non_overlap, replace=False)) if n_non_overlap > 0 else []

    sampled = sampled_overlap + sampled_non_overlap
    rng.shuffle(sampled)

    achieved_overlap_pct = 100.0 * n_overlap / len(sampled) if sampled else 0.0
    print(
        f"Sampled {len(sampled)} latents: {n_overlap} from the important set, "
        f"{n_non_overlap} from the non-important pool "
        f"(requested {overlap_pct}% overlap, achieved {achieved_overlap_pct:.1f}%)."
    )

    stats = {
        "requested_overlap_pct": overlap_pct,
        "achieved_overlap_pct": achieved_overlap_pct,
        "n_overlap": n_overlap,
        "n_non_overlap": n_non_overlap,
    }
    return sampled, stats


def write_latent_union_stats(
    output_dir, base_name, n_sampled=None, n_kept=None, seed=None, overlap_stats=None
):
    lines = ["Important latents per POS (from the important-latent baseline pickle):"]
    for pos, n in sorted(per_pos_latent_counts.items()):
        lines.append(f"  {pos}: {n}")
    lines.append(f"Sum of per-POS counts (with overlap across POS): {sum_per_pos_latents}")
    lines.append(f"Size of the union (unique latents, = N_RANDOM_LATENTS): {union_latent_count}")
    lines.append(f"Overlap removed by union (sum - union): {sum_per_pos_latents - union_latent_count}")
    if seed is not None:
        lines.append(f"Random seed used for this run: {seed}")
    if overlap_stats is not None:
        lines.append(f"Requested overlap with important set: {overlap_stats['requested_overlap_pct']}%")
        lines.append(f"Achieved overlap with important set: {overlap_stats['achieved_overlap_pct']:.1f}%")
        lines.append(f"Latents sampled from important set: {overlap_stats['n_overlap']}")
        lines.append(f"Latents sampled from non-important pool: {overlap_stats['n_non_overlap']}")
    if n_sampled is not None:
        lines.append(f"Random latents sampled from the full activating pool: {n_sampled}")
    if n_kept is not None:
        lines.append(f"Random latents kept after min_latent_freq filtering: {n_kept}")
    text = "\n".join(lines)
    print(text)
    stats_path = os.path.join(output_dir, f"{base_name}_latent_union_stats.txt")
    with open(stats_path, "w") as f:
        f.write(text + "\n")
    print(f"Latent union stats written to {stats_path}")


def probe_column_multiclass(
    df,
    target_col,
    min_token_freq=50,
    min_latent_freq=100,
    C=0.1,
    max_iter=2000,
    top_k=20,
    cv=5,
    seed=42,
    overlap_pct=0,
):
    all_act_cols = [c for c in df.columns if c.startswith("act_")]

    n_to_sample = min(N_RANDOM_LATENTS, len(all_act_cols))
    act_cols, overlap_stats = sample_latents_with_overlap(
        all_act_cols, overlap_pct=overlap_pct, n_total=n_to_sample, seed=seed
    )

    ud_cols = [c for c in df.columns if c not in act_cols]

    mask = (df[act_cols] != 0).sum(axis=0) >= min_latent_freq
    kept_act_cols = df[act_cols].columns[mask].tolist()
    df_filtered = df[ud_cols + kept_act_cols]

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
        "random_seed":            seed,
        "n_random_latents":       n_to_sample,
        "overlap_pct":            overlap_pct,
        "overlap_stats":          overlap_stats,
    }

    return results, scaler, df_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Multiclass probe of SAE latents for a UD column, using a random "
                    "sample of latents (same count as the important-latent selection) "
                    "instead of the hand-picked important-per-POS latents. The sample is "
                    "constructed to overlap with the important-latent set at a controlled "
                    "percentage (--overlap_pct), rather than an uncontrolled random draw."
    )
    parser.add_argument("--input_file",      type=str, required=True)
    parser.add_argument("--target_col",      type=str, required=True)
    parser.add_argument("--min_freq",        type=int, default=1)
    parser.add_argument("--min_latent_freq", type=int, default=1)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--overlap_pct",     type=int, default=0, choices=[0, 25, 50, 75],
                         help="Percentage of the sampled latents drawn from the important-"
                              "latent set (rest drawn from the non-important pool).")
    parser.add_argument("--output_dir",      type=str, default=".")
    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")
    df = pd.read_parquet(args.input_file)

    print(f"Probing target column (multiclass, random latents, {args.overlap_pct}% overlap): {args.target_col}...")
    results, scaler, df_filtered = probe_column_multiclass(
        df,
        target_col=args.target_col,
        min_token_freq=args.min_freq,
        min_latent_freq=args.min_latent_freq,
        seed=args.seed,
        overlap_pct=args.overlap_pct,
    )

    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    base_name = f"{input_basename}_{args.target_col}_multiclass_random_overlap{args.overlap_pct}_seed{args.seed}"

    write_latent_union_stats(
        args.output_dir,
        base_name,
        n_sampled=results["n_random_latents"],
        n_kept=len(results["feature_names"]),
        overlap_stats=results["overlap_stats"],
        seed=args.seed,
    )

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
