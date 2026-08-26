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

# Load the dictionary from the pickle file to determine (a) how many latents
# were used in the "important activations" selection, and (b) which latents
# those are, so we can control how much the random sample overlaps with them.
# The reviewer asked for a controlled overlap sweep (0/25/50/75%) rather than
# uncontrolled random sampling, since a purely random draw's overlap with the
# important set is not directly comparable across runs.
layer_n = 30
with open(f"important_act_per_pos_layer{layer_n}.pkl", "rb") as file:
    important_act_per_pos = pickle.load(file)

important_acts = set([e for l in important_act_per_pos.values() for e in l])
N_RANDOM_LATENTS = len(important_acts)

per_pos_latent_counts = {pos: len(v) for pos, v in important_act_per_pos.items()}
sum_per_pos_latents = sum(per_pos_latent_counts.values())
union_latent_count = len(important_acts)


def sample_latents_with_overlap(all_act_cols, overlap_pct, n_total, seed):
    """Sample n_total latent columns from all_act_cols such that overlap_pct%
    of them are drawn from the important-latent union (important_acts) and
    the remainder are drawn from the non-important pool. Total sample size
    is always n_total (the important-latent baseline's latent count),
    matching the original uncontrolled-random scripts' sample size."""
    overlap_pool = [c for c in all_act_cols if c in important_acts]
    non_overlap_pool = [c for c in all_act_cols if c not in important_acts]

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


def select_features(df_train, min_latent_freq, seed=42, overlap_pct=0):
    all_act_cols_train = [c for c in df_train.columns if c.startswith("act_")]

    n_to_sample = min(N_RANDOM_LATENTS, len(all_act_cols_train))
    act_cols_train, overlap_stats = sample_latents_with_overlap(
        all_act_cols_train, overlap_pct=overlap_pct, n_total=n_to_sample, seed=seed
    )

    mask = (df_train[act_cols_train] != 0).sum(axis=0) >= min_latent_freq
    kept_act_cols = [c for c in act_cols_train if mask[c]]
    return kept_act_cols, n_to_sample, overlap_stats


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
    parser = argparse.ArgumentParser(
        description="Train multiclass probe on train set, evaluate on test set, using a "
                    "random sample of latents (same count as the important-latent selection) "
                    "instead of the hand-picked important-per-POS latents. The sample is "
                    "constructed to overlap with the important-latent set at a controlled "
                    "percentage (--overlap_pct), rather than an uncontrolled random draw."
    )
    parser.add_argument("--train_file",     type=str, required=True)
    parser.add_argument("--test_file",      type=str, required=True)
    parser.add_argument("--target_col",     type=str, required=True)
    parser.add_argument("--min_latent_freq", type=int, default=1)
    parser.add_argument("--C",              type=float, default=0.1)
    parser.add_argument("--max_iter",       type=int, default=1000)
    parser.add_argument("--top_k",          type=int, default=20)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--overlap_pct",    type=int, default=0, choices=[0, 25, 50, 75],
                         help="Percentage of the sampled latents drawn from the important-"
                              "latent set (rest drawn from the non-important pool).")
    parser.add_argument("--output_dir",     type=str, default=".")
    args = parser.parse_args()

    print(f"Loading train data from {args.train_file}...")
    df_train = pd.read_parquet(args.train_file)
    print(f"Loading test data from {args.test_file}...")
    df_test = pd.read_parquet(args.test_file)

    print(f"Selecting random features ({args.overlap_pct}% overlap with important set)...")
    kept_act_cols, n_sampled, overlap_stats = select_features(
        df_train, args.min_latent_freq, seed=args.seed, overlap_pct=args.overlap_pct
    )
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
    base_name = f"{train_basename}_{args.target_col}_multiclass_random_overlap{args.overlap_pct}_seed{args.seed}"

    write_latent_union_stats(
        args.output_dir,
        base_name,
        n_sampled=n_sampled,
        n_kept=len(kept_act_cols),
        seed=args.seed,
        overlap_stats=overlap_stats,
    )

    results = {
        "clf":                   clf,
        "coefficients":          coef_df,
        "top_latents":           top_latents,
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm,
        "classes":               clf.classes_.tolist(),
        "feature_names":         kept_act_cols,
        "random_seed":           args.seed,
        "n_random_latents":      len(kept_act_cols),
        "overlap_pct":           args.overlap_pct,
        "overlap_stats":         overlap_stats,
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
