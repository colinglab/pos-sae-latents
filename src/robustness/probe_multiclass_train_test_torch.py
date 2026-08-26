import os
import time
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


class TorchL1LogisticRegression:
    """
    Minimal sklearn-compatible (fit/predict/classes_/coef_/intercept_) drop-in
    for LogisticRegression(penalty="l1", solver="saga", class_weight="balanced"),
    fit with full-batch FISTA (proximal gradient + Nesterov acceleration +
    backtracking line search) in PyTorch instead of sklearn's saga solver.

    Solves the *same objective* sklearn's saga solver does for multinomial L1:
        minimize_{W,b}  C * sum_i sample_weight_i * CE_i(W, b)  +  ||W||_1
    (intercept b unpenalized; sample_weight from class_weight="balanced" the
    same way sklearn derives it: n_samples / (n_classes * class_count), which
    sums to n_samples overall). Only the *optimizer* changes, not what's being
    optimized, so this should recover comparable coefficients/accuracy to the
    sklearn path - just much faster.

    Kept as a per-script copy rather than a shared import, so each probe
    script stays self-contained.
    """

    def __init__(self, C=1.0, max_iter=1000, tol=1e-6, class_weight="balanced",
                 device=None, verbose=50, random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(
                "backend='torch' requires PyTorch. Install it, or rerun with "
                "--backend sklearn to use the (much slower) original solver."
            ) from e

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)

        classes, y_idx = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(classes)

        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
            w_per_class = n_samples / (n_classes * counts)
            sample_weight = w_per_class[y_idx]
        else:
            sample_weight = np.ones(n_samples)

        X_t = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y_idx, dtype=torch.long, device=device)
        sw_t = torch.as_tensor(sample_weight, dtype=torch.float32, device=device)

        W = torch.zeros(n_features, n_classes, device=device)
        b = torch.zeros(n_classes, device=device)

        def smooth_loss(W_, b_):
            logits = X_t @ W_ + b_
            ce = F.cross_entropy(logits, y_t, reduction="none")
            return self.C * (sw_t * ce).sum()

        def soft_threshold(v, thresh):
            return torch.sign(v) * torch.clamp(v.abs() - thresh, min=0.0)

        z_W, z_b = W.clone(), b.clone()
        theta = 1.0
        t = 1.0  # step size, adapted by backtracking each iteration
        prev_obj = None
        it = 0

        t0 = time.time()
        for it in range(self.max_iter):
            z_W.requires_grad_(True)
            z_b.requires_grad_(True)
            loss_z = smooth_loss(z_W, z_b)
            gW, gb = torch.autograd.grad(loss_z, [z_W, z_b])
            loss_z_val = loss_z.item()
            z_W = z_W.detach()
            z_b = z_b.detach()

            # Backtracking line search: shrink step t until the smooth loss
            # at the proximal point satisfies its quadratic upper bound - the
            # non-smooth L1 term cancels out of this check exactly (Beck &
            # Teboulle 2009), so checking the smooth loss alone is correct.
            while True:
                W_new = soft_threshold(z_W - t * gW, t)
                b_new = z_b - t * gb
                with torch.no_grad():
                    loss_new = smooth_loss(W_new, b_new).item()
                diffW = W_new - z_W
                diffb = b_new - z_b
                quad = (
                    loss_z_val
                    + (gW * diffW).sum().item()
                    + (gb * diffb).sum().item()
                    + (diffW.pow(2).sum().item() + diffb.pow(2).sum().item()) / (2 * t)
                )
                if loss_new <= quad + 1e-6 * max(1.0, abs(quad)):
                    break
                t *= 0.5
                if t < 1e-14:
                    break

            theta_new = (1 + (1 + 4 * theta ** 2) ** 0.5) / 2
            momentum = (theta - 1) / theta_new
            z_W = W_new + momentum * (W_new - W)
            z_b = b_new + momentum * (b_new - b)

            obj = loss_new + W_new.abs().sum().item()
            if self.verbose and (it % max(1, self.verbose) == 0 or it == self.max_iter - 1):
                print(f"  [torch-fista] iter {it:4d}  obj={obj:.6f}  step={t:.2e}  "
                      f"elapsed={time.time() - t0:.1f}s")

            W, b = W_new, b_new
            theta = theta_new
            t = min(t * 1.5, 1.0)

            if prev_obj is not None:
                rel_change = abs(prev_obj - obj) / max(1.0, abs(prev_obj))
                if rel_change < self.tol:
                    if self.verbose:
                        print(f"  [torch-fista] converged at iter {it} "
                              f"(rel change {rel_change:.2e} < tol {self.tol:.2e})")
                    prev_obj = obj
                    break
            prev_obj = obj

        self.coef_ = W.T.cpu().numpy()  # (n_classes, n_features), sklearn convention
        self.intercept_ = b.cpu().numpy()
        self.classes_ = classes
        self.n_iter_ = it + 1
        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.coef_.T + self.intercept_

    def predict(self, X):
        logits = self.decision_function(X)
        return self.classes_[np.argmax(logits, axis=1)]


def select_features(df_train, important_acts, min_latent_freq):
    act_cols_train = [c for c in df_train.columns if c.startswith("act_") and c in important_acts]
    mask = (df_train[act_cols_train] != 0).sum(axis=0) >= min_latent_freq
    kept_act_cols = [c for c in act_cols_train if mask[c]]
    return kept_act_cols


def train_classifier(df_train, target_col, kept_act_cols, C, max_iter, top_k,
                      backend, tol, device, torch_verbose):
    ud_cols = [c for c in df_train.columns if not c.startswith("act_")]
    df_filtered = df_train[ud_cols + kept_act_cols]

    X_train = df_filtered[kept_act_cols].values.astype(np.float32)
    y_train = df_filtered[target_col].values

    # SAE latents are sparse/non-negative, so (unlike dense residual-stream
    # features) we don't mean-center - with_mean=False matches the original
    # sklearn script.
    scaler = StandardScaler(with_mean=False)
    X_sc = scaler.fit_transform(X_train)

    if backend == "torch":
        clf = TorchL1LogisticRegression(
            C=C, max_iter=max_iter, tol=tol, class_weight="balanced",
            device=device, verbose=torch_verbose, random_state=42,
        )
    else:
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",  # saga's default multi_class="auto" already resolves to
                             # multinomial for >2 classes, so no need to set it
                             # explicitly (newer sklearn removed the kwarg entirely)
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1,
        )

    t0 = time.time()
    clf.fit(X_sc, y_train)
    fit_seconds = time.time() - t0
    n_iter = clf.n_iter_ if np.isscalar(clf.n_iter_) else clf.n_iter_[0]

    coef_df = pd.DataFrame(clf.coef_, index=clf.classes_, columns=kept_act_cols)
    top_latents = {
        cat: coef_df.loc[cat][coef_df.loc[cat] > 0]
                    .sort_values(ascending=False)
                    .head(top_k)
        for cat in clf.classes_
    }

    return clf, scaler, coef_df, top_latents, df_filtered, n_iter, fit_seconds


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
        description="Train multiclass probe of SAE latents on train set, evaluate on test "
                    "set, using TorchL1LogisticRegression (full-batch FISTA in PyTorch) "
                    "instead of sklearn's saga solver. Same feature selection / reporting "
                    "as probing/probe_multiclass_train_test.py, just a faster solver."
    )
    parser.add_argument("--train_file",     type=str, required=True)
    parser.add_argument("--test_file",      type=str, required=True)
    parser.add_argument("--target_col",     type=str, required=True)
    parser.add_argument(
        "--important_act_pkl", type=str, required=True,
        help="Path to a pickle mapping POS -> list of important 'act_*' latent names "
             "(e.g. important_act_per_pos_layer30.pkl). Passed explicitly (rather than "
             "the hardcoded layer_n lookup in probing/probe_multiclass_train_test.py) so "
             "a sweep script can vary it.",
    )
    parser.add_argument("--min_latent_freq", type=int, default=100)
    parser.add_argument("--C",              type=float, default=0.1)
    parser.add_argument("--max_iter",       type=int, default=1000)
    parser.add_argument("--top_k",          type=int, default=20)
    parser.add_argument(
        "--backend", type=str, choices=["torch", "sklearn"], default="torch",
        help="'torch' (default): full-batch FISTA in PyTorch. 'sklearn': original "
             "LogisticRegression(solver='saga') path, kept for cross-checking.",
    )
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--torch_verbose", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir",     type=str, default=".")
    args = parser.parse_args()

    print(f"Loading important-latent set from {args.important_act_pkl}...")
    with open(args.important_act_pkl, "rb") as f:
        important_act_per_pos = pickle.load(f)
    important_acts = set(e for l in important_act_per_pos.values() for e in l)

    print(f"Loading train data from {args.train_file}...")
    df_train = pd.read_parquet(args.train_file)
    print(f"Loading test data from {args.test_file}...")
    df_test = pd.read_parquet(args.test_file)

    print("Selecting features...")
    kept_act_cols = select_features(df_train, important_acts, args.min_latent_freq)
    print(f"  {len(kept_act_cols)} features retained after filtering")

    print(f"Training classifier on full training set (target: {args.target_col}, "
          f"backend={args.backend}, C={args.C})...")
    clf, scaler, coef_df, top_latents, df_train_filtered, n_iter, fit_seconds = train_classifier(
        df_train, args.target_col, kept_act_cols, args.C, args.max_iter, args.top_k,
        args.backend, args.tol, args.device, args.torch_verbose,
    )
    print(f"Training finished in {fit_seconds:.1f}s ({n_iter} iterations).")

    print("Evaluating on test set...")
    test_report, test_cm, df_test_predictions = evaluate_on_test(
        clf, scaler, df_test, args.target_col, kept_act_cols
    )

    train_basename = os.path.splitext(os.path.basename(args.train_file))[0]
    iap_basename = os.path.splitext(os.path.basename(args.important_act_pkl))[0]
    c_str = str(args.C).replace(".", "p")
    base_name = f"{train_basename}_{args.target_col}_{iap_basename}_C{c_str}_multiclass_torch"

    results = {
        "backend":               args.backend,
        "clf":                   clf if args.backend == "sklearn" else {
            "coef_": clf.coef_, "intercept_": clf.intercept_,
            "classes_": clf.classes_, "n_iter_": n_iter,
        },
        "coefficients":          coef_df,
        "top_latents":           top_latents,
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm,
        "classes":               clf.classes_.tolist(),
        "feature_names":         kept_act_cols,
        "n_iter":                n_iter,
        "fit_seconds":           fit_seconds,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    results_path      = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    scaler_path       = os.path.join(args.output_dir, f"{base_name}_scaler.pkl")
    train_df_path     = os.path.join(args.output_dir, f"{base_name}_train_filtered.parquet")
    test_preds_path   = os.path.join(args.output_dir, f"{base_name}_test_predictions.parquet")
    metrics_path      = os.path.join(args.output_dir, f"{base_name}_metrics.json")

    print(f"Saving results to {results_path}...")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    df_train_filtered.to_parquet(train_df_path, compression="gzip")
    df_test_predictions.to_parquet(test_preds_path, compression="gzip")

    metrics = {
        "script":              "probe_multiclass_train_test_torch",
        "backend":             args.backend,
        "train_file":          args.train_file,
        "test_file":           args.test_file,
        "important_act_pkl":   args.important_act_pkl,
        "target_col":          args.target_col,
        "C":                   args.C,
        "min_latent_freq":     args.min_latent_freq,
        "n_features":          len(kept_act_cols),
        "n_classes":           len(results["classes"]),
        "test_f1_macro":       test_report["macro avg"]["f1-score"],
        "test_f1_weighted":    test_report["weighted avg"]["f1-score"],
        "test_accuracy":       test_report["accuracy"],
        "n_iter":              int(n_iter),
        "fit_seconds":         fit_seconds,
        "results_path":        results_path,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
