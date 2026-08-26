import os
import time
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")


class TorchL1LogisticRegression:
    """
    Minimal sklearn-compatible (fit/predict/classes_/coef_/intercept_) drop-in
    for LogisticRegression(penalty="l1", solver="saga"/"liblinear",
    class_weight="balanced"), fit with full-batch FISTA (proximal gradient +
    Nesterov acceleration + backtracking line search) in PyTorch instead of
    sklearn's solver.

    Works for any number of classes via a softmax parameterization; for the
    binary one-vs-rest case used here (n_classes == 2) this is an
    over-parameterized but equivalent restatement of binary logistic
    regression - see `binary_coef_` below for the single-vector form
    comparable to sklearn's liblinear output.

    Kept as a per-script copy rather than a shared import, so each probe
    script stays self-contained (same class as in probe_multiclass_torch.py /
    probe_multiclass_train_test_torch.py).
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
        t = 1.0
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
                print(f"      [torch-fista] iter {it:4d}  obj={obj:.6f}  step={t:.2e}  "
                      f"elapsed={time.time() - t0:.1f}s")

            W, b = W_new, b_new
            theta = theta_new
            t = min(t * 1.5, 1.0)

            if prev_obj is not None:
                rel_change = abs(prev_obj - obj) / max(1.0, abs(prev_obj))
                if rel_change < self.tol:
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

    def binary_coef_(self):
        """
        Single-vector coefficients/intercept comparable to sklearn's binary
        liblinear output: the log-odds direction for class `classes_[1]`
        (positive class, expected to be 1) versus `classes_[0]`.
        """
        pos_idx = list(self.classes_).index(1) if 1 in self.classes_ else 1
        neg_idx = 1 - pos_idx
        return self.coef_[pos_idx] - self.coef_[neg_idx], self.intercept_[pos_idx] - self.intercept_[neg_idx]


def make_classifier(backend, C, max_iter, tol, device, torch_verbose):
    if backend == "torch":
        return TorchL1LogisticRegression(
            C=C, max_iter=max_iter, tol=tol, class_weight="balanced",
            device=device, verbose=torch_verbose, random_state=42,
        )
    return LogisticRegression(
        penalty="l1",
        solver="liblinear",  # matches probing/probe_onevsrest.py's original binary solver
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
    )


def get_binary_coef(clf, backend, kept_act_cols):
    if backend == "torch":
        coef, intercept = clf.binary_coef_()
        return pd.Series(coef, index=kept_act_cols), intercept
    return pd.Series(clf.coef_[0], index=kept_act_cols), clf.intercept_[0]


def probe_column_onevsrest(
    df,
    important_acts,
    target_col,
    min_token_freq=50,
    min_latent_freq=100,
    C=0.1,
    max_iter=1000,
    top_k=20,
    cv=5,
    backend="torch",
    tol=1e-6,
    device=None,
    torch_verbose=0,
):
    """
    One classifier per unique value of `target_col`, restricted to the
    `important_acts` feature set (e.g. the union L* of latents selected by
    coverage_analysis.py at some coverage threshold) - same idea as
    probe_multiclass_torch.py's feature restriction, applied per-POS instead
    of to one shared multinomial classifier.
    """
    act_cols = [c for c in df.columns if c.startswith("act_") and c in important_acts]
    ud_cols = [c for c in df.columns if c not in act_cols]

    mask = (df[act_cols] != 0).sum(axis=0) >= min_latent_freq
    kept_act_cols = df[act_cols].columns[mask].tolist()
    df_filtered = df[ud_cols + kept_act_cols]

    counts = df_filtered[target_col].value_counts()
    valid_categories = counts[counts >= min_token_freq].index.tolist()
    df_filtered = df_filtered[df_filtered[target_col].isin(valid_categories)]

    X = df_filtered[kept_act_cols].values.astype(np.float32)
    scaler = StandardScaler(with_mean=False)
    X_sc = scaler.fit_transform(X)

    results = {}
    for cat in sorted(valid_categories):
        y = (df_filtered[target_col] == cat).astype(int).values
        n_pos = int(y.sum())
        print(f"  [{cat}] n_positive={n_pos}  fitting (backend={backend})...")

        # Manual StratifiedKFold CV: TorchL1LogisticRegression doesn't
        # implement get_params/clone, so it can't go through cross_val_score.
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        fold_scores = []
        for train_idx, val_idx in skf.split(X_sc, y):
            fold_clf = make_classifier(backend, C, max_iter, tol, device, torch_verbose=0)
            fold_clf.fit(X_sc[train_idx], y[train_idx])
            y_val_pred = fold_clf.predict(X_sc[val_idx])
            fold_scores.append(f1_score(y[val_idx], y_val_pred, pos_label=1, zero_division=0))
        fold_scores = np.array(fold_scores)

        t0 = time.time()
        clf = make_classifier(backend, C, max_iter, tol, device, torch_verbose)
        clf.fit(X_sc, y)
        fit_seconds = time.time() - t0
        n_iter = clf.n_iter_ if np.isscalar(getattr(clf, "n_iter_", 0)) else clf.n_iter_[0]

        coefs, intercept = get_binary_coef(clf, backend, kept_act_cols)
        top_latents = coefs[coefs > 0].sort_values(ascending=False).head(top_k)

        results[cat] = {
            "backend":       backend,
            "coefficients":  coefs,
            "intercept":     float(intercept),
            "top_latents":   top_latents,
            "cv_f1":         fold_scores.mean(),
            "cv_f1_std":     fold_scores.std(),
            "cv_f1_per_fold": fold_scores.tolist(),
            "n_positive":    n_pos,
            "n_iter":        int(n_iter),
            "fit_seconds":   fit_seconds,
        }
        print(f"    {cat:10s}  F1={fold_scores.mean():.3f} (+/-{fold_scores.std():.3f})  "
              f"non-zero coefs={(coefs != 0).sum():4d}  fit={fit_seconds:.2f}s")

    return results, scaler, df_filtered, kept_act_cols


def main():
    parser = argparse.ArgumentParser(
        description="One-vs-rest binary probe of SAE latents per PoS tag, restricted to an "
                    "important-latent set (L*) and trained with TorchL1LogisticRegression "
                    "(full-batch FISTA in PyTorch) instead of sklearn's liblinear solver. "
                    "Same feature selection / CV / reporting shape as probing/probe_onevsrest.py, "
                    "just filtered to --important_act_pkl and a faster solver - built for "
                    "sweeping C x coverage-threshold the same way probe_multiclass_torch.py "
                    "does for the multiclass probe (see scripts/06_run_onevsrest_hparam_sweep.sh)."
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument(
        "--important_act_pkl", type=str, required=True,
        help="Path to a pickle mapping POS -> list of important 'act_*' latent names "
             "(e.g. important_act_per_pos_layer30_t0.95.pkl, from coverage_analysis.py). "
             "The union across all POS is used as the candidate feature set for every "
             "one-vs-rest classifier.",
    )
    parser.add_argument("--min_token_freq", type=int, default=50)
    parser.add_argument("--min_latent_freq", type=int, default=100)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--backend", type=str, choices=["torch", "sklearn"], default="torch",
        help="'torch' (default): full-batch FISTA in PyTorch. 'sklearn': original "
             "LogisticRegression(solver='liblinear') path, kept for cross-checking.",
    )
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--torch_verbose", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Loading important-latent set from {args.important_act_pkl}...")
    with open(args.important_act_pkl, "rb") as f:
        important_act_per_pos = pickle.load(f)
    important_acts = set(e for l in important_act_per_pos.values() for e in l)

    print(f"Loading data from {args.input_file}...")
    df = pd.read_parquet(args.input_file)

    print(f"Probing target column (one-vs-rest): {args.target_col}...")
    t0 = time.time()
    results, scaler, df_filtered, kept_act_cols = probe_column_onevsrest(
        df, important_acts,
        target_col=args.target_col,
        min_token_freq=args.min_token_freq,
        min_latent_freq=args.min_latent_freq,
        C=args.C,
        max_iter=args.max_iter,
        top_k=args.top_k,
        cv=args.cv,
        backend=args.backend,
        tol=args.tol,
        device=args.device,
        torch_verbose=args.torch_verbose,
    )
    total_seconds = time.time() - t0

    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    iap_basename = os.path.splitext(os.path.basename(args.important_act_pkl))[0]
    c_str = str(args.C).replace(".", "p")
    base_name = f"{input_basename}_{args.target_col}_{iap_basename}_C{c_str}_onevsrest_torch"

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{base_name}_results.pkl")
    scaler_path = os.path.join(args.output_dir, f"{base_name}_scaler.pkl")
    df_filtered_path = os.path.join(args.output_dir, f"{base_name}_df_filtered.parquet")
    metrics_path = os.path.join(args.output_dir, f"{base_name}_metrics.json")

    print(f"Saving results to {results_path}...")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    df_filtered.to_parquet(df_filtered_path, compression="gzip")

    per_pos = [
        {
            "pos": pos,
            "n_positive": r["n_positive"],
            "cv_f1_mean": r["cv_f1"],
            "cv_f1_std": r["cv_f1_std"],
            "n_nonzero_coefs": int((r["coefficients"] != 0).sum()),
            "n_iter": r["n_iter"],
            "fit_seconds": r["fit_seconds"],
        }
        for pos, r in sorted(results.items())
    ]
    macro_f1 = float(np.mean([r["cv_f1"] for r in results.values()])) if results else float("nan")

    metrics = {
        "script": "probe_onevsrest_torch",
        "backend": args.backend,
        "input_file": args.input_file,
        "important_act_pkl": args.important_act_pkl,
        "target_col": args.target_col,
        "C": args.C,
        "min_latent_freq": args.min_latent_freq,
        "min_token_freq": args.min_token_freq,
        "n_features": len(kept_act_cols),
        "n_pos_tags": len(results),
        "macro_f1_mean": macro_f1,
        "total_seconds": total_seconds,
        "per_pos": per_pos,
        "results_path": results_path,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMacro-F1 across {len(results)} PoS tags: {macro_f1:.4f}")
    print(f"Saved metrics to {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
