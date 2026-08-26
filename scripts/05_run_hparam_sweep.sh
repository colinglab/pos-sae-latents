#!/usr/bin/env bash
# Sweeps the torch-backed SAE-latent probes over combinations of
# important_act_per_pos file (i.e. coverage threshold tau) x C, saving
# everything into a fresh timestamped results folder and producing a
# summary.csv at the end. Supplementary hyperparameter-sensitivity check,
# not one of the paper's main reported tables.
#
# Edit the CONFIG block below, then run:
#   ./05_run_hparam_sweep.sh
set -euo pipefail

# ============================== CONFIG ======================================

# Which probe(s) to sweep: "cv", "traintest", or "both".
#   cv        -> probe_multiclass_torch.py            (single input file, CV)
#   traintest -> probe_multiclass_train_test_torch.py (fixed train/test files)
MODE="both"

# Data files (fixed across the whole sweep - only important_act_pkl and C vary).
INPUT_FILE="layer-wise-latents/latents_train_meta-llama_Meta-Llama-3-8B_30.parquet"          # used when MODE is cv/both
TRAIN_FILE="layer-wise-latents/latents_train_meta-llama_Meta-Llama-3-8B_30.parquet"    # used when MODE is traintest/both
TEST_FILE="layer-wise-latents/latents_test_meta-llama_Meta-Llama-3-8B_15.parquet"      # used when MODE is traintest/both

TARGET_COL="upos"

# The dimensions actually being swept. Generate these with coverage_analysis.py
# --threshold 0.9/0.95/0.99 --pkl_suffix _t0.9/_t0.95/_t0.99.
IMPORTANT_ACT_PKLS=(
    "important_act_per_pos_layer30_t0.9.pkl"
    "important_act_per_pos_layer30_t0.95.pkl"
    "important_act_per_pos_layer30_t0.99.pkl"
)
C_VALUES=(0.01 0.05 0.1 0.5 1.0)

# Passed straight through to both scripts.
BACKEND="torch"
MIN_LATENT_FREQ=100
MAX_ITER=1000
TOP_K=20
CV_FOLDS=5
DEVICE=""            # e.g. "cuda:0"; empty = auto-detect
TORCH_VERBOSE=0       # 0 = silence per-iteration FISTA logging during the sweep

OUTPUT_ROOT="sensitivity_sweep_results"

# ============================================================================

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
mkdir -p "${RUN_DIR}"
echo "Results will be written to ${RUN_DIR}"

# Always pass --device, even when DEVICE="": argparse then sets args.device
# to "", and `self.device or (...)` in TorchL1LogisticRegression.fit treats
# an empty string as falsy, so it still auto-detects. This avoids building an
# optional-args array, which bash 3.2 (macOS's default /bin/bash) mishandles
# under `set -u` when empty ("${arr[@]}" raises "unbound variable").

N_RUN=0
N_TOTAL=0
for _ in "${IMPORTANT_ACT_PKLS[@]}"; do
    for _ in "${C_VALUES[@]}"; do
        N_TOTAL=$((N_TOTAL + 1))
    done
done
if [[ "${MODE}" == "both" ]]; then
    N_TOTAL=$((N_TOTAL * 2))
fi

for IAP in "${IMPORTANT_ACT_PKLS[@]}"; do
    for C in "${C_VALUES[@]}"; do
        if [[ "${MODE}" == "cv" || "${MODE}" == "both" ]]; then
            N_RUN=$((N_RUN + 1))
            echo ""
            echo "=== [${N_RUN}/${N_TOTAL}] cv: important_act_pkl=${IAP} C=${C} ==="
            python "${SRC_DIR}/robustness/probe_multiclass_torch.py" \
                --input_file "${INPUT_FILE}" \
                --target_col "${TARGET_COL}" \
                --important_act_pkl "${IAP}" \
                --min_latent_freq "${MIN_LATENT_FREQ}" \
                --C "${C}" \
                --max_iter "${MAX_ITER}" \
                --top_k "${TOP_K}" \
                --cv "${CV_FOLDS}" \
                --backend "${BACKEND}" \
                --torch_verbose "${TORCH_VERBOSE}" \
                --device "${DEVICE}" \
                --output_dir "${RUN_DIR}"
        fi

        if [[ "${MODE}" == "traintest" || "${MODE}" == "both" ]]; then
            N_RUN=$((N_RUN + 1))
            echo ""
            echo "=== [${N_RUN}/${N_TOTAL}] traintest: important_act_pkl=${IAP} C=${C} ==="
            python "${SRC_DIR}/robustness/probe_multiclass_train_test_torch.py" \
                --train_file "${TRAIN_FILE}" \
                --test_file "${TEST_FILE}" \
                --target_col "${TARGET_COL}" \
                --important_act_pkl "${IAP}" \
                --min_latent_freq "${MIN_LATENT_FREQ}" \
                --C "${C}" \
                --max_iter "${MAX_ITER}" \
                --top_k "${TOP_K}" \
                --backend "${BACKEND}" \
                --torch_verbose "${TORCH_VERBOSE}" \
                --device "${DEVICE}" \
                --output_dir "${RUN_DIR}"
        fi
    done
done

echo ""
echo "=== Sweep complete. Aggregating results ==="
python "${SRC_DIR}/robustness/aggregate_sweep_results.py" --results_dir "${RUN_DIR}"

echo ""
echo "Done. Full results + summary.csv in ${RUN_DIR}"
