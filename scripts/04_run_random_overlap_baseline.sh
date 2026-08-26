#!/bin/bash
# Random-latent baseline (paper Table 3): re-trains the compact-feature
# multiclass probe on latent sets that share only overlap_pct% of their
# latents with the real L* set, at a controlled sweep of overlap percentages.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

TARGET_COL="upos"
INPUT_DIR="layer-wise-latents"
CV_OUTPUT_DIR="probing-results-random-overlap"
TRAIN_TEST_OUTPUT_DIR="probing-results-train-test-random-overlap"
SEED=42
OVERLAP_LEVELS=(0 25 50 75)  # Table 3 also reports "100 (original)" = the main L* pipeline itself

mkdir -p "${CV_OUTPUT_DIR}" "${TRAIN_TEST_OUTPUT_DIR}"

echo "=== Cross-validation experiment (probe_multiclass_random_cv.py) ==="
input_file="${INPUT_DIR}/latents_train_meta-llama_Meta-Llama-3-8B_30.parquet"
basename=$(basename "${input_file}" .parquet)

for overlap_pct in "${OVERLAP_LEVELS[@]}"; do
    expected_output="${CV_OUTPUT_DIR}/${basename}_${TARGET_COL}_multiclass_random_overlap${overlap_pct}_seed${SEED}_df_filtered.parquet"
    if [ -f "${expected_output}" ]; then
        echo "Skipping ${input_file} (overlap=${overlap_pct}%, already processed)"
        continue
    fi
    echo "Processing ${input_file} (overlap=${overlap_pct}%)..."
    python "${SRC_DIR}/controls_and_baselines/probe_multiclass_random_cv.py" \
        --input_file "${input_file}" \
        --target_col "${TARGET_COL}" \
        --overlap_pct "${overlap_pct}" \
        --seed "${SEED}" \
        --output_dir "${CV_OUTPUT_DIR}"
done

echo "=== Train/test experiment (probe_multiclass_random_train_test.py) ==="
for train_file in ${INPUT_DIR}/latents_train_*.parquet; do
    train_basename=$(basename "${train_file}" .parquet)
    test_file="${INPUT_DIR}/${train_basename/latents_train_/latents_test_}.parquet"

    if [ ! -f "${test_file}" ]; then
        echo "Skipping ${train_file} (no matching test file ${test_file})"
        continue
    fi

    for overlap_pct in "${OVERLAP_LEVELS[@]}"; do
        expected_output="${TRAIN_TEST_OUTPUT_DIR}/${train_basename}_${TARGET_COL}_multiclass_random_overlap${overlap_pct}_seed${SEED}_test_predictions.parquet"
        if [ -f "${expected_output}" ]; then
            echo "Skipping ${train_file} (overlap=${overlap_pct}%, already processed)"
            continue
        fi
        echo "Processing ${train_file} / ${test_file} (overlap=${overlap_pct}%)..."
        python "${SRC_DIR}/controls_and_baselines/probe_multiclass_random_train_test.py" \
            --train_file "${train_file}" \
            --test_file "${test_file}" \
            --target_col "${TARGET_COL}" \
            --overlap_pct "${overlap_pct}" \
            --seed "${SEED}" \
            --output_dir "${TRAIN_TEST_OUTPUT_DIR}"
    done
done

echo "=== Aggregating results (Table 3) ==="
python "${SRC_DIR}/controls_and_baselines/aggregate_random_overlap_results.py" \
    --cv_dir "${CV_OUTPUT_DIR}" \
    --train_test_dir "${TRAIN_TEST_OUTPUT_DIR}"
