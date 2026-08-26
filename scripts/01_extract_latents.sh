#!/bin/bash
# Extracts token-level SAE latent activations for one or more LLM layers.
# Run from a working data directory that contains UD_English-GUM/ (see README).
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

# 30 is the layer used for the paper's main results. Add 2 and 15 here too to
# reproduce the layer-robustness check in Appendix A.1 (Figure 8), then feed
# their outputs to 03_run_multilayer_robustness.sh.
layers=(30)

for i in "${layers[@]}"; do
    python "${SRC_DIR}/extraction/extract_latents.py" \
        --split train \
        --model_id meta-llama/Meta-Llama-3-8B \
        --sae_id EleutherAI/sae-llama-3-8b-32x \
        --sae_hookpoint "layers.${i}" \
        --hidden_layer "${i}" \
        --output_dir layer-wise-latents
done
