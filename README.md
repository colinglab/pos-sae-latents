# Parts-of-Speech as Emergent Categories in SAE Latent Space

Code for the EMNLP 2026 paper *"Parts-of-Speech as Emergent Categories in SAE Latent
Space"* (Bondielli, Passaro, Auriemma, Lenci). We study whether part-of-speech
(PoS) categories are encoded by individual Sparse Autoencoder (SAE) latents or
by structured groups of latents, using LLaMA-3-8B activations on the UD
English GUM treebank encoded with the `EleutherAI/sae-llama-3-8b-32x` SAE.

The pipeline follows the paper's three research questions:

- **RQ1 - Recoverability**: are PoS distinctions linearly recoverable from SAE
  activations? (one-vs-rest probing classifiers)
- **RQ2 - Organization**: how is that information organized in latent space?
  (feature salience, coverage/compactness, compact-feature multiclass
  classification)
- **RQ3 - Stability**: do the selected latent groups remain stable and
  systematic on held-out data?

## Repository structure

```
src/
  extraction/              Token-level SAE latent extraction from the GUM treebank
    extract_latents.py           SAE latent activations (main pipeline input)
    extract_dense_activations.py Raw dense/static hidden states (Table 2 baselines)

  probing/                 Core probing pipeline (RQ1-RQ3)
    probe_onevsrest.py            RQ1: one-vs-rest binary probes per PoS (Fig. 2, 3)
    probe_multiclass_cv.py        RQ2: compact-feature multiclass probe, CV (Fig. 5, 12; Table 7)
    probe_multiclass_train_test.py RQ3: same, trained on train / evaluated on held-out test

  coverage_and_results/    Salience, coverage, and results analysis (RQ2/RQ3)
    coverage_analysis.py            Feature salience + coverage/compactness (Fig. 4, 13; Table 6);
                                     produces the important_act_per_pos_layer{N}.pkl (L*) files
                                     consumed by probe_multiclass_cv.py / probe_multiclass_train_test.py
    multiclass_results_analysis.py  Coefficient heatmap + confusion matrix + LaTeX report
                                     from a probe_multiclass_* results.pkl (Fig. 5, 12; Table 7)
    coactivation_analysis.py        RQ3 held-out co-activation matrix + distinctiveness D(c)
                                     (Fig. 6; Table 8)

  controls_and_baselines/  Controls and baselines (Section 4.4/5.4)
    probe_bucket_eval.py                    Shared bucketed-evaluation helpers (ambiguous/OOV/open-closed)
    control_task_cv.py / control_task_train_test.py
                                             Hewitt & Liang (2019) control task: lexical-identity confound
    probe_dense_train_test.py               Dense (layer 30) / static-embedding (layer 0) baselines (Table 2)
    majority_baseline_train_test.py         Per-word-type majority/lexicon baseline
    compare_baselines.py                    Combines SAE/dense/static/majority into one bucketed table
    probe_multiclass_random_cv.py / probe_multiclass_random_train_test.py
                                             Random-latent-subset baseline at controlled overlap % (Table 3)
    aggregate_random_overlap_results.py     Summarizes the overlap sweep above

  robustness/              Supplementary hyperparameter-sensitivity sweeps (not a main-paper table)
    probe_multiclass_torch.py / probe_multiclass_train_test_torch.py
                                             GPU-accelerated (PyTorch FISTA) re-implementation of the
                                             compact-feature probe, for sweeping C x coverage threshold
    aggregate_sweep_results.py              Collects the above sweep runs into summary.csv
    probe_onevsrest_torch.py                Same idea, one-vs-rest per PoS instead of one multinomial
                                             classifier: binary FISTA probe per PoS restricted to an
                                             L* important-latent set, for sweeping C x coverage threshold
    aggregate_onevsrest_sweep_results.py    Collects the one-vs-rest sweep into a long-format table plus
                                             a PoS x C pivot per coverage-threshold file

  visualization/
    visualize_probing.py                    Per-layer summary/top-latent/UMAP plots from probe_onevsrest.py results
    visualize_probing_multilayer.py         Cross-layer F1 plot by PoS class (Appendix Fig. 8)

scripts/                   Shell entry points chaining the above (see "Running the pipeline")
requirements.txt
LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

You will also need:

- **UD English GUM treebank**: not vendored here. Clone it into your working
  data directory:
  ```bash
  git clone https://github.com/UniversalDependencies/UD_English-GUM.git
  ```
  GUM is distributed under CC BY 4.0 (with some subcorpora under more
  restrictive terms) - see the treebank's own repository for details.
- **Model/SAE access**: `meta-llama/Meta-Llama-3-8B` (Llama 3 Community
  License) and `EleutherAI/sae-llama-3-8b-32x`, both loaded from HuggingFace
  Hub at runtime (no local copy needed).
- **Large data artifacts** (extracted activations, trained probe results):
  this repo is code-only. The authors will publish the token-level SAE
  activations and probe outputs used in the paper on HuggingFace
  (link to be added upon release / acceptance).

All scripts are plain argparse CLIs; run them from whatever working directory
holds your data (they don't assume they live next to your data files).

## Running the pipeline

Run from a data directory (with `UD_English-GUM/` alongside it) using the
scripts in `scripts/`, or call the underlying `src/` modules directly.

1. **Extract SAE latents** (`scripts/01_extract_latents.sh`): parses the GUM
   CoNLL-U files, runs LLaMA-3-8B + the SAE, and aligns subword activations to
   UD tokens -> `layer-wise-latents/latents_{split}_..._{layer}.parquet`.
2. **RQ1 one-vs-rest probing** (`scripts/02_run_onevsrest_probes.sh`): one
   L1-regularized logistic-regression classifier per PoS tag, 5-fold CV.
3. **RQ2 salience & coverage** (`src/coverage_and_results/coverage_analysis.py`):
   ranks latents by classifier coefficient, finds the smallest per-PoS latent
   set reaching 95% token coverage, and saves it as
   `important_act_per_pos_layer{N}.pkl`.
4. **RQ2 compact-feature classification**
   (`src/probing/probe_multiclass_cv.py`): multinomial classifier trained only
   on the union L* of the sets from step 3.
5. **RQ3 held-out validation** (`src/probing/probe_multiclass_train_test.py`
   + `src/coverage_and_results/coactivation_analysis.py`): trains on the GUM
   train split, evaluates on the held-out test split, and computes the
   cross-PoS co-activation matrix and distinctiveness scores.
6. **Controls and baselines** (`src/controls_and_baselines/`): lexical-identity
   control task, dense/static/majority baselines, and the random-latent
   overlap sweep (`scripts/04_run_random_overlap_baseline.sh`), see Section
   4.4/5.4 and Tables 2-3.
7. **Robustness checks** (optional): `scripts/03_run_multilayer_robustness.sh`
   repeats steps 4-5 at additional layers (Appendix A.1, Fig. 8, needs
   `visualize_probing_multilayer.py` to plot); `scripts/05_run_hparam_sweep.sh`
   sweeps the coverage threshold and C hyperparameter for the compact-feature
   multiclass probe; `scripts/06_run_onevsrest_hparam_sweep.sh` does the same
   sweep for the RQ1 one-vs-rest probes (needs the
   `important_act_per_pos_layer{N}_t{0.9,0.95,0.99}.pkl` files from step 3).
   Both sweep scripts use a GPU-accelerated PyTorch solver
   (`src/robustness/*_torch.py`) instead of sklearn, for speed.

## Known gap: controlled dataset (Section 3.1/4.3)

This repository does **not** include the code that generates or evaluates the
paper's 180-item controlled dataset (the noun/verb sentence templates like
*"There is a dog"* / *"I see the dog"*, Table 1's pseudo-multilabel results,
and Figures 7 and 14-17). That code was not found in the source working
directory this repo was built from - if you have it, add it under e.g.
`src/controlled_dataset/` before treating this repo as a full reproduction of
the paper.

## Citation

```bibtex
@inproceedings{bondielli2026pos,
  title     = {Parts-of-Speech as Emergent Categories in {SAE} Latent Space},
  author    = {Bondielli, Alessandro and Passaro, Lucia and Auriemma, Serena and Lenci, Alessandro},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  month     = oct,
  publisher = {Association for Computational Linguistics},
  note      = {To appear}
}
```

## License

MIT - see [LICENSE](LICENSE).
