import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-use the exact same CoNLL-U parsing as the SAE-latent extraction pipeline
# so token/sentence boundaries line up identically across experiments.
from extract_latents import parse_conllu, conllu_to_token_df


def extract_hidden_aligned(token_df, model, tokenizer, device, hidden_layers, window=0):
    """
    Extract raw dense hidden-state vectors aligned to UD tokens, for each
    requested `hidden_states` index. No SAE involved: hidden_states[0] is the
    pre-transformer token-embedding output (non-contextual "static"
    baseline); hidden_states[30] (or whatever index matches this project's
    SAE hookpoint) is the contextual "dense" baseline.

    Mirrors the subword-alignment logic in extract_latents.py's
    extract_latents_aligned, but mean-pools raw vectors over the alignment
    window instead of taking an elementwise max over sparse activations
    (max-over-window made sense for "did this latent fire nearby"; mean is
    the standard choice for continuous dense vectors).
    """
    grouped = token_df.groupby("sent_id", sort=False)
    per_layer_vectors = {layer: [] for layer in hidden_layers}
    all_indices = []

    with torch.no_grad():
        for sent_id, group in tqdm(grouped, desc="Extracting hidden states"):
            sentence = group["text"].iloc[0]
            ud_forms = group["form"].tolist()
            row_indices = group.index.tolist()

            encoding = tokenizer(
                sentence,
                return_tensors="pt",
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offset_mapping = encoding.pop("offset_mapping")[0].tolist()

            outputs = model(**encoding.to(device), output_hidden_states=True)
            layer_hidden = {
                layer: outputs.hidden_states[layer][0].float().cpu()
                for layer in hidden_layers
            }
            n_subwords = layer_hidden[hidden_layers[0]].shape[0]

            char_pos = 0
            for ud_form, row_idx in zip(ud_forms, row_indices):
                token_start = sentence.find(ud_form, char_pos)
                if token_start == -1:
                    for layer in hidden_layers:
                        per_layer_vectors[layer].append(None)
                    all_indices.append(row_idx)
                    continue
                token_end = token_start + len(ud_form)
                char_pos = token_end

                aligned_subwords = [
                    i for i, (s, e) in enumerate(offset_mapping)
                    if s < token_end and e > token_start and e > s
                ]
                if not aligned_subwords:
                    for layer in hidden_layers:
                        per_layer_vectors[layer].append(None)
                    all_indices.append(row_idx)
                    continue

                anchor = aligned_subwords[0]
                win_start = max(0, anchor - window)
                win_end = min(n_subwords - 1, anchor + window)

                for layer in hidden_layers:
                    vec = layer_hidden[layer][win_start:win_end + 1].mean(dim=0).numpy()
                    per_layer_vectors[layer].append(vec)

                all_indices.append(row_idx)

    hidden_size = next(
        v for vectors in per_layer_vectors.values() for v in vectors if v is not None
    ).shape[0]

    matrices = {}
    for layer in hidden_layers:
        mat = np.zeros((len(all_indices), hidden_size), dtype=np.float32)
        for i, vec in enumerate(per_layer_vectors[layer]):
            if vec is not None:
                mat[i] = vec
        matrices[layer] = mat

    return matrices, all_indices


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw dense LLM hidden states for GUM tokens (no SAE): "
                    "layer 0 as a non-contextual 'static embedding' baseline, and a "
                    "later layer (e.g. 30, matching this project's SAE hookpoint) as a "
                    "'dense residual-stream' baseline for the same L1 probe."
    )
    parser.add_argument("--split", type=str, required=True, help="GUM split (train, test, dev)")
    parser.add_argument("--model_id", type=str, required=True, help="LLM model ID")
    parser.add_argument(
        "--hidden_layers", type=str, default="0,30",
        help="Comma-separated hidden_states indices to extract, e.g. '0,30'. "
             "0 = pre-transformer token embeddings (static baseline); "
             "30 = this project's SAE hookpoint layer (dense baseline).",
    )
    parser.add_argument("--window", type=int, default=0, help="Symmetric window half-width")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Default as per the SAE extraction script
    device = args.device
    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    print(f"Loading Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.eval()

    print(f"Parsing GUM {args.split} split...")
    conllu_path = f"UD_English-GUM/en_gum-ud-{args.split}.conllu"
    sentences = parse_conllu(conllu_path)
    token_df = conllu_to_token_df(sentences)

    print(f"Extracting hidden states for layers {hidden_layers}...")
    matrices, row_indices = extract_hidden_aligned(
        token_df, model, tokenizer, device, hidden_layers, window=args.window
    )

    meta_df = token_df.loc[row_indices].reset_index(drop=True)
    model_tag = args.model_id.replace("/", "_")

    meta_path = os.path.join(
        args.output_dir, f"dense_activations_{args.split}_{model_tag}_meta.parquet"
    )
    meta_df.to_parquet(meta_path, compression="gzip")
    print(f"Saved metadata ({len(meta_df)} rows) to {meta_path}")

    for layer in hidden_layers:
        npy_path = os.path.join(
            args.output_dir, f"dense_activations_{args.split}_{model_tag}_layer{layer}.npy"
        )
        np.save(npy_path, matrices[layer])
        print(f"Saved layer {layer} activations {matrices[layer].shape} to {npy_path}")

    print("Done.")


if __name__ == "__main__":
    main()
