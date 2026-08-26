import os
import re
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae

def parse_conllu(filepath: str) -> list[dict]:
    """Parses a CoNLL-U file into a list of sentence dictionaries."""
    sentences = []
    with open(filepath, encoding="utf-8") as fh:
        raw = fh.read()

    blocks = re.split(r"\n\s*\n", raw.strip())

    for block in blocks:
        lines = block.splitlines()
        sent = {
            "sent_id": None,
            "text": None,
            "metadata": {},
            "features": defaultdict(int),
            "tokens": [],
        }

        for line in lines:
            if line.startswith("#"):
                m = re.match(r"#\s*([\w.]+)\s*=\s*(.+)", line)
                if m:
                    key, value = m.group(1).strip(), m.group(2).strip()
                    if key == "sent_id":
                        sent["sent_id"] = value
                    elif key == "text":
                        sent["text"] = value
                    else:
                        sent["metadata"][key] = value
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue
            token_id = parts[0]
            if "-" in token_id or "." in token_id:
                continue

            token = {
                "id": token_id, "form": parts[1], "lemma": parts[2],
                "upos": parts[3], "xpos": parts[4], "feats": parts[5],
                "head": parts[6], "deprel": parts[7], "deps": parts[8],
                "misc": parts[9],
            }
            sent["tokens"].append(token)

            if token["feats"] not in ("_", ""):
                for feat in token["feats"].split("|"):
                    sent["features"][feat] += 1

        sent["features"] = dict(sent["features"])
        sentences.append(sent)

    return sentences

def conllu_to_token_df(sentences: list[dict]) -> pd.DataFrame:
    """Converts parsed sentences into a flat DataFrame of tokens."""
    rows = []
    for s in sentences:
        if not s["text"]:
            continue
        for token in s["tokens"]:
            rows.append({
                "sent_id":  s["sent_id"],
                "text":     s["text"],
                "token_id": token["id"],
                "form":     token["form"],
                "lemma":    token["lemma"],
                "upos":     token["upos"],
                "xpos":     token["xpos"],
                "feats":    token["feats"],
                "head":     token["head"],
                "deprel":   token["deprel"],
                "deps":     token["deps"],
                "misc":     token["misc"],
            })
    return pd.DataFrame(rows)

def extract_latents_aligned(
    token_df: pd.DataFrame,
    model,
    tokenizer,
    sae,
    device,
    hidden_layer: int = 10,
    window: int = 0,
):
    """
    Extract SAE latent activations aligned to UD tokens.
    """
    grouped = token_df.groupby("sent_id", sort=False)
    all_sparse = []
    all_indices = []

    with torch.no_grad():
        for sent_id, group in tqdm(grouped, desc="Extracting latents"):
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
            hidden = outputs.hidden_states[hidden_layer].flatten(0, 1)
            latent_acts = sae.encode(hidden)

            top_indices = latent_acts.top_indices.cpu()
            top_acts = latent_acts.top_acts.cpu()
            n_subwords = top_indices.shape[0]

            subword_sparse = []
            for t_idx, t_act in zip(top_indices.tolist(), top_acts.tolist()):
                subword_sparse.append({idx: act for idx, act in zip(t_idx, t_act)})

            char_pos = 0
            for ud_form, row_idx in zip(ud_forms, row_indices):
                token_start = sentence.find(ud_form, char_pos)
                if token_start == -1:
                    all_sparse.append({})
                    all_indices.append(row_idx)
                    continue
                token_end = token_start + len(ud_form)
                char_pos = token_end

                aligned_subwords = [
                    i for i, (s, e) in enumerate(offset_mapping)
                    if s < token_end and e > token_start and e > s
                ]

                if not aligned_subwords:
                    all_sparse.append({})
                    all_indices.append(row_idx)
                    continue

                anchor = aligned_subwords[0]
                win_start = max(0, anchor - window)
                win_end = min(n_subwords - 1, anchor + window)

                token_sparse = {}
                for pos in range(win_start, win_end + 1):
                    for idx, act in subword_sparse[pos].items():
                        if act > token_sparse.get(idx, 0.0):
                            token_sparse[idx] = act

                all_sparse.append(token_sparse)
                all_indices.append(row_idx)

    all_latent_ids = sorted({idx for d in all_sparse for idx in d})
    acts_df = pd.DataFrame(
        [{lid: d.get(lid, 0.0) for lid in all_latent_ids} for d in all_sparse],
        index=all_indices,
    ).round(4)
    acts_df.columns = [f"act_{i}" for i in acts_df.columns]

    return acts_df

def main():
    parser = argparse.ArgumentParser(description="Extract SAE latents for GUM tokens.")
    parser.add_argument("--split", type=str, required=True, help="GUM split (train, test, dev)")
    parser.add_argument("--model_id", type=str, required=True, help="LLM model ID")
    parser.add_argument("--sae_id", type=str, required=True, help="SAE hub ID")
    parser.add_argument("--sae_hookpoint", type=str, required=True, help="SAE hookpoint")
    parser.add_argument("--hidden_layer", type=int, default=10, help="Layer to extract hidden states from")
    parser.add_argument("--window", type=int, default=0, help="Symmetric window half-width")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save parquet")

    args = parser.parse_args()

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "6" # Default as per original code, but could be an arg
    device = args.device

    print(f"Loading SAE: {args.sae_id} at {args.sae_hookpoint}")
    sae = Sae.load_from_hub(args.sae_id, hookpoint=args.sae_hookpoint).to(device)

    print(f"Loading Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)

    print(f"Parsing GUM {args.split} split...")
    conllu_path = f"UD_English-GUM/en_gum-ud-{args.split}.conllu"
    sentences = parse_conllu(conllu_path)
    token_df = conllu_to_token_df(sentences)

    print("Extracting latents...")
    acts_df = extract_latents_aligned(
        token_df, model, tokenizer, sae, device,
        hidden_layer=args.hidden_layer,
        window=args.window
    )

    final_df = pd.concat([token_df, acts_df], axis=1)

    # Naming: latents_{split}_{llm}_{sae_layer}.parquet
    # We extract the layer from hookpoint if possible or use the arg
    layer_name = args.sae_hookpoint.split(".")[-1] if "." in args.sae_hookpoint else args.sae_hookpoint
    filename = f"latents_{args.split}_{args.model_id.replace('/', '_')}_{layer_name}.parquet"
    output_path = os.path.join(args.output_dir, filename)

    print(f"Saving to {output_path}...")
    final_df.to_parquet(output_path, compression="gzip")
    print("Done.")

if __name__ == "__main__":
    main()
