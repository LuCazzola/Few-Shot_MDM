import os
import json
import argparse


from pathlib import Path
from os.path import join as pjoin
import numpy as np
import json
import torch
from tqdm import tqdm
import clip
from sklearn.neighbors import NearestNeighbors

def load_humanml3d_descriptions(root_folder):
    descriptions = []
    all_txt_files = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if f.endswith('.txt')]
    for file_path in tqdm(all_txt_files, desc="Loading HumanML3D descriptions"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    clean_text = line.split('#')[0].strip()
                    descriptions.append(clean_text)
    return descriptions

def encode_texts(text_list, model, device, batch_size=32):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding texts"):
            batch = text_list[i:i+batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

def get_or_compute_embeddings(text_list, model, device, args, out_file="humanml3d_clip_embeddings.pt"):

    cached_embeddigs = os.path.join(args.cache_dir, out_file)
    
    if os.path.exists(cached_embeddigs):
        print(f"Loading cached embeddings from: {cached_embeddigs}")
        return torch.load(cached_embeddigs, map_location='cpu')
    else:
        print(f"Computing embeddings and saving to: {cached_embeddigs}")
        embeddings = encode_texts(text_list, model, device)
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save(embeddings, cached_embeddigs)
        return embeddings

def compute_knn_stats(
    ntu_label_embeds, ntu_action_labels, humanml3d_embeds,
    descriptions, args
):
    print(f"Computing density using K={args.k} nearest neighbors...")
    os.makedirs(args.save_dir, exist_ok=True)

    nn = NearestNeighbors(n_neighbors=args.k, metric='cosine')
    nn.fit(humanml3d_embeds)

    label_stats = []

    for idx, label_embed in enumerate(ntu_label_embeds):
        label_embed = label_embed.unsqueeze(0)
        distances, indices = nn.kneighbors(label_embed, return_distance=True)
        distances = distances.flatten()
        indices = indices.flatten()

        mean_dist = float(np.mean(distances))
        density = 1.0 - mean_dist if mean_dist > 0 else float('inf')

        top_5 = sorted(zip(distances, indices))[:5]
        top_5_closest = [
            {"distance": float(d), "text": descriptions[i]} for d, i in top_5
        ]

        label_stats.append({
            "action_label": ntu_action_labels[idx],
            "density_score": density,
            "top_5_closest": top_5_closest
        })
    # Final JSON structure
    result = {
        "k": args.k,
        "labels": label_stats
    }
    # Optionally save to a file
    out_file = os.path.join(args.save_dir, f"ntu_density_k{args.k}_{'action-label-only' if args.use_action_label else ''}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute k-NN stats between NTU actions and HumanML3D descriptions.")
    parser.add_argument("--k", type=int, default=200, help="Number of nearest neighbors to consider.")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory to save cached embeddings.")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results.")
    parser.add_argument("--use-action-label", action="store_true", default=False, help="Use action label directly as text prompt for action classes, otherwise use natural language adaptation (default: False).")
    args = parser.parse_args()

    ROOT = Path('.').resolve()
    OUT_PATH = Path(__file__).parent.relative_to(ROOT)
    NTU_ACTION_CAPTIONS = pjoin(ROOT, 'data', 'NTU60', 'action_captions.json')
    HML_TXT_ROOT = pjoin(ROOT, 'data', 'HumanML3D', 'texts')
    CACHE_DIR = pjoin(ROOT, args.cache_dir)
    SAVE_DIR = pjoin(ROOT, args.save_dir)

    print("Loading NTU action labels...")
    with open(NTU_ACTION_CAPTIONS, 'r') as file:
        ntu_data = json.load(file)    
    ntu_action_labels = [
        (values['action'] if args.use_action_label else values['captions'][0]) # first caotuib only among the possible ones
        for action_id, values in ntu_data.items()
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print(f"Loading & formatting HumanML3D descriptions from {HML_TXT_ROOT}...")
    humanml3d_descriptions = load_humanml3d_descriptions(HML_TXT_ROOT)

    print("Encoding HumanML3D descriptions and NTU action labels...")
    humanml3d_embeds = get_or_compute_embeddings(humanml3d_descriptions, model, device, args)

    print("Encoding NTU RGB+D action labels...")
    ntu_label_embeds = encode_texts(ntu_action_labels, model, device)

    print("Computing k-NN stats...")
    compute_knn_stats(ntu_label_embeds, ntu_action_labels, humanml3d_embeds, humanml3d_descriptions, args)