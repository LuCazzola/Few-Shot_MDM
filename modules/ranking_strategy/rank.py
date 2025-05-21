import os
import json
import argparse

import numpy as np
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
    parser.add_argument(
        "--texts-root", type=str, required=True,
        help="Path to the folder containing HumanML3D .txt description files."
    )
    parser.add_argument(
        "--k", type=int, default=200,
        help="Number of nearest neighbors to consider."
    )
    parser.add_argument(
        "--cache-dir", type=str, default="cache",
        help="Directory to save cached embeddings."
    )
    parser.add_argument(
        "--save-dir", type=str, default="results",
        help="Directory to save results."
    )
    parser.add_argument(
        "--use-action-label", action="store_true", default=False,
        help="Use action label directly as text prompt for action classes, otherwise use natural language adaptation (default: False)."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    texts_root = args.texts_root
    print(f"Loading & formatting HumanML3D descriptions from {texts_root}...")
    humanml3d_descriptions = load_humanml3d_descriptions(texts_root)

    print("Encoding HumanML3D descriptions and NTU action labels...")
    humanml3d_embeds = get_or_compute_embeddings(
        humanml3d_descriptions, model, device, args
    )

    ntu_action_labels = {}
    if args.use_action_label :
        # Use directly the labels from the NTU RGB+D dataset
        ntu_action_labels = {
            0: "drink water", 1: "eat meal", 2: "brush teeth", 3: "brush hair", 4: "drop", 5: "pick up", 6: "throw", 7: "sit down", 8: "stand up", 9: "clapping",
            10: "reading", 11: "writing", 12: "tear up paper", 13: "put on jacket", 14: "take off jacket", 15: "put on a shoe", 16: "take off a shoe", 17: "put on glasses", 18: "take off glasses", 19: "put on a hat/cap",
            20: "take off a hat/cap", 21: "cheer up", 22: "hand waving", 23: "kicking something", 24: "reach into pocket", 25: "hopping", 26: "jump up", 27: "phone call", 28: "play with phone/tablet", 29: "type on a keyboard",
            30: "point to something", 31: "taking a selfie", 32: "check time (from watch)", 33: "rub two hands", 34: "nod head/bow", 35: "shake head", 36: "wipe face", 37: "salute", 38: "put palms together", 39: "cross hands in front",
            40: "sneeze/cough", 41: "staggering", 42: "falling down", 43: "headache", 44: "chest pain", 45: "back pain", 46: "neck pain", 47: "nausea/vomiting", 48: "fan self", 49: "punch/slap",
            50: "kicking", 51: "pushing", 52: "pat on back", 53: "point finger", 54: "hugging", 55: "giving object", 56: "touch pocket", 57: "shaking hands", 58: "walking towards", 59: "walking apart",
            60: "hit with object", 61: "put on headphone", 62: "take off headphone", 63: "shoot at basket", 64: "bounce ball", 65: "tennis bat swing", 66: "juggle table tennis ball", 67: "hush", 68: "flick hair", 69: "thumb up",
            70: "thumb down", 71: "make OK sign", 72: "make victory sign", 73: "staple book", 74: "counting money", 75: "cutting nails", 76: "cutting paper", 77: "snap fingers", 78: "open bottle", 79: "sniff/smell",
            80: "squat down", 81: "toss a coin", 82: "fold paper", 83: "ball up paper", 84: "play magic cube", 85: "apply cream on face", 86: "apply cream on hand", 87: "put on bag", 88: "take off bag", 89: "put object into bag",
            90: "take object out of bag", 91: "open a box", 92: "move heavy objects", 93: "shake fist", 94: "throw up cap/hat", 95: "capitulate", 96: "cross arms", 97: "arm circles", 98: "arm swings", 99: "run on the spot",
            100: "butt kicks", 101: "cross toe touch", 102: "side kick", 103: "yawn", 104: "stretch oneself", 105: "blow nose", 106: "wield knife", 107: "knock over", 108: "grab stuff", 109: "shoot with gun",
            110: "step on foot", 111: "high-five", 112: "cheers and drink", 113: "carry object", 114: "take a photo", 115: "follow", 116: "whisper", 117: "exchange things", 118: "support somebody", 119: "rock-paper-scissors"
        }
    else :
        # Use natural language descriptions for NTU RGB+D actions
        ntu_action_labels = {
            0: "A person drinks water.", 1: "A person eats a meal.", 2: "A person brushes their teeth.", 3: "A person brushes their hair.", 4: "A person drops an object.", 5: "A person picks something up.", 6: "A person throws something.", 7: "A person sits down.", 8: "A person stands up.", 9: "A person claps their hands.",
            10: "A person reads.", 11: "A person writes.", 12: "A person tears up paper.", 13: "A person puts on a jacket.", 14: "A person takes off a jacket.", 15: "A person puts on a shoe.", 16: "A person takes off a shoe.", 17: "A person puts on glasses.", 18: "A person takes off glasses.", 19: "A person puts on a hat or cap.",
            20: "A person takes off a hat or cap.", 21: "A person cheers up.", 22: "A person waves their hand.", 23: "A person kicks something.", 24: "A person reaches into their pocket.", 25: "A person hops.", 26: "A person jumps up.", 27: "A person makes a phone call.", 28: "A person plays with a phone or tablet.", 29: "A person types on a keyboard.",
            30: "A person points to something.", 31: "A person takes a selfie.", 32: "A person checks the time on their watch.", 33: "A person rubs their hands together.", 34: "A person nods or bows.", 35: "A person shakes their head.", 36: "A person wipes their face.", 37: "A person salutes.", 38: "A person puts their palms together.", 39: "A person crosses their hands in front.",
            40: "A person sneezes or coughs.", 41: "A person staggers.", 42: "A person falls down.", 43: "A person has a headache.", 44: "A person experiences chest pain.", 45: "A person experiences back pain.", 46: "A person experiences neck pain.", 47: "A person feels nauseous or vomits.", 48: "A person fans themselves.", 49: "A person punches or slaps.",
            50: "A person kicks.", 51: "A person pushes something.", 52: "A person pats someone on the back.", 53: "A person points a finger.", 54: "A person hugs someone.", 55: "A person gives an object to someone.", 56: "A person touches their pocket.", 57: "A person shakes hands.", 58: "A person walks towards something.", 59: "A person walks away.",
            60: "A person hits something with an object.", 61: "A person puts on headphones.", 62: "A person takes off headphones.", 63: "A person shoots a basketball.", 64: "A person bounces a ball.", 65: "A person swings a tennis bat.", 66: "A person juggles a table tennis ball.", 67: "A person makes a hush gesture.", 68: "A person flicks their hair.", 69: "A person gives a thumbs up.",
            70: "A person gives a thumbs down.", 71: "A person makes an OK sign.", 72: "A person makes a victory sign.", 73: "A person staples a book.", 74: "A person counts money.", 75: "A person cuts their nails.", 76: "A person cuts paper.", 77: "A person snaps their fingers.", 78: "A person opens a bottle.", 79: "A person sniffs or smells something.",
            80: "A person squats down.", 81: "A person tosses a coin.", 82: "A person folds paper.", 83: "A person balls up paper.", 84: "A person plays with a magic cube.", 85: "A person applies cream on their face.", 86: "A person applies cream on their hand.", 87: "A person puts on a bag.", 88: "A person takes off a bag.", 89: "A person puts an object into a bag.",
            90: "A person takes an object out of a bag.", 91: "A person opens a box.", 92: "A person moves heavy objects.", 93: "A person shakes their fist.", 94: "A person throws a cap or hat upward.", 95: "A person capitulates.", 96: "A person crosses their arms.", 97: "A person performs arm circles.", 98: "A person swings their arms.", 99: "A person runs on the spot.",
            100: "A person does butt kicks.", 101: "A person touches opposite toes in a cross motion.", 102: "A person performs a side kick.", 103: "A person yawns.", 104: "A person stretches themselves.", 105: "A person blows their nose.", 106: "A person wields a knife.", 107: "A person knocks something over.", 108: "A person grabs something.", 109: "A person shoots with a gun.",
            110: "A person steps on someone's foot.",111: "A person gives a high-five.",112: "A person says cheers and drinks.",113: "A person carries an object.",114: "A person takes a photo.",115: "A person follows someone.",116: "A person whispers.",117: "A person exchanges things.",118: "A person supports somebody.",119: "A person plays rock-paper-scissors."
        }
    ntu_action_labels = [ntu_action_labels[i] for i in range(len(ntu_action_labels))]

    print("Encoding NTU RGB+D action labels...")
    ntu_label_embeds = encode_texts(ntu_action_labels, model, device)

    compute_knn_stats(
        ntu_label_embeds,
        ntu_action_labels,
        humanml3d_embeds,
        humanml3d_descriptions,
        args
    )