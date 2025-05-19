import os
import json
import re
import argparse
import shutil
import numpy as np
from tqdm import tqdm

def extract_action_index(filename):
    """Extracts the zero-based action index from NTU filename."""
    match = re.search(r'A(\d{3})', filename)
    if not match:
        raise ValueError(f"Cannot extract action ID from {filename}")
    action_id = int(match.group(1))  # e.g. A001 -> 1
    return action_id - 1  # zero-based index

def annotate_with_pos_preserving_spacing(caption):
    # Capture words and all surrounding whitespace
    tokens = re.findall(r'\S+|\s+', caption.rstrip('.'))
    annotated = ""
    for token in tokens:
        if token.isspace():
            annotated += token  # Preserve exact spacing
        else:
            annotated += f"{token}/UNK"
    return annotated

def main(dataset, smpl_data):
    class_captions = os.path.join(dataset, 'class_captions.json')
    annotations_path = smpl_data
    smpl_data = os.path.join(smpl_data, 'annotations')
    
    texts_output_dir = os.path.join(dataset, 'texts')
    annots_output_dir = os.path.join(dataset, 'annotations')
    splits_output_root = os.path.join(dataset, 'splits', 'default')

    os.makedirs(texts_output_dir, exist_ok=True)
    os.makedirs(annots_output_dir, exist_ok=True)
    os.makedirs(splits_output_root, exist_ok=True)

    # Copy and route all .txt split files into splits/default/<splitname>/<splitpart>.txt
    split_files = [f for f in os.listdir(annotations_path) if f.endswith('.txt')]
    for split_file in tqdm(split_files, desc="Copying splits"):
        try:
            split_name, split_part = os.path.splitext(split_file)[0].rsplit('_', 1)
        except ValueError:
            print(f"⚠️ Skipping malformed split filename: {split_file}")
            continue

        dst_dir = os.path.join(splits_output_root, split_name)
        os.makedirs(dst_dir, exist_ok=True)

        src_path = os.path.join(annotations_path, split_file)
        dst_path = os.path.join(dst_dir, f"{split_part}.txt")
        shutil.copy2(src_path, dst_path)

    # Load caption JSON
    with open(class_captions, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)

    # List all .npy files
    npy_files = [f for f in os.listdir(smpl_data) if f.endswith('.npy')]
    pbar = tqdm(npy_files, desc="Processing samples")
    for filename in pbar:
        try:
            pbar.set_postfix_str(f"{filename}")
            action_idx = extract_action_index(filename)
            entry = captions_dict.get(str(action_idx))
            if entry is None:
                pbar.write(f"⚠️ Warning: No entry found for action index {action_idx}")
                continue

            # Generate and save caption text
            captions = entry['captions']
            formatted_lines = []
            for cap in captions:
                pos_string = annotate_with_pos_preserving_spacing(cap)
                formatted_lines.append(f"{cap}#{pos_string}#0.0#0.0")

            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(texts_output_dir, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as out_f:
                out_f.write("\n".join(formatted_lines))

            # Load, flatten, and save motion data
            motion_path = os.path.join(smpl_data, filename)
            data = np.load(motion_path)  # [T, J, D]
            if data.ndim != 3:
                pbar.write(f"⚠️ Skipping {filename}: expected [T, J, D], got {data.shape}")
                continue
            
            T, J, D = data.shape
            flattened = data.reshape(T, J * D).astype(np.float32)
            out_npy_path = os.path.join(annots_output_dir, filename)
            np.save(out_npy_path, flattened)

        except Exception as e:
            pbar.write(f"❌ Error processing {filename}: {e}")

    print(f"📄 Captions saved in:     {texts_output_dir}")
    print(f"💾 Annotations saved in:  {annots_output_dir}")
    print(f"🗂️  Split files copied to: {splits_output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to local dataset folder")
    parser.add_argument("--smpl_data", type=str, required=True, help="Path to folder Output of forward mapping")
    args = parser.parse_args()

    main(args.dataset, args.smpl_data)
