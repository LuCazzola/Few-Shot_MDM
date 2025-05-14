import os
import json
import re
import argparse
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

def main(json_path, npy_folder):
    # Load caption JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)

    # Output folder
    output_folder = os.path.join(os.path.dirname(json_path), 'texts')
    os.makedirs(output_folder, exist_ok=True)

    # List all .npy files
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

    # Process each .npy file with progress bar
    for filename in tqdm(npy_files, desc="Generating caption text files"):
        try:
            action_idx = extract_action_index(filename)
            entry = captions_dict.get(str(action_idx))
            if entry is None:
                print(f"Warning: No entry found for action index {action_idx}")
                continue

            captions = entry['captions']
            formatted_lines = []
            for cap in captions:
                pos_string = annotate_with_pos_preserving_spacing(cap)
                formatted_lines.append(f"{cap}#{pos_string}#0.0#0.0")

            # Write to file
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, txt_filename)
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write("\n".join(formatted_lines))

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON caption file")
    parser.add_argument("--npy_folder", type=str, required=True, help="Path to the folder containing .npy files")
    args = parser.parse_args()

    main(args.json_path, args.npy_folder)
