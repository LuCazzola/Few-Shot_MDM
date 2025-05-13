import os
import argparse
import random
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=str, required=True, help='Path to the forw/ directory (with .txt and annotations/)')
    parser.add_argument('--class-list', type=int, nargs='+', required=True, help='List of class indices to include (e.g. 0 1 2 3 4)')
    parser.add_argument('--shots', type=int, required=True, help='Number of samples per class in training splits')
    parser.add_argument('--eval-multiplier', type=int, default=5, help='Multiplier for eval splits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset-dir', type=str, default='NTU_RGBD', help='Dataset directory (default: NTU_RGBD)')
    return parser.parse_args()

def extract_label_from_name(name: str) -> int:
    """Extract class index from NTU filename like S001C001P001R001A059 → 58"""
    try:
        return int(name[-4:].replace('A', '')) - 1
    except Exception:
        return -1

def main():
    args = parse_args()
    random.seed(args.seed)

    split_files = [f for f in os.listdir(args.input_root) if f.endswith('.txt')]
    for split_file in split_files:
        split_name = split_file.replace('.txt', '')
        is_train = split_name.endswith('_train')
        num_samples = args.shots if is_train else args.shots * args.eval_multiplier

        with open(os.path.join(args.input_root, split_file), 'r') as f:
            all_ids = [line.strip() for line in f if line.strip()]

        # Organize by class
        class_to_ids = defaultdict(list)
        for fid in all_ids:
            label = extract_label_from_name(fid)
            if label in args.class_list:
                class_to_ids[label].append(fid)

        selected_ids = []
        for cls in args.class_list:
            ids = class_to_ids.get(cls, [])
            if len(ids) < num_samples:
                print(f"⚠️ Not enough samples for class {cls} in {split_name}: requested {num_samples}, found {len(ids)}")
            selected_ids.extend(random.sample(ids, min(num_samples, len(ids))))

        output_path = os.path.join(args.dataset_dir, "fewshot_data", f"{len(args.class_list)}way_{args.shots}shot_seed{args.seed}", f"{split_name}.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for sid in selected_ids:
                f.write(f"{sid}\n")
        print(f"✅ Wrote {len(selected_ids)} samples to {output_path}")

if __name__ == '__main__':
    main()
