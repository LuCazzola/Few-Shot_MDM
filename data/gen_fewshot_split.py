import os
import argparse
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset directory (containing annotations/ and splits/)')
    parser.add_argument('--class-list', type=int, nargs='+', required=True,
                        help='List of class indices to include (e.g. 0 1 2 3 4)')
    parser.add_argument('--shots', type=int, required=True,
                        help='Number of samples per class in training splits')
    parser.add_argument('--eval-multiplier', type=int, default=5,
                        help='Defines size of validation sets as: shots * eval_multiplier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def extract_label_from_name(name: str) -> int:
    """Extract class index from NTU filename like S001C001P001R001A059 → 58"""
    try:
        return int(name[-4:].replace('A', '')) - 1
    except Exception:
        return -1

def compute_stats_from_frame_zero(filenames, annotations_dir, output_dir):
    """Compute mean/std from frame-0 across all sequences in [T, J*3] flattened format."""
    first_frames = []
    for fname in filenames:
        path = annotations_dir / f"{fname}.npy"
        if not path.exists():
            print(f"[WARNING] Missing annotation: {path}")
            continue
        data = np.load(path)  # Expecting [T, J*3]
        if data.shape[0] == 0:
            continue
        first_frames.append(data[0])  # shape [J*3]

    if not first_frames:
        print(f"[WARNING] No valid frame-0 data in {output_dir}")
        return

    first_frames = np.stack(first_frames, axis=0)  # [N, J*3]
    mean = np.mean(first_frames, axis=0)           # [J*3]
    std = np.std(first_frames, axis=0)             # [J*3]

    np.save(output_dir / 'Mean.npy', mean.astype(np.float32))
    np.save(output_dir / 'Std.npy', std.astype(np.float32))

    print(f"✅ Saved stats to {output_dir}/Mean.npy and Std.npy")

def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_dir = Path(args.dataset)
    annotations_dir = dataset_dir / 'annotations'
    if not annotations_dir.exists():
        raise FileNotFoundError(f"'annotations/' folder not found inside {args.dataset}")

    default_splits_dir = dataset_dir / 'splits' / 'default'
    fewshot_base = dataset_dir / 'splits' / 'fewshot' / f"{len(args.class_list)}way_{args.shots}shot_seed{args.seed}"
    fewshot_base.mkdir(parents=True, exist_ok=True)

    # Parse available default splits
    split_groups = defaultdict(dict)
    for splitname in os.listdir(default_splits_dir):
        split_folder = default_splits_dir / splitname
        if not split_folder.is_dir():
            continue
        for part in ['train', 'val']:
            split_file = split_folder / f"{part}.txt"
            if split_file.exists():
                split_groups[splitname][part] = split_file

    for splitname, parts in split_groups.items():
        split_output_dir = fewshot_base / splitname
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for split_type in ['train', 'val']:
            if split_type not in parts:
                continue

            split_file = parts[split_type]
            is_train = split_type == 'train'
            num_samples = args.shots if is_train else args.shots * args.eval_multiplier

            with open(split_file, 'r') as f:
                all_ids = [line.strip() for line in f if line.strip()]

            class_to_ids = defaultdict(list)
            for fid in all_ids:
                label = extract_label_from_name(fid)
                if label in args.class_list:
                    class_to_ids[label].append(fid)

            selected_ids = []
            for cls in args.class_list:
                ids = class_to_ids.get(cls, [])
                if len(ids) < num_samples:
                    print(f"⚠️ Not enough samples for class {cls} in {split_file.name}: requested {num_samples}, found {len(ids)}")
                selected_ids.extend(random.sample(ids, min(num_samples, len(ids))))

            output_path = split_output_dir / f"{split_type}.txt"
            with open(output_path, 'w') as f:
                for sid in selected_ids:
                    f.write(f"{sid}\n")

            print(f"📄 Saved {split_type}: {output_path}  ({len(selected_ids)} samples)")

            if is_train:
                compute_stats_from_frame_zero(selected_ids, annotations_dir, split_output_dir)

if __name__ == '__main__':
    main()
