import os
import argparse
import random
import numpy as np
from itertools import chain
import json
import pickle
from tqdm import tqdm

from setup import filter_data_consistency, DATA_FILENAME
from skel_adaptation.skel_mapping import backward_preprocess, backward_map


def compute_stats(data_names, data_path):
    """Compute mean/std w.r.t. frame-0 of motion data."""
    first_frames = []
    for fname in data_names:
        path = os.path.join(data_path, f"{fname}.npy")
        assert os.path.exists(path), f"Missing annotation: {path}"
        data = np.load(path)
        first_frames.append(data[0])
    assert first_frames, "No frames found in the provided filenames."

    first_frames = np.stack(first_frames, axis=0)
    return np.mean(first_frames, axis=0), np.std(first_frames, axis=0)


def parse_split_files(split_dir, split_sets):
    """Find all existing split .txt files under each split name and set."""
    split_names = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
    split_files = [
        os.path.join(split_dir, s_name, f"{s_set}.txt")
        for s_name in split_names for s_set in split_sets
        if os.path.exists(os.path.join(split_dir, s_name, f"{s_set}.txt"))
    ]
    return split_files, split_names


def process_split_file(split_file, class_list, N, annotations_path, out_dir):
    """Processes one split file: sampling, writing, and computing stats."""
    with open(split_file, 'r') as f:
        sample = [line.strip() for line in f]

    label_path = split_file.replace('.txt', '_y.txt')
    assert os.path.exists(label_path), f"Missing labels for {split_file}"
    with open(label_path, 'r') as f:
        label = [int(line.strip()) for line in f]

    # Group samples by class
    class_sample = [[] for _ in class_list]
    for idx, s in enumerate(sample):
        if label[idx] not in class_list:
            continue
        class_idx = class_list.index(label[idx])
        class_sample[class_idx].append(s)

    assert all(len(s) >= N for s in class_sample), "Some classes have fewer samples than requested."
    low_resource_data = [random.sample(s, N) for s in class_sample]

    # save split + labels
    split_name = os.path.splitext(os.path.basename(split_file))[0]
    with open(os.path.join(out_dir, f"{split_name}.txt"), 'w') as f:
        for s in low_resource_data:
            f.write('\n'.join(s) + '\n')
    with open(os.path.join(out_dir, f"{split_name}_y.txt"), 'w') as f:
        for class_idx, s in enumerate(low_resource_data):
            f.write('\n'.join([str(class_list[class_idx])] * len(s)) + '\n')

    # compute and save stats
    mean, std = compute_stats(list(chain.from_iterable(low_resource_data)), annotations_path)
    np.save(os.path.join(out_dir, 'Mean.npy'), mean)
    np.save(os.path.join(out_dir, 'Std.npy'), std)

def merge_split(data, split_names, out_path):
    """
    Merge few-shot splits with the full dataset.
    Keeps full entries for non-fewshot classes, and trims fewshot classes to sampled entries.
    """
    with open(os.path.join(out_path, 'meta.json'), 'r') as f:
        class_list = set(json.load(f)['class_list'])

    framedir_to_label = {ann['frame_dir']: ann['label'] for ann in data['annotations']}
    new_split = {}

    for split_name in split_names:
        split_dir = os.path.join(out_path, split_name)
        for subset in ['train', 'val', 'test']:
            
            key = f"{split_name}_{subset}"
            txt_path = os.path.join(split_dir, f"{subset}.txt")

            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    selected = set(x.strip() for x in f if x.strip())

                original = data['split'].get(key, [])
                new_split[key] = [x for x in original if framedir_to_label[x] not in class_list or x in selected]


    data['split'] = new_split
    data = filter_data_consistency(data)

    out_file = os.path.join(out_path, "pyskl_data.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

def convert_synth_data(data):
    pass

def create_unique_split_dir(base_dir, run_info):
    """Creates a new uniquely numbered subfolder and stores metadata."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [f for f in os.listdir(base_dir) if f.isdigit()]
    next_id = f"{max([int(x) for x in existing] + [-1]) + 1:04d}"
    run_dir = os.path.join(base_dir, next_id)
    os.makedirs(run_dir)

    with open(os.path.join(run_dir, "meta.json"), 'w') as f:
        json.dump(run_info, f, indent=2)

    return run_dir


if __name__ == '__main__':

    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument('--mode', type=str, default='generate', choices=['generate', 'convert'])
    mode_args, remaining = mode_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[mode_parser])
    parser.add_argument('--dataset', type=str, default='NTU60', choices=['NTU60', 'NTU120'])
    parser.add_argument('--seed', type=int, default=42)

    if mode_args.mode == 'generate':
        parser.add_argument('--class-list', type=int, nargs='+', required=True)
        parser.add_argument('--shots', type=int, default=10)
        parser.add_argument('--eval-multiplier', type=int, default=5)
    elif mode_args.mode == 'convert':
        parser.add_argument('--synth-data', type=str, required=True)
        parser.add_argument('--fewshot-split-id', type=str, required=True)
        parser.add_argument('--split', type=str, default='xset', choices=['xset', 'xsub', 'xview'])
    else:
        raise ValueError("Invalid mode.")

    args = parser.parse_args()
    random.seed(args.seed)

    # Hardcoded paths / data
    base_dataset_path = os.path.join("data", args.dataset)
    formatted_dataset_path = os.path.join(
        base_dataset_path,
        f"{DATA_FILENAME[args.dataset].split('.')[0]}_formatted.{DATA_FILENAME[args.dataset].split('.')[1]}"
    )
    annotations_path = os.path.join(base_dataset_path, 'annotations')
    default_splits_base_path = os.path.join(base_dataset_path, 'splits', 'default')
    fewshot_base_path = os.path.join(base_dataset_path, 'splits', 'fewshot')
    mdm_base_save_path = os.path.join('external', 'motion-diffusion-model', 'save')
    split_sets = ['train', 'val', 'test']

    # Differentate between modes
    if args.mode == 'generate':
        # In this mode, we generate a few-shot split from the formatted dataset.
        print(f"Generating few-shot split for {args.dataset} with {args.shots} shots per class.")
        run_metadata = {
            "class_list": args.class_list,
            "seed": args.seed,
            "shots": args.shots,
            "eval_multiplier": args.eval_multiplier
        }
        fewshot_splits_path = create_unique_split_dir(fewshot_base_path, run_metadata)
        os.makedirs(fewshot_splits_path, exist_ok=True)

        # 1. Parse split flies
        split_files, split_names = parse_split_files(default_splits_base_path, split_sets)
        assert split_files, "No default splits found."

        # 2. Sample from split files and save (lists + stats)
        for split_file in split_files:
            N = args.shots if split_file.endswith('train.txt') else args.shots * args.eval_multiplier
            split_name = os.path.basename(os.path.dirname(split_file))
            split_output_path = os.path.join(fewshot_splits_path, split_name)
            os.makedirs(split_output_path, exist_ok=True)
            process_split_file(split_file, args.class_list, N, annotations_path, split_output_path)

        # 3. Merge split with formatted dataset and save it
        with open(formatted_dataset_path, 'rb') as f:
            formatted_data = pickle.load(f)
        merge_split(formatted_data, split_names, fewshot_splits_path)
        
        print(f"Done! generated few-shot split at {fewshot_splits_path}.")
    
    elif mode_args.mode == 'convert':
        # In this mode, we merge synthetic data from MDM together with some few-shot split.
        print(f"Converting data from {args.synth_data}...")
        fewshot_splits_path = os.path.join(fewshot_base_path, args.fewshot_split_id)
        assert os.path.exists(fewshot_splits_path), f"FewShot split not found: {fewshot_splits_path}"

        with open(os.path.join(fewshot_splits_path, "meta.json"), 'rb') as f:
            meta_data = json.load(f)
        with open(os.path.join(mdm_base_save_path, args.synth_data, 'results_cls.txt'), 'rb') as f:
            label = [int(line.strip()) for line in f]
        
        # Check consistency between generated labels and specified fewshot split
        label_set = set(label)
        assert sorted(meta_data['class_list']) == sorted(label_set), "Generated samples labels do not match the fewshot split class list."

        # create new pkl within the correct split which merges the pyskl_data.pkl with the synthetic data, by updating accordingly the ['annotations'] and ['split'] keys
        with open(os.path.join(fewshot_splits_path, 'pyskl_data.pkl'), 'rb') as f:
            pyskl_data = pickle.load(f)
                
        # read the synth data
        synth_data = np.load(os.path.join(mdm_base_save_path, args.synth_data, 'results.npy'), allow_pickle=True).item()
        motion = synth_data['motion'].transpose(0, 3, 1, 2)
        lengths = synth_data['lengths']
        
        synth_data_conv = []
        for i, (motion_data, length) in enumerate(zip(motion, lengths)):
            motion_data = motion_data[:length, :, :]
            motion_data = backward_preprocess(motion_data)
            motion_data = backward_map(motion_data)
            
            synth_data_conv.append({
                'frame_dir': f"SYN{i:04d}A{label[i]+1:03d}",
                'label': label[i],
                'keypoint': np.expand_dims(motion_data, axis=0), # Add extra dimension to mimic number of skeletons
                'total_frames': length
            })
        
        # merge the two datasets
        pyskl_data['annotations'] += synth_data_conv
        pyskl_data['split'][str(args.split+'_train')] += [data['frame_dir'] for data in synth_data_conv]
        # remove redundant data
        allowed_keys = {f"{args.split}_train", f"{args.split}_val", f"{args.split}_test"}
        keys_to_clear = [key for key in pyskl_data['split'].keys() if key not in allowed_keys]
        for key in keys_to_clear:
            pyskl_data['split'][key] = []
        pyskl_data = filter_data_consistency(pyskl_data)
        # save the merged data
        out_dir = os.path.join(fewshot_splits_path, args.split, 'pyskl_data_wsyn.pkl')
        with open(out_dir, 'wb') as f:
            pickle.dump(pyskl_data, f)
        
        print(f"Done! Converted data saved at {out_dir}.")