import os
import argparse
import random
import numpy as np
from itertools import chain
import json
import pickle
from tqdm import tqdm
from os.path import join as pjoin

from scripts.setup import filter_data_consistency
from utils.constants import DATA_FILENAME, DATA_NUM_CLASSES, DATA_IGNORE_CLASSES
from scripts.skel_adaptation.skel_mapping import backward_map

def compute_stats(data_names, data_path):
    """Compute mean/std w.r.t. frame-0 of motion data."""
    first_frames = []
    for fname in data_names:
        path = pjoin(data_path, f"{fname}.npy")
        assert os.path.exists(path), f"Missing annotation: {path}"
        data = np.load(path)
        if np.isnan(data).any():
            print(f"{fname}.npy contains NaNs")
        first_frames.append(data[0])
    assert first_frames, "No frames found in the provided filenames."

    first_frames = np.stack(first_frames, axis=0)
    mean = np.mean(first_frames, axis=0)
    std = np.std(first_frames, axis=0)
    return mean, std


def parse_split_files(split_dir, split_sets):
    """Find all existing split .txt files under each split name and set."""
    split_names = [f for f in os.listdir(split_dir) if os.path.isdir(pjoin(split_dir, f))]
    split_files = [
        pjoin(split_dir, s_name, f"{s_set}.txt")
        for s_name in split_names for s_set in split_sets
        if os.path.exists(pjoin(split_dir, s_name, f"{s_set}.txt"))
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
    with open(pjoin(out_dir, f"{split_name}.txt"), 'w') as f:
        for s in low_resource_data:
            f.write('\n'.join(s) + '\n')
    with open(pjoin(out_dir, f"{split_name}_y.txt"), 'w') as f:
        for class_idx, s in enumerate(low_resource_data):
            f.write('\n'.join([str(class_list[class_idx])] * len(s)) + '\n')

    # compute and save stats
    mean, std = compute_stats(list(chain.from_iterable(low_resource_data)), annotations_path)
    np.save(pjoin(out_dir, 'Mean.npy'), mean)
    np.save(pjoin(out_dir, 'Std.npy'), std)

def merge_split(data, split_names, out_path):
    """
    Merge few-shot splits with the full dataset.
    Keeps full entries for non-fewshot classes, and trims fewshot classes to sampled entries.
    """
    with open(pjoin(out_path, 'meta.json'), 'r') as f:
        class_list = set(json.load(f)['class_list'])

    framedir_to_label = {ann['frame_dir']: ann['label'] for ann in data['annotations']}
    new_split = {}

    for split_name in split_names:
        split_dir = pjoin(out_path, split_name)
        for subset in ['train', 'val', 'test']:
            
            key = f"{split_name}_{subset}"
            txt_path = pjoin(split_dir, f"{subset}.txt")

            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    selected = set(x.strip() for x in f if x.strip())

                original = data['split'].get(key, [])
                new_split[key] = [x for x in original if framedir_to_label[x] not in class_list or x in selected]


    data['split'] = new_split
    data = filter_data_consistency(data)

    out_file = pjoin(out_path, "pyskl_data.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def create_unique_split_dir(base_dir, run_info):
    """Creates a new uniquely numbered subfolder (with 'S' prefix) and stores metadata."""
    os.makedirs(base_dir, exist_ok=True)
    existing = []
    for f in os.listdir(base_dir):
        if f.startswith('S') and f[1:].isdigit():
            existing.append(int(f[1:]))
    next_id = f"{(max(existing) + 1) if existing else 0:04d}"
    run_dir = pjoin(base_dir, f"S{next_id}")
    os.makedirs(run_dir)
    with open(pjoin(run_dir, "meta.json"), 'w') as f:
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
        parser.add_argument('--class-list', type=int, nargs='+', default=None, help='List of classes to include in the few-shot split. If not provided, all classes (except black-listed ones) will be used.')
        parser.add_argument('--shots', type=int, default=10, help='Number of shots per class for the few-shot split.')
        parser.add_argument('--eval-multiplier', type=int, default=5, help='multiplier for the number of shots in evaluation splits (val/test).')
    elif mode_args.mode == 'convert':
        parser.add_argument('--synth-data', type=str, required=True, help='Name of the synthetic data folder to merge with the few-shot split.')
        parser.add_argument('--fewshot-split-id', type=str, required=True, help='ID of the few-shot split to merge with the synthetic data.')
        parser.add_argument('--split', type=str, default='xset', choices=['xset', 'xsub', 'xview'], help='Split type to use for merging synthetic data. Default is "xset".')
    else:
        raise ValueError("Invalid mode.")

    args = parser.parse_args()
    random.seed(args.seed)

    # Hardcoded paths / data
    base_dataset_path = pjoin("data", args.dataset)
    formatted_dataset_path = pjoin(
        base_dataset_path,
        f"{DATA_FILENAME[args.dataset].split('.')[0]}_preproc.{DATA_FILENAME[args.dataset].split('.')[1]}"
    )
    annotations_path = pjoin(base_dataset_path, 'new_joint_vecs')
    default_splits_base_path = pjoin(base_dataset_path, 'splits', 'default')
    fewshot_base_path = pjoin(base_dataset_path, 'splits', 'fewshot')
    mdm_base_save_path = pjoin('external', 'motion-diffusion-model', 'save')
    split_sets = ['train', 'val', 'test']


    if args.class_list is None:
        # If no class list is provided, use all classes except ignored ones
        args.class_list = [i for i in range(DATA_NUM_CLASSES[args.dataset]) if i not in DATA_IGNORE_CLASSES[args.dataset]]
        print(f"No class list provided, using all classes except ignored ones:\n{args.class_list}")

    assert not (set(args.class_list) & set(DATA_IGNORE_CLASSES[args.dataset])), "Class list contains ignored classes. Please remove them from the class list."
    args.class_list = sorted(set(args.class_list))        

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
        print(f"Parsing split files...")
        for split_file in tqdm(split_files):
            N = args.shots if split_file.endswith('train.txt') else args.shots * args.eval_multiplier
            split_name = os.path.basename(os.path.dirname(split_file))
            split_output_path = pjoin(fewshot_splits_path, split_name)
            os.makedirs(split_output_path, exist_ok=True)
            process_split_file(split_file, args.class_list, N, annotations_path, split_output_path)

        # 3. Merge split with formatted dataset and save it
        with open(formatted_dataset_path, 'rb') as f:
            formatted_data = pickle.load(f)
        print(f"Merging splits with formatted dataset from {formatted_dataset_path}...")
        merge_split(formatted_data, split_names, fewshot_splits_path)
        
        print(f"Done! generated few-shot split at {fewshot_splits_path}.")
    
    elif mode_args.mode == 'convert':
        # In this mode, we merge synthetic data from MDM together with some few-shot split.
        print(f"Converting data from {args.synth_data}...")
        fewshot_splits_path = pjoin(fewshot_base_path, args.fewshot_split_id)
        assert os.path.exists(fewshot_splits_path), f"FewShot split not found: {fewshot_splits_path}"

        with open(pjoin(fewshot_splits_path, "meta.json"), 'rb') as f:
            meta_data = json.load(f)
        with open(pjoin(mdm_base_save_path, args.synth_data, 'results_cls.txt'), 'rb') as f:
            label = [int(line.strip()) for line in f]
        
        # Check consistency between generated labels and specified fewshot split
        label_set = set(label)
        assert sorted(meta_data['class_list']) == sorted(label_set), "Generated samples labels do not match the fewshot split class list."

        # create new pkl within the correct split which merges the pyskl_data.pkl with the synthetic data, by updating accordingly the ['annotations'] and ['split'] keys
        with open(pjoin(fewshot_splits_path, 'pyskl_data.pkl'), 'rb') as f:
            pyskl_data = pickle.load(f)
                
        # read the synth data
        synth_data = np.load(pjoin(mdm_base_save_path, args.synth_data, 'results.npy'), allow_pickle=True).item()
        motion = synth_data['motion'].transpose(0, 3, 1, 2)
        lengths = synth_data['lengths']
        
        synth_data_conv = []
        for i, (motion_data, length) in enumerate(zip(motion, lengths)):
            motion_data = motion_data[:length, :, :]
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
        out_dir = pjoin(fewshot_splits_path, args.split, 'pyskl_data_wsyn.pkl')
        with open(out_dir, 'wb') as f:
            pickle.dump(pyskl_data, f)
        
        print(f"Done! Converted data saved at {out_dir}.")