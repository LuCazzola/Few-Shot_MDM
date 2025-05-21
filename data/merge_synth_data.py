import os
import argparse
import pickle
import warnings

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def main(args):
    
    fewshot_dir = os.path.join(args.dataset, 'splits', 'fewshot', args.fewshot_split)
    fewshot_parent_split = os.path.basename(args.fewshot_split)  # e.g., 'xset'

    # Load few-shot filenames from train.txt
    with open(os.path.join(fewshot_dir, 'train.txt')) as f:
        fewshot_filenames = set(line.strip() for line in f)

    # Locate original .pkl file (the only .pkl in dataset folder)
    # Get all pkl files in the dataset directory
    original_pkl_path = os.path.join(args.dataset, [f for f in os.listdir(args.dataset) if f.endswith('.pkl')][-1])
    assert original_pkl_path is not None, f"No .pkl file found in {args.dataset}"
    
    print(f"Using {original_pkl_path} as reference dataset...")
    # Note: this is a bit of a hack, but we need to check if the dataset is formatted
    # A formatted dataset is one to which basic pre-processing has been applied, check main README.md
    assert original_pkl_path.endswith('_formatted.pkl'), f"You're using a non-formatted dataset. A formatted dataset is expected to be found in {args.dataset} folder"

    # Load original and synthetic data
    original_data = load_pickle(original_pkl_path)
    synth_data = load_pickle(args.synth_data)

    # Extract all relevant split keys
    all_splits = original_data['split'].keys()
    relevant_splits = [k for k in all_splits if k.startswith(fewshot_parent_split)]
    train_split = [s for s in relevant_splits if s.endswith('_train')][0]
    kept_filenames = set()
    for key in relevant_splits:
        kept_filenames.update(original_data['split'][key])

    # Index annotations by frame_dir
    frame_to_ann = {ann['frame_dir']: ann for ann in original_data['annotations']}
    synth_labels = set(ann['label'] for ann in synth_data['annotations'])

    # Filter annotations
    lr_counter = 0
    filtered_annotations = []
    for fname in kept_filenames: # within the fewshot split
        ann = frame_to_ann.get(fname)
        if ann is None:
            # Note: it's intentional to trigger on NTU60, as it contains also NTU120 data
            warnings.warn('Some annotations are never mentioned in Any split.')
            continue
        if ann['label'] not in synth_labels or fname in fewshot_filenames:
            # sample is not a low-shot class or it's present in the low-shot set (it's real data)
            if fname in fewshot_filenames: lr_counter += 1
            filtered_annotations.append(ann)
    assert lr_counter != len(kept_filenames), f"Not all low-resources data was found. Found {lr_counter} but {len(kept_filenames)} was specified"
    
    # Clean up
    original_data['annotations'] = filtered_annotations # annotations
    for s in all_splits: # irrelevant splits
        if s not in relevant_splits:
            original_data['split'][s] = []
    # Merge and update
    filtered_annotations.extend(synth_data['annotations']) # synth. data
    for s in relevant_splits:
        original_data['split'][s] = [
            f for f in original_data['split'][s]
            if f in fewshot_filenames or frame_to_ann.get(f,{}).get('label') not in synth_labels
        ]
    original_data['split'][train_split] += [ann['frame_dir'] for ann in synth_data['annotations']] # synth. data in right train_split

    # Save merged output
    out_path = os.path.join(fewshot_dir, f'merged_{fewshot_parent_split}.pkl')
    save_pickle(original_data, out_path)
    print(f"Merged data saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Name of dataset folder, e.g., NTU60')
    parser.add_argument('--fewshot_split', required=True, help='Relative path under splits/fewshot')
    parser.add_argument('--synth_data', required=True, help='Path to the synthetic data .pkl file')
    args = parser.parse_args()

    args.dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
    main(args)