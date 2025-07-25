import os
import argparse
import json
import pickle
import numpy as np
from os.path import join as pjoin

from scripts.data_prep import filter_data_consistency
from scripts.skel_adaptation.skel_mapping import backward_map

## FIXME !!! Script not updated in a WHILE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NTU60', choices=['NTU60'])
    parser.add_argument('--synth-data', type=str, required=True, 
                       help='Name of the synthetic data folder to merge with the few-shot split.')
    parser.add_argument('--fewshot-split-id', type=str, required=True, 
                       help='ID of the few-shot split to merge with the synthetic data.')
    parser.add_argument('--split', type=str, default='xset', choices=['xset', 'xsub', 'xview'], 
                       help='Split type to use for merging synthetic data. Default is "xset".')

    args = parser.parse_args()

    # Hardcoded paths
    base_dataset_path = pjoin("data", args.dataset)
    fewshot_base_path = pjoin(base_dataset_path, 'splits', 'fewshot')
    mdm_base_save_path = pjoin('external', 'motion-diffusion-model', 'save')

    # Convert synthetic data
    print(f"Converting data from {args.synth_data}...")
    fewshot_splits_path = pjoin(fewshot_base_path, args.fewshot_split_id)
    assert os.path.exists(fewshot_splits_path), f"FewShot split not found: {fewshot_splits_path}"

    with open(pjoin(fewshot_splits_path, "meta.json"), 'r') as f:
        meta_data = json.load(f)
    with open(pjoin(mdm_base_save_path, args.synth_data, 'results_cls.txt'), 'r') as f:
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
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, 'wb') as f:
        pickle.dump(pyskl_data, f)
    
    print(f"Done! Converted data saved at {out_dir}.")