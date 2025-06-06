import os
import argparse
import pickle
import json
import numpy as np
from tqdm import tqdm
import copy

from scripts.skel_adaptation import forward_map, resample_motion, align_motion

from utils.pos2red_feat import compute_redundant_motion_features
from utils.humanml3d.text_process import process_text
from utils.constants.data import DATA_FILENAME, CLASS_CAPTIONS_FILENAME, DATASET_FPS
from utils.constants.skel import JOINTS_2_DROP, FEET_THRE, FLOOR_THRE

def filter_data_consistency(data):
    """
    filters from dead instances
    """
    ann_frame_dirs = {ann['frame_dir'] for ann in data['annotations']}    
    split_frame_dirs = set(fd for split in data['split'].values() for fd in split)
    # Intersection: only keep entries present in both
    valid_frame_dirs = ann_frame_dirs & split_frame_dirs
    # Filter
    filtered_annotations = [ann for ann in data['annotations'] if ann['frame_dir'] in valid_frame_dirs]
    filtered_splits = {
        key: [fd for fd in data['split'][key] if fd in valid_frame_dirs]
        for key in data['split']
    }
    out_data = {
        'annotations': filtered_annotations,
        'split': filtered_splits
    }
    return out_data

def apply_forward(data, out_path, dataset, default_splits_path):
    """
    Application of forward mapping on the given PySkl dataset and stores files.
    """
    # Apply forward mapping and store files
    for ann in tqdm(data['annotations']):
        frame_dir = ann['frame_dir'].strip()
        keypoint = ann['keypoint']
        if keypoint.shape[0] == 0:
            continue
        
        # NOTE: We ignore multiple skeletons, ONLY KEEP THE FIRST ONE.
        # NOTE: naturally such action classes won't be used for MDM training.
        ntu_joints = keypoint[0] 
        
        # Map
        ntu_joints = resample_motion(ntu_joints, original_fps=DATASET_FPS[dataset], target_fps=DATASET_FPS['HML3D'])
        smpl_joints = forward_map(ntu_joints)
        smpl_joints = compute_redundant_motion_features(smpl_joints, floor_thre=FLOOR_THRE, feet_thre=FEET_THRE)  # (T_new-1, 263)

        np.save(os.path.join(out_path, f"{frame_dir}.npy"), smpl_joints)
    

    # Compute statistics for default splits
    for split in tqdm(os.listdir(default_splits_path), desc="Computing statistics for splits"):
        split_dir = os.path.join(default_splits_path, split)
        with open(os.path.join(split_dir, 'train.txt'), 'r') as f:
            fnames = [os.path.join(out_path, line.strip() + ".npy") for line in f.readlines()]
            motion = [np.load(fn) for fn in fnames]
        all_motions = np.concatenate(motion, axis=0).astype(np.float64)
        mean = all_motions.mean(axis=0)
        std = all_motions.std(axis=0)
        np.save(os.path.join(split_dir, f"{dataset.lower()}_mean.npy"), mean)
        np.save(os.path.join(split_dir, f"{dataset.lower()}_std.npy"), std)


def store_preprocessed_dataset(data, out_path, dataset):
    """
    Given the original NTU dataset, stores a formatted copy such that
    - it's sub-sampled to 20 FPS
    - uses 19 joints (excluding hands) instead of 25
    - TODO ?other?
    - Shifting according to frame Zero :
        - Root at (0, Y, 0) => XZ is horizontal plane
        - Floor at Y=0 => vertical displacement
    """
    data_copy = copy.deepcopy(data)
    for idx, ann in enumerate(tqdm(data_copy['annotations'])):
        motion = ann['keypoint'] # (N, T, 25, 3)
        resampled_joints = []
        
        # Transform
        displacement = None
        for i in range(motion.shape[0]):
            resampled = resample_motion(motion[i], original_fps=DATASET_FPS[dataset], target_fps=DATASET_FPS['HML3D'])  # (T_new, 25, 3)
            
            if i == 0 :
                resampled, displacement = align_motion(resampled)
            else:
                resampled, _ = align_motion(resampled, displacement=displacement)

            resampled = np.delete(resampled, list(JOINTS_2_DROP["KINECT"]), axis=1)  # (T_new, 19, 3)
            resampled_joints.append(resampled)
        # store back
        data_copy['annotations'][idx]['keypoint'] = np.stack(resampled_joints, axis=0)  # (N, T_new, 19, 3)
        data_copy['annotations'][idx]['total_frames'] = data_copy['annotations'][idx]['keypoint'].shape[1] # T_new
    # Save
    with open(out_path, 'wb') as f:
        pickle.dump(data_copy, f)


def format_default_splits(data, out_path):
    """
    Transcribes PySkl splits to MDM format (.txt)
    """
    framedir_2_label = {ann['frame_dir']: ann['label'] for ann in data['annotations']}
    
    for key in tqdm(data['split']):
        split_name, split_set = key.split('_')
        split_dir = os.path.join(out_path, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_framedirs = data['split'][key]
        # Store split file
        out_file_path = os.path.join(split_dir, f"{split_set}.txt")
        with open(out_file_path, 'w') as f:
            for item in split_framedirs:
                f.write(f"{item}\n")
        # Store label file
        out_label_path = os.path.join(split_dir, f"{split_set}_y.txt")
        with open(out_label_path, 'w') as f:
            for item in split_framedirs:
                f.write(f"{framedir_2_label[item]}\n")


def format_texts(data, out_path, class_captions):
    """
    Given a CLASS_CAPTIONS file, transcribes data into .txt files using HumanML3D POS tagging logic.
    """
    with open(class_captions, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)

    for sample in tqdm(data['annotations']):
        entry = captions_dict.get(str(sample['label']))
        assert entry is not None, (
            f"No entry in {class_captions} was found for action index {sample['label']}"
        )

        formatted_lines = []
        for cap in entry['captions']:
            word_list, pos_list = process_text(cap)
            pos_string = ' '.join(f'{word_list[i]}/{pos_list[i]}' for i in range(len(word_list)))
            formatted_lines.append(f"{cap}#{pos_string}#0.0#0.0")

        txt_path = os.path.join(out_path, f"{sample['frame_dir']}.txt")
        with open(txt_path, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(formatted_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--dataset", type=str, default='NTU60', choices=["NTU60", "NTU120"], help="dataset to setup")
    parser.add_argument("--force", action="store_true", help="Force application of all operations (overriding prev. files).")

    args = parser.parse_args()
    base_dataset_path = os.path.join("data", args.dataset)

    # 1.
    print(f"Loading {args.dataset} dataset...")
    data_path = os.path.join(base_dataset_path, DATA_FILENAME[args.dataset])
    assert os.path.exists(data_path), f"Input data {data_path} not found."
    with open(data_path, 'rb') as f:
        data = filter_data_consistency(pickle.load(f))

    # 2.
    print(f"\nFormatting default splits...")
    out_defsplit_path = os.path.join(base_dataset_path, "splits", "default")
    os.makedirs(out_defsplit_path, exist_ok=True)
    if args.force or not os.listdir(out_defsplit_path):
        format_default_splits(data, out_defsplit_path)
        print(f"Default splits formatted and stored at {out_defsplit_path}")
    else:
        print("Skipping formatting of default splits...")

    # 3.
    print(f"\nPre-processing dataset...")
    out_processed_data_path = os.path.join(
        base_dataset_path,
        DATA_FILENAME[args.dataset].split('.')[0] + '_preproc.' + DATA_FILENAME[args.dataset].split('.')[1]
    )
    os.makedirs(os.path.dirname(out_processed_data_path), exist_ok=True)
    if args.force or not os.path.exists(out_processed_data_path):
        store_preprocessed_dataset(data, out_processed_data_path, args.dataset)
        print(f"Pre-processed default {args.dataset} dataset stored at {out_processed_data_path}")
    else:
        print("Skipping pre-processing...")

    # 4.
    print(f"\nApplying forward mapping on {args.dataset} dataset...")
    out_annotations_path = os.path.join(base_dataset_path, "annotations")
    os.makedirs(out_annotations_path, exist_ok=True)
    if args.force or not os.listdir(out_annotations_path):
        apply_forward(data, out_annotations_path, args.dataset, out_defsplit_path)
        print(f"Forward mapping applied to {args.dataset} dataset stored at {out_annotations_path}")
    else:
        print("Skipping application of forward...")
    
    # 5.
    print(f"\nFormatting texts...")
    class_captions = os.path.join(base_dataset_path, CLASS_CAPTIONS_FILENAME)
    assert os.path.exists(class_captions), f"Input data {class_captions} not found."
    out_texts_path = os.path.join(base_dataset_path, "texts")
    os.makedirs(out_texts_path, exist_ok=True)
    if args.force or not os.listdir(out_texts_path):
        format_texts(data, out_texts_path, class_captions)
        print(f"Annotations formatted and stored at {out_texts_path}")
    else:
        print("Skipping formatting of annotations...")
    
    # Done
    print(f"\nDone! dataset {args.dataset} stored at {base_dataset_path} .")