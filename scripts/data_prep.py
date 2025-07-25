import numpy as np
import torch
import os
import json
import argparse
import pickle
import copy

from os.path import join as pjoin
from types import SimpleNamespace
from tqdm import tqdm

from scripts.skel_adaptation import forward_map, resample_motion, align_motion
from utils.humanml3d import process_text, motion_2_hml_vec, cal_mean_variance, recover_from_ric
from utils.constants.data import DATA_FILENAME, DATA_IGNORE_CLASSES, ACTION_CAPTIONS_FILENAME, DATASET_FPS
from utils.constants.skel import SKEL_INFO, JOINTS_2_DROP, FEET_THRE, FLOOR_THRE

def filter_data_consistency(data):
    """
    filters from dead instances and remove them
    """
    ann_frame_dirs = set(ann['frame_dir'] for ann in data['annotations'])
    split_frame_dirs = set(fd for split in data['split'].values() for fd in split)
    # Intersect
    valid_frame_dirs = ann_frame_dirs & split_frame_dirs
    # Filter
    filtered_annotations = [
        ann for ann in data['annotations']
        if ann['frame_dir'] in valid_frame_dirs
    ]
    filtered_splits = {
        key: [fd for fd in data['split'][key]
        if fd in valid_frame_dirs]
        for key in data['split']
    }
    return {'annotations': filtered_annotations, 'split': filtered_splits}

def get_samples_to_skip(data, outliers):
    """
    Returns 2 sets of frame_dir (name of samples) to skip based on some criteria.
    1st set (SKIP_LIST) contains data that should be ignored completely due to:
    - Instances with multiple skeletons (e.g., mutual actions)
    - Instances belonging to classes in DATA_IGNORE_CLASSES
    - Instances with NaN values in keypoints

    2nd set : (OUTLIER_LIST) : contains samples that should be ignored when computing statistics
    - Corrupted instances due to high noise or other issues
    - Such list is the outcome of a statistical analysis of the dataset (uses IQR cutoff on motion properties)

    Data from such lists is excluded during the generative model's training, altho it
    will be included during the classifier training for consistency.

    in the case of samples in BLACKLIST the forward mapping is still computed and stored
    to allow statistics computation on the whole dataset.
    """
    skip_names = set()
    for ann in data['annotations']:
        frame_dir = ann['frame_dir'].strip()
        keypoint = ann['keypoint']
        if (ann['label'] in DATA_IGNORE_CLASSES[DATASET] or # label in ignore classes
            keypoint.shape[0] > 1 or # presence of multiple skeletons
            np.isnan(keypoint).any()): # nans in keypoints (for safety) 
            # add
            skip_names.add(frame_dir)
    
    outlier_names = set()
    if outliers != '':
        if os.path.exists(outliers):
            with open(outliers, 'r') as f:
                outlier_names = set(line.strip() for line in f.readlines())
        else:
            print(f"WARNING: Requested to exclude outliers, but {outliers} does not exist.")

    return skip_names, outlier_names

def apply_forward(data, out_path, default_splits_path):
    """
    Application of forward mapping on the given PySkl dataset and stores files.
    """

    # Apply forward mapping and store files
    for ann in tqdm(data['annotations']):
        frame_dir = ann['frame_dir'].strip()
        keypoint = ann['keypoint']
        
        if frame_dir in SKIP_LIST: # avoid forw() for samples in SKIP_LIST (still parse OUTLIER_LIST)
            continue
        ntu_joints = keypoint[0] # (1, T, 25 ,3) -> (T, 25, 3)

        # Map
        ntu_joints = resample_motion(ntu_joints, original_fps=DATASET_FPS[DATASET], target_fps=DATASET_FPS['HML3D'])
        smpl_joints = forward_map(ntu_joints)
        # Preprocess
        new_joint_vecs = motion_2_hml_vec(smpl_joints, floor_thre=FLOOR_THRE, feet_thre=FEET_THRE)  # (T_new-1, 263)
        new_joints = recover_from_ric(torch.from_numpy(new_joint_vecs).unsqueeze(0).float(), SKEL_INFO["smpl"].joints_num).squeeze().numpy() # (T_new-1, 22, 3)
        new_joints = new_joints.reshape(new_joints.shape[0], -1) # (T_new-1, 22*3)
        # Store
        assert not np.isnan(new_joint_vecs).any() and not np.isnan(new_joints).any(), f"NaN values found in joint vectors for {frame_dir}."
        np.save(pjoin(out_path.joint_vecs, f"{frame_dir}.npy"), new_joint_vecs)
        np.save(pjoin(out_path.joints, f"{frame_dir}.npy"), new_joints)

    
    # Compute statistics for default splits
    for split in tqdm(os.listdir(default_splits_path), desc="Computing statistics for splits"):
        split_dir = pjoin(default_splits_path, split)
        for attr in ['joint_vecs', 'joints']:
            datapath = getattr(out_path, attr)
            with open(pjoin(split_dir, 'train.txt'), 'r') as f:
                fnames = [line.strip() for line in f.readlines()]
            fnames = [pjoin(datapath, fn + ".npy") for fn in fnames if fn not in BLACKLIST]
            motion = np.concatenate([np.load(fn) for fn in fnames], axis=0) # (N, T_new-1, 22*3)
            # Compute mean and std
            if attr == 'joint_vecs':
                # For hml vec. representation
                mean, std = cal_mean_variance(motion, SKEL_INFO["smpl"].joints_num)
            else :
                # for general representation (e.g., xyz)
                mean = np.mean(motion, axis=0)
                std = np.std(motion, axis=0)
            np.save(pjoin(split_dir, f"Mean_{attr}.npy"), mean)
            np.save(pjoin(split_dir, f"Std_{attr}.npy"), std)


def store_preprocessed_dataset(data, out_path):
    """
    Given the original NTU dataset, stores a formatted copy such that
    - it's sub-sampled to 20 FPS
    - uses 19 joints (excluding hands) instead of 25
    - Shifting according to frame Zero :
        - Root at (0, Y, 0) => XZ is horizontal plane
        - Floor at Y=0 => vertical displacement
    """
    data_copy = copy.deepcopy(data)
    for idx, ann in enumerate(tqdm(data_copy['annotations'])):
        motion = ann['keypoint'] # (N, T, 25, 3)
        new_joints = []

        # Transform
        displacement = None
        for i in range(motion.shape[0]):
            resampled = resample_motion(motion[i], original_fps=DATASET_FPS[DATASET], target_fps=DATASET_FPS['HML3D'])  # (T_new, 25, 3)
            
            # NOTE: motion alignment is likely redundant, as in PySKL preprocessing includes
            # alignment over spine direction and centering based on skel. gravity center
            
            if i == 0 : # Alignment is computed w.r.t. the first skeleton
                resampled, displacement = align_motion(resampled, displacement=None)
            else:
                resampled, _ = align_motion(resampled, displacement=displacement)

            resampled = np.delete(resampled, list(JOINTS_2_DROP["kinect"]), axis=1)  # (T_new, 19, 3)
            new_joints.append(resampled)
        # store back
        data_copy['annotations'][idx]['keypoint'] = np.stack(new_joints, axis=0)  # (N, T_new, 19, 3)
        data_copy['annotations'][idx]['total_frames'] = data_copy['annotations'][idx]['keypoint'].shape[1] # T_new
    # Save
    with open(out_path, 'wb') as f:
        pickle.dump(data_copy, f)


def format_default_splits(data, out_path):
    """
    Transcribes PySkl splits to MDM format (.txt)
    """
    framedir_2_label = {ann['frame_dir'] : ann['label'] for ann in data['annotations']}

    for key in tqdm(data['split']):
        split_name, split_set = key.split('_') # e.g. xset_train, xsub_test, ...
        split_dir = pjoin(out_path, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_framedirs = data['split'][key]
        # Store split file
        out_file_path = pjoin(split_dir, f"{split_set}.txt")
        with open(out_file_path, 'w') as f:
            for item in split_framedirs:
                if item not in BLACKLIST:
                    f.write(f"{item}\n")
        # Store label file
        out_label_path = pjoin(split_dir, f"{split_set}_y.txt")
        with open(out_label_path, 'w') as f:
            for item in split_framedirs:
                if item not in BLACKLIST:
                    f.write(f"{framedir_2_label[item]}\n")

def format_texts(data, out_path, action_captions):
    """
    Given a action_captions file, transcribes data into .txt files using HumanML3D POS tagging logic.
    """
    with open(action_captions, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)

    for sample in tqdm(data['annotations']):

        if sample['frame_dir'] in SKIP_LIST: # skip samples in SKIP_LIST (still parse OUTLIER_LIST)
            continue
        
        entry = captions_dict.get(str(sample['label']))
        assert entry is not None, "No entry in {} was found for action index {}".format(action_captions, sample['label'])

        formatted_lines = []
        for cap in entry['captions']:
            cap = cap.lower() # Lowercase the caption
            word_list, pos_list = process_text(cap)
            pos_string = ' '.join(f'{word_list[i]}/{pos_list[i]}' for i in range(len(word_list)))
            formatted_lines.append(f"{cap}#{pos_string}#0.0#0.0")

        txt_path = pjoin(out_path, f"{sample['frame_dir']}.txt")
        with open(txt_path, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(formatted_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="setup options")
    parser.add_argument("--dataset", type=str, default='NTU60', choices=["NTU60"], help="dataset from PySKL to process")
    parser.add_argument("--include-outliers", action="store_true", help="If unset (default), uses the local outliers.txt file to exclude outliers from the dataset.")
    args = parser.parse_args()

    global DATASET, SKIP_LIST, OUTLIER_LIST, BLACKLIST
    DATASET = args.dataset

    # 1.
    print(f"\nLoading {DATASET} dataset...")
    base_dataset_path = pjoin("data", DATASET)
    data_path = pjoin(base_dataset_path, DATA_FILENAME[DATASET])
    assert os.path.exists(data_path), "Input data {} not found.".format(data_path)
    with open(data_path, 'rb') as f:
        data = filter_data_consistency(pickle.load(f))
    outliers_path = '' if args.include_outliers else pjoin(base_dataset_path, "outliers.txt")
    SKIP_LIST, OUTLIER_LIST = get_samples_to_skip(data, outliers_path)
    BLACKLIST = SKIP_LIST | OUTLIER_LIST

    # 2.
    print("\nFormatting default splits...")
    out_defsplit_path = pjoin(base_dataset_path, "splits", "default")
    os.makedirs(out_defsplit_path, exist_ok=True)
    # .
    format_default_splits(data, out_defsplit_path)
    print(f"Default splits formatted and stored at {out_defsplit_path}")
    
    # 3.
    print("\nPre-processing dataset...")
    out_processed_data_path = pjoin(
        base_dataset_path,
        DATA_FILENAME[DATASET].replace('.pkl', '_preproc.pkl')
    )
    os.makedirs(os.path.dirname(out_processed_data_path), exist_ok=True)
    # .
    store_preprocessed_dataset(data, out_processed_data_path)
    print(f"Pre-processed default {DATASET} dataset stored at {out_processed_data_path}")

    # 4.
    print(f"\nApplying forward mapping on {DATASET} dataset...")
    out_annotations_path = SimpleNamespace(
        joints = pjoin(base_dataset_path, "new_joints"),
        joint_vecs = pjoin(base_dataset_path, "new_joint_vecs")
    )
    os.makedirs(out_annotations_path.joints, exist_ok=True)
    os.makedirs(out_annotations_path.joint_vecs, exist_ok=True)
    # .
    apply_forward(data, out_annotations_path, out_defsplit_path)
    print(f"Forward mapping applied to {DATASET}")
    print(f"{out_annotations_path.joint_vecs} : joint vector representations (hml_vec format)")
    print(f"{out_annotations_path.joints} : joint position vectors (xyz)")
    
    # 5.
    print(f"\nFormatting texts...")
    action_captions = pjoin(base_dataset_path, ACTION_CAPTIONS_FILENAME)
    assert os.path.exists(action_captions), f"Input data {action_captions} not found."
    out_texts_path = pjoin(base_dataset_path, "texts")
    os.makedirs(out_texts_path, exist_ok=True)
    # .
    format_texts(data, out_texts_path, action_captions)
    print(f"Annotations formatted and stored at {out_texts_path}")
    
    # Done
    print(f"\nDone! dataset {DATASET} stored at {base_dataset_path} .")