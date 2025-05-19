import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

from scipy.interpolate import interp1d

NTU_TO_SMPL_DIRECT_MAP = {
    0: 0, 20: 9, 2: 12, 3: 15, 4: 16, 5: 18, 6: 20,
    8: 17, 9: 19, 10: 21, 12: 1, 13: 4, 14: 7, 15: 10,
    16: 2, 17: 5, 18: 8, 19: 11,
}

def resample_motion(motion: np.ndarray, original_fps: int = 30, target_fps: int = 20) -> np.ndarray:
    """
    Subsample motion to target_fps.
    """
    T = motion.shape[0]
    original_times = np.linspace(0, T / original_fps, T)
    target_T = int(T * target_fps / original_fps)
    target_times = np.linspace(0, T / original_fps, target_T)
    interp_fn = interp1d(original_times, motion, axis=0, kind='linear')
    return interp_fn(target_times)

def forward_map(ntu_joints: np.ndarray):
    """
    Map NTU joints -> SMPL joints.
    Returns: (T, 22, 3) joints: full HumanML3D compatible skeletons
    """
    T = ntu_joints.shape[0]
    smpl_joints = np.zeros((T, 22, 3), dtype=np.float16)

    for ntu_idx, smpl_idx in NTU_TO_SMPL_DIRECT_MAP.items():
        smpl_joints[:, smpl_idx, :] = ntu_joints[:, ntu_idx, :].astype(np.float16)

    spine_base = ntu_joints[:, 0, :]
    spine_mid  = ntu_joints[:, 1, :]
    chest      = ntu_joints[:, 20, :]

    smpl_joints[:, 3, :] = (spine_base + spine_mid) / 2.0  # spine1
    smpl_joints[:, 6, :] = (spine_mid + chest) / 2.0       # spine2

    neck = ntu_joints[:, 2, :]
    left_shoulder = ntu_joints[:, 4, :]
    right_shoulder = ntu_joints[:, 8, :]

    smpl_joints[:, 13, :] = 0.75 * neck + 0.25 * left_shoulder
    smpl_joints[:, 14, :] = 0.75 * neck + 0.25 * right_shoulder

    return smpl_joints

def backward_map(smpl_joints: np.ndarray):
    """
    Map NTU joints -> SMPL joints.
    Returns : (T, 19, 3) as hand-related joints are dropped
    """
    T = smpl_joints.shape[0]
    ntu_joints = np.zeros((T, 25, 3), dtype=np.float16)

    # Drop hand-related joints
    OMIT_JOINTS = {7, 11, 21, 22, 23, 24}
    # Reverse the direct mapping (collar-bone is implicitly dropped)
    smpl_to_ntu_map = {v: k for k, v in NTU_TO_SMPL_DIRECT_MAP.items() if k not in OMIT_JOINTS and k != 1}
    for smpl_idx, ntu_idx in smpl_to_ntu_map.items():
        ntu_joints[:, ntu_idx, :] = smpl_joints[:, smpl_idx, :]

    # Reconstruct NTU joint 1 (spine_mid) = midpoint of spine1 (3) and spine2 (6)
    spine1 = smpl_joints[:, 3, :]
    spine2 = smpl_joints[:, 6, :]
    ntu_joints[:, 1, :] = 0.5 * (spine1 + spine2)

    # Drop un-used joint dimensions from npy array
    ntu_joints = np.delete(ntu_joints, list(OMIT_JOINTS), axis=1) # new shape (T, 19, 3)

    return ntu_joints

def forward_preprocess(joints: np.ndarray):
    # Normalize root joint to [0, Y, 0] at all frames
    root_x = joints[:, 0, 0:1]  # shape (T, 1)
    root_z = joints[:, 0, 2:3]  # shape (T, 1)
    joints[:, :, 0] -= root_x  # X centered
    joints[:, :, 2] -= root_z  # Z centered
    return joints

def backward_preprocess(joints: np.ndarray):
    # Example placeholder: does nothing yet
    return joints

def main_forward(args):
    with open(args.input_data, 'rb') as f:
        raw_data = pickle.load(f)

    split = raw_data.get('split', {})
    annotations = raw_data['annotations']

    os.makedirs(os.path.join(args.out_dir, 'forw', 'annotations'), exist_ok=True)
    split_lists = {k: [] for k in split}

    saved_count = 0
    for ann in tqdm(annotations, desc="Processing annotations"):
        frame_dir = ann['frame_dir'].strip()
        keypoint = ann['keypoint']
        if keypoint.shape[0] == 0:
            continue

        ntu_joints = keypoint[0].astype(np.float16)
        ntu_joints = resample_motion(ntu_joints, original_fps=30, target_fps=20)
        smpl_joints = forward_map(ntu_joints)
        smpl_joints = forward_preprocess(smpl_joints)

        out_path = os.path.join(args.out_dir, 'forw', 'annotations', f"{frame_dir}.npy")
        np.save(out_path, smpl_joints)
        saved_count += 1

    for key in split.keys():
        split_txt_path = os.path.join(args.out_dir, 'forw', f"{key}.txt")
        with open(split_txt_path, 'w') as f:
            for fid in split[key]:
                f.write(f"{fid}\n")

    print(f"✅ Saved {saved_count} sequences to {args.out_dir}/forw/annotations/")
    print(f"📄 Generated split lists: {', '.join(split_lists.keys())}")


def main_backward(args):
    split = {}
    annotations = []

    # Load all split files from input-data root (e.g., xsub_train.txt)
    txt_splits = [f for f in os.listdir(args.input_data) if f.endswith('.txt')]
    for txt_file in txt_splits:
        split_name = txt_file.replace('.txt', '')
        with open(os.path.join(args.input_data, txt_file), 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        split[split_name] = ids

    # Invert: frame_dir → split_name
    frame_to_split = {fid: split_name for split_name, ids in split.items() for fid in ids}

    annotation_dir = os.path.join(args.input_data, 'annotations')
    npy_files = [f for f in os.listdir(annotation_dir) if f.endswith('.npy')]

    for npy_file in tqdm(npy_files, desc="Reconstructing annotations"):
        sample_name = npy_file.replace('.npy', '')
        npy_path = os.path.join(annotation_dir, npy_file)

        smpl_joints = np.load(npy_path).astype(np.float16)
        smpl_joints = backward_preprocess(smpl_joints)
        ntu_joints = backward_map(smpl_joints)

        try:
            label = int(sample_name[-4:].replace('A', '')) - 1
        except ValueError:
            print(f"⚠️ Warning: Unable to parse label from {sample_name}, setting label = -1")
            label = -1

        annotations.append({
            'frame_dir': sample_name,
            'label': label,
            'keypoint': [ntu_joints],
            'total_frames': ntu_joints.shape[0],
        })

    out_pkl_path = os.path.join(args.out_dir, 'back', 'ntu60_3danno_back.pkl')
    os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
    with open(out_pkl_path, 'wb') as f:
        pickle.dump({'annotations': annotations, 'split': split}, f)

    print(f"✅ Reconstructed .pkl saved to {out_pkl_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True, help='Path to input .pkl (forward) or folder with .npy files (backward)')
    parser.add_argument('--out-dir', default='out', type=str, help='Directory to store outputs')
    parser.add_argument('--forward', action='store_true', help='Apply forward mapping (NTU -> SMPL)')
    parser.add_argument('--backward', action='store_true', help='Apply backward mapping (SMPL -> NTU)')
    args = parser.parse_args()
    args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)

    if args.forward and args.backward:
        raise ValueError("Cannot apply both forward and backward mapping at the same time.")
    elif args.forward:
        print("🔄 Applying forward mapping (NTU → SMPL)")
        main_forward(args)
    elif args.backward:
        print("🔄 Applying backward mapping (SMPL → NTU)")
        main_backward(args)
    else:
        raise ValueError("Please specify either --forward or --backward flag.")