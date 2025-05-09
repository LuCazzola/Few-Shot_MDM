from scipy.interpolate import interp1d
import os
import pickle
import numpy as np

NTU_TO_SMPL_DIRECT_MAP = {
    0: 0, 20: 9, 2: 12, 3: 15, 4: 16, 5: 18, 6: 20,
    8: 17, 9: 19, 10: 21, 12: 1, 13: 4, 14: 7, 15: 10,
    16: 2, 17: 5, 18: 8, 19: 11,
}

def resample_motion(motion: np.ndarray, original_fps: int = 30, target_fps: int = 20) -> np.ndarray:
    T = motion.shape[0]
    original_times = np.linspace(0, T / original_fps, T)
    target_T = int(T * target_fps / original_fps)
    target_times = np.linspace(0, T / original_fps, target_T)
    interp_fn = interp1d(original_times, motion, axis=0, kind='linear')
    return interp_fn(target_times)

def forward_map(ntu_joints: np.ndarray):
    """
    Map NTU joints -> SMPL joints.
    22 joints, no hands compatible with HumanML3D dataset
    """
    T = ntu_joints.shape[0]
    smpl_joints = np.zeros((T, 22, 3), dtype=np.float32)

    for ntu_idx, smpl_idx in NTU_TO_SMPL_DIRECT_MAP.items():
        smpl_joints[:, smpl_idx, :] = ntu_joints[:, ntu_idx, :].astype(np.float32)

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
    22 joints, no hands compatible with HumanML3D dataset
    """
    pass

def main(pkl_path: str, output_dir: str):
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)

    split = raw_data.get('split', {})
    annotations = raw_data['annotations']

    # Build a reverse lookup: frame_dir → split_name (xsub/xset/xview)
    frame_to_split = {}
    for key in split:
        if key.endswith('_train'):
            split_name = key.replace('_train', '')
            for fid in split[key]:
                frame_to_split[fid] = split_name

    print(f"📦 Total training samples: {len(frame_to_split)}")

    saved_count = 0
    for ann in annotations:
        frame_dir = ann['frame_dir'].strip()
        if frame_dir not in frame_to_split:
            continue

        keypoint = ann['keypoint']
        if keypoint.shape[0] == 0:
            continue

        ntu_joints = keypoint[0].astype(np.float32)  # shape: (T, 25, 3)
        ntu_joints = resample_motion(ntu_joints, original_fps=30, target_fps=20)  # Resample to 20 FPS
        smpl_joints = forward_map(ntu_joints)

        # ✅ Normalize root joint to [0, Y, 0] at all frames
        root_x = smpl_joints[:, 0, 0:1]  # shape (T, 1)
        root_z = smpl_joints[:, 0, 2:3]  # shape (T, 1)
        smpl_joints[:, :, 0] -= root_x  # X centered
        smpl_joints[:, :, 2] -= root_z  # Z centered

        # Compose full path: output_dir/split_name/frame_dir.npy
        split_name = frame_to_split[frame_dir]
        split_out_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        out_path = os.path.join(split_out_dir, f"{frame_dir}.npy")
        np.save(out_path, smpl_joints)
        saved_count += 1

    print(f"✅ Saved {saved_count} training sequences to {output_dir} (organized by split)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to ntu60_3danno.pkl')
    parser.add_argument('--output_dir', type=str, default='ntu60_train_smpl', help='Directory to store .npy outputs')
    args = parser.parse_args()

    main(args.input, args.output_dir)
