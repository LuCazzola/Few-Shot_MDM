import numpy as np
from scipy.interpolate import interp1d

NTU_TO_SMPL_DIRECT_MAP = {
    0: 0, 20: 9, 2: 12, 3: 15, 4: 16, 5: 18, 6: 20,
    8: 17, 9: 19, 10: 21, 12: 1, 13: 4, 14: 7, 15: 10,
    16: 2, 17: 5, 18: 8, 19: 11,
}
# Ids of hand joints to be dropped (when needed)
OMIT_JOINTS = {7, 11, 21, 22, 23, 24}

def resample_motion(motion: np.ndarray, original_fps: int = 30, target_fps: int = 20) -> np.ndarray:
    """
    Subsample motion to target_fps.
    """
    T = motion.shape[0]
    original_times = np.linspace(0, T / original_fps, T)
    target_T = int(T * target_fps / original_fps)
    target_times = np.linspace(0, T / original_fps, target_T)
    interp_fn = interp1d(original_times, motion, axis=0, kind='linear')
    return interp_fn(target_times).astype(motion.dtype)

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