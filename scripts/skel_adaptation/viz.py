import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import argparse

from common.constants import DATA_FILENAME
from scripts.skel_adaptation.skel_mapping import backward_map, backward_preprocess

# SMPL skeleton (22 joints, no hands)
SMPL_BONES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # Spine
    (1, 0), (4, 1), (7, 4), (10, 7), # R. Leg
    (2, 0), (5, 2), (8, 5), (11, 8), # L. Leg
    (13, 9), (16, 13), (18, 16), (20, 18), # R. Arm
    (14, 9), (17, 14), (19, 17), (21, 19) # L. Arm
]
# Full original NTU skeleton
NTU_BONES = [
    (1, 2), (2, 21), (3, 21), (4, 3), # Torso
    (5, 21), (6, 5), (7, 6), (8, 7), # R. Arm
    (9, 21), (10, 9), (11, 10), (12, 11), # L. Arm
    (13, 1), (14, 13), (15, 14), (16, 15), # R. Leg
    (17, 1), (18, 17), (19, 18), (20, 19), # L. Leg
    (22, 8), (23, 8), # R. Hand
    (24, 12), (25, 12) # L. Hand
]
NTU_BONES = [(i-1, j-1) for i, j in NTU_BONES]  # Convert to 0-indexed

NTU_REDUCED_BONES = [
    (1, 2), (2, 19), (3, 19), (4, 3), # Torso
    (5, 19), (6, 5), (7, 6),# R. Arm
    (8, 19), (9, 8), (10, 9), # L. Arm
    (11, 1), (12, 11), (13, 12), (14, 13), # R. Leg
    (15, 1), (16, 15), (17, 16), (18, 17), # L. Leg
]
NTU_REDUCED_BONES = [(i-1, j-1) for i, j in NTU_REDUCED_BONES]  # Convert to 0-indexed


def render_dual_animation(smpl_motion: np.ndarray, ntu_motion: np.ndarray, save_path: str, fps: int = 20):
    T = smpl_motion.shape[0]
    assert ntu_motion.shape[0] == T, "Both motions must have same frame count"

    fig = plt.figure(figsize=(12, 6))
    ax_smpl = fig.add_subplot(121, projection='3d')
    ax_ntu = fig.add_subplot(122, projection='3d')

    def set_axes(ax, center, zoom=0.7):
        ax.set_xlim(center[0] - zoom, center[0] + zoom)
        ax.set_ylim(center[1] - zoom, center[1] + zoom)
        ax.set_zlim(center[2] - zoom, center[2] + zoom)

        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])

    def update(t):
        ax_smpl.cla()
        ax_ntu.cla()

        smpl_joints = smpl_motion[t][:, [0, 2, 1]]
        ntu_joints = ntu_motion[t][:, [0, 2, 1]]

        smpl_root = smpl_joints[0]
        ntu_root = ntu_joints[0]

        set_axes(ax_smpl, smpl_root)
        set_axes(ax_ntu, ntu_root)

        ax_smpl.set_title(f"SMPL Frame {t+1}/{T}")
        ax_ntu.set_title(f"NTU Frame {t+1}/{T}")

        for i, j in SMPL_BONES:
            ax_smpl.plot([smpl_joints[i, 0], smpl_joints[j, 0]],
                         [smpl_joints[i, 1], smpl_joints[j, 1]],
                         [smpl_joints[i, 2], smpl_joints[j, 2]], 'b-')
        for i, j in NTU_REDUCED_BONES:
            if i >= ntu_joints.shape[0] or j >= ntu_joints.shape[0]:
                continue
            ax_ntu.plot([ntu_joints[i, 0], ntu_joints[j, 0]],
                        [ntu_joints[i, 1], ntu_joints[j, 1]],
                        [ntu_joints[i, 2], ntu_joints[j, 2]], 'g-')

        ax_smpl.scatter(smpl_joints[:, 0], smpl_joints[:, 1], smpl_joints[:, 2], c='r', s=20)
        ax_ntu.scatter(ntu_joints[:, 0], ntu_joints[:, 1], ntu_joints[:, 2], c='orange', s=20)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=fps)
    print(f"Saved dual animation to: {save_path}")

def render_original_ntu(raw_motion: np.ndarray, save_path: str, fps: int = 30):
    T = raw_motion.shape[0]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def set_axes(ax, center, zoom=0.7):
        ax.set_xlim(center[0] - zoom, center[0] + zoom)
        ax.set_ylim(center[1] - zoom, center[1] + zoom)
        ax.set_zlim(center[2] - zoom, center[2] + zoom)

        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])

    def update(t):
        ax.cla()
        joints = raw_motion[t][:, [0, 2, 1]]
        root = joints[0]
        set_axes(ax, root)

        for i, j in NTU_BONES:
            ax.plot([joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]], 'k-')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='purple', s=20)
        ax.set_title(f"Original NTU (30 FPS) - Frame {t+1}/{T}")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=fps)
    print(f"Saved original NTU animation to: {save_path}")

def pick_random_sample(orig_data, backw_data, forw_data_path, args):
    
    # Pick from training splits only
    candidates = [(a['frame_dir'], split_name)
                  for split_name, sample_list in orig_data['split'].items() if split_name.endswith('_train')
                  for a in orig_data['annotations'] if a['label'] == args.class_idx and a['frame_dir'] in sample_list]

    if not candidates:
        raise ValueError(f"No samples found for class {args.class_idx}")

    selected, split = random.choice(candidates)
    print(f"Selected sample: {selected} (class {args.class_idx}) from split: {split}")

    orig_motion = next(a['keypoint'][0] for a in orig_data['annotations'] if a['frame_dir'] == selected)
    ntu_motion = next(a['keypoint'][0] for a in backw_data['annotations'] if a['frame_dir'] == selected)
    smpl_motion = np.load(os.path.join(forw_data_path, f"{selected}.npy"))
    
    return smpl_motion, ntu_motion, orig_motion, selected

def cache_backward_data(forw_data, forw_filenames, out_file_path):
    """
    Cache backward converted data from forward data.
    - This is just to evaluate visually the correctness of Forward-Backward process.
    """
    backw_data = {
        'annotations': [],
        'split': {}
    }
    for motion, name in zip(forw_data, forw_filenames):
        motion = backward_preprocess(motion)
        motion = backward_map(motion)
        backw_data['annotations'].append({
            'frame_dir': name.replace(".npy", ""),
            'label': -1, # Placeholder for label
            'keypoint': np.expand_dims(motion, axis=0), # Add extra dimension to mimic number of skeletons
            'total_frames': motion.shape[0]
        })

    with open(out_file_path, 'wb') as f:
        pickle.dump(backw_data, f)

    return backw_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NTU60', choices=['NTU60', 'NTU120'], help='Dataset to visualize')
    parser.add_argument('--class-idx', type=int, required=True)
    parser.add_argument('--output-dir', type=str, default='media', help='Base output directory')
    parser.add_argument('--cache', type=str, default='cache', help='Cache directory for data')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join('data', args.dataset)
    base_cache_path = os.path.join(script_dir, args.cache)
    
    orig_data_path = os.path.join(base_data_path, DATA_FILENAME[args.dataset])
    forw_data_path = os.path.join(base_data_path, 'annotations')
    backw_data_path = os.path.join(base_cache_path, f"{args.dataset}_backw.pkl")
    args.output_dir = os.path.join(script_dir, args.output_dir, args.dataset)

    # 1. Load data
    print(f"Loading orignal data from: {orig_data_path}")
    with open(orig_data_path, 'rb') as f:
        orig_data = pickle.load(f)

    print(f"Loading forward data from: {forw_data_path}")
    forw_filenames = [f for f in os.listdir(forw_data_path) if f.endswith('.npy')]
    forw_data = []
    for filename in forw_filenames:
        with open(os.path.join(forw_data_path, filename), 'rb') as file:
           forw_data.append(np.load(file))

    print(f"Loading backward data from: {backw_data_path}")
    backw_data = []
    if not os.path.exists(backw_data_path):
        print(f"Cache not found at {backw_data_path}, creating new cache...")
        os.makedirs(base_cache_path, exist_ok=True)
        backw_data = cache_backward_data(forw_data, forw_filenames,backw_data_path)
    else:
        with open(backw_data_path, 'rb') as f:
            backw_data = pickle.load(f)

    # 2. Pick a random sample from the dataset
    smpl, ntu, orig, name = pick_random_sample(orig_data, backw_data, forw_data_path, args)

    # 3. Render animations
    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    render_dual_animation(smpl, ntu, os.path.join(out_dir, 'fb_process.gif'))
    render_original_ntu(orig, os.path.join(out_dir, 'orig.gif'))