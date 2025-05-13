import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import argparse

# SMPL skeleton (22 joints, no hands)
smpl_bones = {
    0: [1, 2, 3], 3: [6], 6: [9], 9: [12, 13, 14], 12: [15],
    13: [16], 16: [18], 18: [20], 20: [],
    14: [17], 17: [19], 19: [21], 21: [],
    1: [4], 4: [7], 7: [10], 10: [],
    2: [5], 5: [8], 8: [11], 11: []
}

# Full original NTU skeleton
ntu_bones_original = {
    0: [1, 12, 16], 1: [20], 20: [2, 4, 8], 2: [3], 3: [],
    4: [5], 5: [6], 6: [7], 7: [21, 22], 21: [], 22: [],
    8: [9], 9: [10], 10: [11], 11: [23, 24], 23: [], 24: [],
    12: [13], 13: [14], 14: [15], 15: [],
    16: [17], 17: [18], 18: [19], 19: []
}

# Partial NTU skeleton (19 joints, no hands)
OMIT_JOINTS = {7, 11, 21, 22, 23, 24}
ntu_reduced_map = [None if i in OMIT_JOINTS else i - sum(j < i for j in OMIT_JOINTS) for i in range(25)]
ntu_bones_reduced = {
    ntu_reduced_map[p]: [ntu_reduced_map[c] for c in children if ntu_reduced_map[c] is not None]
    for p, children in ntu_bones_original.items()
    if ntu_reduced_map[p] is not None
}

def get_edges_from_bones(bone_dict):
    return [(parent, child) for parent, children in bone_dict.items() for child in children]

SMPL_EDGES = get_edges_from_bones(smpl_bones)
NTU_EDGES = get_edges_from_bones(ntu_bones_reduced)

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

        for i, j in SMPL_EDGES:
            ax_smpl.plot([smpl_joints[i, 0], smpl_joints[j, 0]],
                         [smpl_joints[i, 1], smpl_joints[j, 1]],
                         [smpl_joints[i, 2], smpl_joints[j, 2]], 'b-')
        for i, j in NTU_EDGES:
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
    print(f"✅ Saved dual animation to: {save_path}")

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

        for i, j in get_edges_from_bones(ntu_bones_original):
            ax.plot([joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]], 'k-')
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='purple', s=20)
        ax.set_title(f"Original NTU (30 FPS) - Frame {t+1}/{T}")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer='pillow', fps=fps)
    print(f"✅ Saved original NTU animation to: {save_path}")

def pick_random_sample(args):
    with open(args.backward_data_pkl, 'rb') as f:
        data = pickle.load(f)
    annotations = data['annotations']
    split_dict = data['split']

    # Pick from training splits only
    candidates = [(a['frame_dir'], split_name)
                  for split_name, sample_list in split_dict.items() if split_name.endswith('_train')
                  for a in annotations if a['label'] == args.class_idx and a['frame_dir'] in sample_list]

    if not candidates:
        raise ValueError(f"No samples found for class {args.class_idx}")

    selected, split = random.choice(candidates)
    print(f"🎯 Selected sample: {selected} (class {args.class_idx}) from split: {split}")

    # Load NTU motion (backward converted)
    ntu_motion = next(a['keypoint'][0] for a in annotations if a['frame_dir'] == selected)

    # Load original NTU motion
    with open(args.orig_data_pkl, 'rb') as f:
        orig_data = pickle.load(f)
    orig_motion = next(a['keypoint'][0] for a in orig_data['annotations'] if a['frame_dir'] == selected)

    # Load SMPL motion from flat structure
    smpl_path = os.path.join(args.forward_data_root, 'annotations', f"{selected}.npy")
    if not os.path.exists(smpl_path):
        raise FileNotFoundError(f"❌ SMPL file not found at {smpl_path}")
    smpl_motion = np.load(smpl_path)

    return smpl_motion, ntu_motion, orig_motion, selected

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-data-pkl', type=str, required=True, help='Path to original NTU .pkl file (30 FPS)')
    parser.add_argument('--forward-data-root', type=str, required=True, help='Path to SMPL root directory (outcome of forward)')
    parser.add_argument('--backward-data-pkl', type=str, required=True, help='Path to NTU .pkl file (outcome of backward)')
    parser.add_argument('--class-idx', type=int, required=True)
    parser.add_argument('--output-dir', type=str, default='media', help='Base output directory')
    args = parser.parse_args()

    smpl, ntu, orig, name = pick_random_sample(args)

    out_dir = os.path.join(args.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    render_dual_animation(smpl, ntu, os.path.join(out_dir, 'fb_process.gif'))
    render_original_ntu(orig, os.path.join(out_dir, 'orig.gif'))