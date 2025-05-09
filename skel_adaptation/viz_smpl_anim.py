import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import argparse
import os

# Your custom SMPL bone hierarchy (22 joints, no hands)
smpl_bones = {
    0: [1, 2, 3], 3: [6], 6: [9], 9: [12, 13, 14], 12: [15],
    13: [16], 16: [18], 18: [20], 20: [],
    14: [17], 17: [19], 19: [21], 21: [],
    1: [4], 4: [7], 7: [10], 10: [],
    2: [5], 5: [8], 8: [11], 11: []
}

def get_edges_from_bones(bone_dict):
    edges = []
    for parent, children in bone_dict.items():
        for child in children:
            edges.append((parent, child))
    return edges

SMPL_EDGES = get_edges_from_bones(smpl_bones)

def render_animation(motion: np.ndarray, save_path: str, fps: int = 20):
    T, J, _ = motion.shape
    assert J == 22, f"Expected 22 joints, got {J}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def set_axes(ax, center):
        range_val = 1.0
        ax.set_xlim(center[0] - range_val, center[0] + range_val)
        ax.set_ylim(center[1] - range_val, center[1] + range_val)
        ax.set_zlim(center[2] - range_val, center[2] + range_val)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        # Hide every other tick label
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            labels = axis.get_ticklabels()
            for i, label in enumerate(labels):
                if i % 2 != 0:
                    label.set_visible(False)

    def update(t):
        ax.cla()
        joints = motion[t]
        joints = joints[:, [0, 2, 1]]  # swap Y and Z for matplotlib viz.
        root = joints[0]  # joint 0 is the pelvis/root

        set_axes(ax, root)

        # draw bones
        for i, j in SMPL_EDGES:
            ax.plot(
                [joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                [joints[i, 2], joints[j, 2]],
                'b-'
            )

        # draw joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=20)
        ax.set_title(f"Frame {t+1}/{T}")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save as GIF using PillowWriter
    save_path = save_path.replace('.mp4', '.gif') if save_path.endswith('.mp4') else save_path
    ani.save(save_path, writer='pillow', fps=fps)
    print(f"✅ Saved animation to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, required=True, help='Path to .npy file (shape: T x 22 x 3)')
    parser.add_argument('--output', type=str, default=None, help='Output GIF path')
    args = parser.parse_args()

    # Auto-name output file if not provided
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.npy))[0]
        args.output = f'output/{base}.gif'

    motion = np.load(args.npy)  # shape: (T, 22, 3)

    # Check root position at frame 0
    root = motion[0, 0]
    print(f"Root (t=0) => [X={root[0]:.4f}, Y={root[1]:.4f}, Z={root[2]:.4f}]")

    render_animation(motion, args.output)
