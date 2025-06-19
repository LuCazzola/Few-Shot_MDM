"""
Constants related to skeletons and their mapping to HumanML3D format.
"""

from types import SimpleNamespace

SKEL_INFO = {
    'HML3D': SimpleNamespace(
        # Lower legs
        l_idx1=5,
        l_idx2=8,
        # Right/Left foot
        fid_r=[8, 11],
        fid_l=[7, 10],
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        face_joint_indx=[2, 1, 17, 16],
        # l_hip, r_hip
        r_hip=2,
        l_hip=1,
        joints_num=22,
    ),
    
    "KINECT": SimpleNamespace(
        # Lower legs
        l_idx1=0,
        l_idx2=0,
        # Right/Left foot
        fid_r=[18, 19],
        fid_l=[14, 15],
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        face_joint_indx=[16, 12, 8, 4],
        # l_hip, r_hip
        r_hip=16,
        l_hip=12,
        joints_num=25,
    )
}

# Joints directly mappable from some skeletons to HumanML3D format
SMPL_DIRECT_MAP = {
    "KINECT" : {
        0:   0, # SpineBase: Pelvis
        2:  12, # Neck: Neck
        3:  15, # Head: Head
        4:  16, # ShoulderLeft: Lshoulder
        5:  18, # ElbowLeft: Lelbow
        6:  20, # WristLeft: Lwrist
        8:  17, # ShoulderRight: Rshoulder
        9:  19, # ElbowRight: Relbow
        10: 21, # WristRight: Rwrist
        12:  1, # HipLeft: Lhip
        13:  4, # KneeLeft: Lknee
        14:  7, # AnkleLeft: Lankle
        15: 10, # FootLeft: Lfoot
        16:  2, # HipRight: Rhip
        17:  5, # KneeRight: Rknee
        18:  8, # AnkleRight: Rankle
        19: 11, # FootRight: Rfoot
    }
}

# Joints to drop when converting some skeleton into HumanML3D format
JOINTS_2_DROP = {
    "KINECT": {7, 11, 21, 22, 23, 24}
}

# Number of joints to consider when inferring floor height
# Used on preprocessing stage to ground skeletons
FLOOR_THRE = 20

# Threshold for feet grounding detection
FEET_THRE = 0.002

# Coefficients used during Skeleton Forward mapping
FCOEFF = SimpleNamespace(
    clavicle_offset = 0.4,
    spine2_offset = 0.1,
    spine2_curve = 0.02,
    spine1_curve = 0.06
)
