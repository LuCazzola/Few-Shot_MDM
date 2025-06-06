"""
Constants relative to
> fixed filenames/paths used in the codebase.
> dataset-specific parameters
"""

# Dataset files formatted as in PySkl toolbox
DATA_FILENAME = {
    'NTU60': 'ntu60_3danno.pkl',
    'NTU120': 'ntu120_3danno.pkl'
}

# File which contains a mapping from action index to a list of viable captions
# for such action class
CLASS_CAPTIONS_FILENAME = "class_captions.json"

# Time Resampling parameters
DATASET_FPS = {
    'NTU60': 30,
    'NTU120': 30,
    'HML3D': 20
}