# Skeleton Adaptation

Contains scripts to handle skeleton annotation convertions between NTU RGB+D 60/120 and HumanML3D

## Details
* NTU $\rightarrow$ Kinect format (25 joints).
* HumanML3D $\rightarrow$ SMPL format (without hand joints, 22 total).

| NTU Skeleton | SMPL Skeleton |
|:------------:|:-------------:|
| <img src="../media/ntu_skele.svg" width="30%"> | <img src="../media/smpl_skele.svg" width="30%"> |

## Usage

To Apply forward mapping: NTU (Kinekt) $\rightarrow$ HumanML3D (SMPL):
```
python3 skel_mapping.py --input-data ntu_data/ntu60_3danno.pkl --forward
```

To Apply backward mapping: HumanML3D (SMPL) $\rightarrow$ NTU (Kinekt):
```
python3 skel_mapping.py --input-data out/forw --backward
```

## Visualization

You can inspect both format for some random action class one next to the other using the following script
```
python3 viz.py \
    --orig-data-pkl ntu_data/ntu60_3danno.pkl \
    --forward-data-root out/forw \
    --backward-data-pkl out/back/ntu60_3da \
    nno_back.pkl --class-idx 25 \
```
