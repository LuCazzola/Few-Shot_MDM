# Few-Shot Motion-Diffusion-Model for downstream Human Action Recognition tasks

Official implementation of {...}

## 🧠 Overview: Leveraging Text-to-Motion Priors for Few-Shot Human Action Recognition

This project presents a novel pipeline that leverages **Text-to-Motion generative priors** to enhance **Few-Shot Learning** in **Human Action Recognition (HAR)**. Rather than focusing solely on classifier design, we introduce a data-centric strategy that employs powerful generative models, such as the **Motion Diffusion Model (MDM)**, to synthesize 3D skeletal motion sequences from free-form text prompts (e.g., _"A person waves"_).

The generated motions act as realistic surrogates for rare or unseen classes, enriching the training set of downstream HAR models such as **ST-GCN**. This augmentation strategy boosts generalization and addresses the class imbalance common in few-shot scenarios.

### 🚀 Key Contributions

1. **Text-to-Motion for HAR Augmentation**  
   We introduce the first pipeline (to the best of our knowledge) that uses pretrained Text-to-Motion models as motion data generators to augment Human Action Recognition tasks.

2. **Few-Shot Mixture-of-Experts (MoE) Framework**  
   We propose a parameter-efficient adaptation strategy based on **Mixture-of-Experts**, enabling domain adaptation of the generative model with minimal real data.

This cross-modal approach bridges natural language understanding and motion synthesis, offering a scalable and modular solution to real-world data scarcity in HAR.


## Setup

To setup MDM dependancies (after you already created a Conda .env) run:

```bash
bash prep/mdm_env_init.sh
```

## Data

<details>
  <summary><b>1. Download Datasets</b></summary>

#### Automatic installation (recomended):
```bash
bash prep/data_init.sh
```

#### Manual installation:
* **HumanML3D**: We don't need the training dataset as we don't plan on training on this dataset, so simply download text + dataset informations accordingly to [MDM repo.](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md#download-the-pre-processed-skeletons) we work only with 3D skeletal data, so download either `NTU RGB+D [3D skeleton]` or the `NTU RGB+D 120 [3D skeleton]`.
* **NTU RGB+D**: Download NTU data from the [PySkl repo.](https://github.com/kennymckormick/pyskl?tab=readme-ov-file#data-preparation)

</details>

<details>
  <summary><b>2. Data Convertion</b></summary>

Data from NTU RGB+D needs to be converted in order to be coherent with HumanML3D. Check details at [skel_adaptation](modules/skel_adaptation) for further details. Execute

```bash
python3 modules/skel_adaptation/skel_mapping.py \
    --input-data data/NTU60/ntu60_3danno.pkl \
    --forward
```

</details>

<details>
  <summary><b>3. Data Formatting</b></summary>

Now, all raw data should be converted and stored in `modules/skel_adaptation/out`. Some other steps are required so that data is formatted accordingly to MDM expected input formatting.

Format data for MDM:
```bash
python3 data/prep_mdm_data.py \
  --dataset NTU60 \
  --smpl_data modules/skel_adaptation/out/forw
```

Symlink data within MDM folder (for convenience)
```bash
bash prep/mdm_dataset_init.sh NTU60
```
</details>


## Usage

The actual modules are stored within submodule in `external/`. Respectively:
* **motion-diffusion-model**: our adapted version of MDM.
* **PySkl**: repository of reference of the Human Action Recognition model, ST-GCN in our case.



<details>
  <summary><b>Training MDM</b></summary>

First enter the submodule
```bash
cd external/motion-diffusion-model
```

If all steps specified in sections [Data](Data) and [Setup](Setup) you should be able to run the trainig with no problem. 
```bash
python -m train.train_mdm \
    --few_shot \
    --dataset ntu60 \
    --split 5way_10shot_seed19/xset/train \
    --save_dir save/my_few_shot_ntu60_trans_enc_512
```
</details>



