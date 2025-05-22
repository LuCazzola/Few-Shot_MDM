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
git clone --recursive https://github.com/LuCazzola/Few-Shot_MDM.git
bash prep/mdm_env_init.sh
```

<br>

**NOTE**: As `gdown` might fail you may need to manually download few resources into `external/motion-diffusion-model` directory and run the bash `prep/mdm_env_init.sh` script again. Respectively download and move the files:
* [smpl.zip](https://drive.usercontent.google.com/download?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2&authuser=1) in `external/motion-diffusion-model/body_models`
* [t2m.zip](https://drive.usercontent.google.com/download?id=1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb&authuser=1) in `external/motion-diffusion-model`
* [kit.zip](https://drive.usercontent.google.com/download?id=12liZW5iyvoybXD8eOw4VanTgsMtynCuU&authuser=1) in `external/motion-diffusion-model`

## Data

<details>
  <summary><b>1. Download Datasets</b></summary>

#### Automatic installation (recomended):
```bash
bash prep/data_init.sh
```

#### Manual installation:
* **HumanML3D**: We don't need the training dataset, so simply download text + dataset informations accordingly to [MDM repo.](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md#download-the-pre-processed-skeletons) we work only with 3D skeletal data, so download either `NTU RGB+D [3D skeleton]` or the `NTU RGB+D 120 [3D skeleton]`.
* **NTU RGB+D**: Download NTU data from the [PySkl repo.](https://github.com/kennymckormick/pyskl?tab=readme-ov-file#data-preparation)
</details>

<details>
  <summary><b>2. Data Convertion</b></summary>

<br>

Data from NTU RGB+D needs to be converted in order to be coherent with HumanML3D. Check details at [skel_adaptation](modules/skel_adaptation) for further details. Execute

```bash
python3 modules/skel_adaptation/skel_mapping.py \
    --input-data data/NTU60/ntu60_3danno.pkl \
    --mode=forward
```

</details>

<details>
  <summary><b>3. Data Formatting</b></summary>

<br>

Now, all raw data should be converted and stored in `modules/skel_adaptation/out`. Some other steps are required so that data is formatted accordingly to MDM expected input formatting. You can find more details and tools in [*Data*](data/)

Format data for MDM:
```bash
python3 data/prep_mdm_data.py \
  --dataset NTU60 \
  --smpl_data modules/skel_adaptation/out/forw
```

<br>

Symlink data within MDM folder (for convenience)
```bash
bash prep/mdm_dataset_init.sh NTU60
```
</details>


## Usage

The actual *models* are stored within submodules defined in `external/`. Respectively:
* **motion-diffusion-model**: our adapted version of MDM, enabling Few-Shot training and sampling.
* **PySkl**: repository of reference of the Human Action Recognition model, ST-GCN in our case.

<details>
  <summary><b>Few-Shot MDM</b></summary>

<br>

First enter the submodule

```bash
cd external/motion-diffusion-model
```

### Few-Shot Training

If all steps specified in sections **Setup** and **Data** sections were done correctly, you should be able to run the trainig with no problem. 

```bash
python -m train.train_mdm \
  --few_shot \                        
  --dataset ntu60 \
  --split splits/fewshot/5way_10shot_seed19/xset/train \
  --save_dir save/my_few_shot_ntu60_trans_enc_512 \
  --diffusion_steps 50 \
  --mask_frames \
  --use_ema
```

### Text-2-Motion Action Synthesis

Execute the following script to synthetyze motion from free text, such that:
* Textual prompts are natural language convertions of Action classes. Check [`class_captions.json`](data/NTU60/class_captions.json) for better understanding.
* At each `--shot` (repetition) all `--action_labels` are generated given a random conditioning sampled from the `.json`.

```bash
python3 -m sample.generate \
  --few_shot \
  --action_labels 0 1 2 \
  --shots 10 \
  --class_captions ../../data/NTU60/class_captions.json \
  --model_path save/humanml_enc_512_50steps/model000750000.pt \
  --no_render
```

Remove `--no_render` to trigger the rendering into `.mp4` animations and actually see the synthetic motion (its time demandingm recomend to use with small number of shots and action labels).

</details>


<details>
  <summary><b>Training ST-GCN</b></summary>

<br>

Once you've generated some synthetic data through MDM and you want to use it on your ST-GCN classifier you should first apply a format convertion back from SMPL to NTU. Assuming `--input-data` is the folder where synthetic data is stored, execute the following: 
```bash
python3 modules/skel_adaptation/skel_mapping.py \
  --input-data external/motion-diffusion-model/save/humanml_enc_512_50steps/samples_humanml_enc_512_50steps_000750000_seed10 \
  --mode=backward
```
This produces `.pkl` file inside `modules/skel_adaptation/out/back` structured in a format compatible with `PySkl` repository and containing a custom split `synth` which stores all synthetically generate data.

<br>

After that you can apply basic pre-processing to the NTU dataset by running
This is essential as it returns a transformed copy of the original data on which few basic needs are applied, such as: lowering frame-rate, dropping un-wanted joints, ...
```bash
python3 modules/skel_adaptation/skel_mapping.py \
  --input-data data/NTU60/ntu60_3danno.pkl \
  --mode=format_dataset
```
This produces a `<ds_name>_formatted.pkl` file within the `--dataset` derectory. 

<br>

Finally, you can merge the original data together with the synthetic one to obtain a final, single `.pkl` file which can directly be ported and executed into the `PySkl` toolbox. Run:
```bash
python3 data/merge_synth_data.py \
  --dataset NTU60 \
  --fewshot_split 5way_10shot_seed19/xset \
  --synth_data modules/skel_adaptation/out/back/ntu60_synth_back.pkl
```

This produces a final, unique .pkl file in which all data associated to the low-represented action classes is removed, to the exception of: 
1. Samples within `<fewshot_split>/train.txt`, which is the available low-resources real data.
2. The synthetic data prduced by MDM in `<synth_data>`.

</details>


