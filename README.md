# Few-Shot Motion-Diffusion-Model for downstream Human Action Recognition tasks

Official implementation of {...}

## 🧠 Overview: Leveraging Text-to-Motion Priors for Few-Shot Human Action Recognition

This project presents a novel pipeline that leverages **Text-to-Motion generative priors** to enhance **Few-Shot Learning** in **Human Action Recognition (HAR)**. Rather than focusing solely on classifier design, we introduce a data-centric strategy that employs powerful generative models, such as the **Motion Diffusion Model (MDM)**, to synthesize 3D skeletal motion sequences.

The generated motions act as realistic surrogates for rare or unseen classes, enriching the training set of downstream HAR models such as **ST-GCN**. This augmentation strategy boosts generalization and addresses the class imbalance common in few-shot scenarios.

### 🚀 Key Contributions

1. **Text-to-Motion for HAR Augmentation**  
   We introduce the first pipeline (to the best of our knowledge) that uses pretrained Text-to-Motion models as motion data generators to augment Human Action Recognition tasks.

2. **Few-Shot Mixture-of-Experts (MoE) Framework**  
   We propose a parameter-efficient adaptation strategy based on **Mixture-of-Experts**, enabling domain adaptation of the generative model with minimal real data.

This cross-modal approach bridges natural language understanding and motion synthesis, offering a scalable and modular solution to real-world data scarcity in HAR.


## How to use

First of all, **follow [setup instructions](docs/setup.md)**.

The actual *models* are stored within submodules defined in `external/`. Respectively:
* **motion-diffusion-model**: our adapted version of MDM, enabling Few-Shot training and sampling.
* **PySkl**: repository of reference of the Human Action Recognition model, ST-GCN in our case.

<details>
  <summary><b>Generating Few-Shot splits</b></summary>

You can randomly generate Few-Shot splits by executing the following command
```bash
python3 -m scripts.handle_fewshot_split \
  --mode generate --dataset NTU60 --seed 19 \
  --class-list 2 3 19 29 \
  --shots 128 --eval-multiplier 0.25
```

This process generates a support set of size `N * len(--class-list)`, where:

- `N` is set to `--shots` for training splits
- `N` is set to `--shots * --eval-multiplier` for validation and test splits

You can also avoid specifying `--class-list`. Doing so, will use all classes in the dataset, apart prohibited ones (those having multiple skeletons)


The operation is applied independently to all available splits (e.g., `xset`, `xsub`, and `xview` for the NTU60 dataset). 

During generation:
- Statistics such as `Mean` and `Std` are computed using only the training samples
- A `pyskl_data.pkl` file is created, representing an **unbalanced** few-shot dataset

This means that:
- If `--class-list 2 19 29` is specified, the resulting dataset will retain only the sampled few-shot instances for those classes
- All other classes will remain unchanged with their full original instances

</details>


<details>
  <summary><b>Training MDM & CycleMDM</b></summary>

<br>

First enter the submodule

```bash
cd external/motion-diffusion-model
```

Pre-Trained MDM can be downloaded from the [Original Repo](https://github.com/GuyTevet/Motion-Diffusion-Model?tab=readme-ov-file#3-download-the-pretrained-models) and then stored under `save/` directory.

### Motion Synthesis

```bash
python3 -m sample.generate \
  --action_id 2 29 29 \
  --num_samples 10 --num_repetitions 3 \
  --model_path ./save/yor/action/conditioned/mdm/model
```

To run generation on a text-conditioned model specify instead `--text_prompt` "the person ..."

<br>

### Training the models

If all steps specified in sections **Setup** and **Data** sections were done correctly, you should be able to run the trainig with no problem.

```bash
python -m train.train_mdm \
  --model_type MDM --single_stream target \
  --save_dir ./save/ntu60_trans_enc \
  --starting_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt \
  --peft LoRA
```

For information about input arguments, see [`parser_util.py`](external/motion-diffusion-model/utils/parser_util.py). <u>We highly suggest to modify parameters directly there (there's a lot!)</u>. Most relevant to our work are:
* `--model_type` => specifies the model you want to train, choosing between `MDM` and `CycleMDM`. When using `MDM` a single stream model is used (as in the original paper) and `--single_stream` need to be specified to select which dataset configuration to use.
* `--peft [LoRA, MoE]` => you can specify which adapter to plug in the model (even both as a list). where they will be placed withing the model depends on other arguments. You should avoid same modules twice (ex. LoRA on denoising head, and also MoE on denoising head).

Other quality of life flags
1. `--train_platform_type` => to log your results, we suggest `WandBPlatform` option
2. .

</details>


<details>
  <summary><b>Training ST-GCN</b></summary>

<br>

Since **(for the moment)** we're not using a classifier-in-the-loop approach, training a classifier is straightforward: simply follow the [PySkl instructions](https://github.com/kennymckormick/pyskl) for training an ST-GCN model and substitute your dataset accordingly. Just **remember to use customized version** you can find in `external/pyskl`.

Here is an overview of the "usable" data files and their purposes:

1. `data/<DATASET>/<DATA>_formatted.pkl`
  → This file contains the fully pre-processed dataset. It can be used to train and evaluate a model under standard preprocessing conditions (e.g., 20 FPS resampling, no hand joints). It also serves as a baseline to investigate whether hand joints, although noisy, contribute meaningfully to action recognition.

2. `data/<DATASET>/splits/fewshot/<ID>/pyskl_data.pkl`  
  → This file is produced after generating a few-shot split. It contains an **unbalanced** dataset where only the selected few-shot classes retain a limited number of instances. Use this to evaluate how your classifier performs under data-scarce conditions for specific classes.

3. `data/<DATASET>/splits/fewshot/<ID>/<SPLIT>/pyskl_data_wsyn.pkl`
  → This version of the dataset includes synthetic motion data generated by the MDM pipeline. It serves as the primary benchmark for evaluating whether synthetic samples improve classification performance in the few-shot setting.

</details>


