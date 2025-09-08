# Few-Shot Motion-Diffusion-Model for downstream Human Action Recognition tasks

Official implementation of {...}

## 🧠 Overview: Leveraging Text-to-Motion Priors for Few-Shot Human Action Recognition

This project presents a novel pipeline that leverages **Text-to-Motion generative priors** to enhance **Few-Shot Learning** in **Human Action Recognition (HAR)**. Rather than focusing solely on classifier design, we introduce a data-centric employing generative models, such as the **Motion Diffusion Model (MDM)**, to synthesize 3D skeletal motion sequences.

The generated motions act as realistic surrogates for rare or unseen classes, enriching the training set of downstream HAR models such as **ST-GCN**. This augmentation strategy boosts generalization and addresses the class imbalance common in few-shot scenarios.

### 🚀 Key Contributions

1. .
2. .
3. .


## How to use

First of all, **follow [setup instructions](docs/setup.md)**.

The actual *models* are stored within submodules defined in `external/`. Respectively:
* **motion-diffusion-model**: our adapted version of MDM.

<details>
  <summary><b>Building Few-Shot splits</b></summary>

You can randomly generate Few-Shot splits by executing the following command
```bash
python3 -m scripts.sample_fewshot_split --dataset NTU60 --seed 19 --class-list 2 3 19 29 --shots 128
```

You can also avoid specifying `--class-list`, this will use all classes in the dataset (apart prohibited ones, like those having multiple skeletons). The operation is applied independently to all available tasks (e.g., `xset`, `xsub`, and `xview` for the NTU60 dataset). The script also produces validation splits by considering the full validation set and keeping only specified classes, in order to make validation invariant to the sampled data from the training set.

</details>


<details>
  <summary><b>Training MDM & CycleMDM</b></summary>

<br>

First enter the submodule

```bash
cd external/motion-diffusion-model
```

Pre-Trained MDM can be downloaded from the [Original Repo](https://github.com/GuyTevet/Motion-Diffusion-Model?tab=readme-ov-file#3-download-the-pretrained-models) and then stored under `./save/` directory.


<br>

### Training the models

If all steps specified in sections **Setup** and **Data** sections were done correctly, you should be able to run the trainig with no problem.

```bash
python3 -m train.train_mdm  --model_type MDM --single_stream target --save_dir ./save/ntu60_trans_enc --starting_checkpoint ./save/humanml_enc_512_50steps/model000750000.pt
```

For information about input arguments, see [`parser_util.py`](external/motion-diffusion-model/utils/parser_util.py). <u>We highly suggest to modify parameters directly there (there's a lot!)</u>. Most relevant to our work are:
* `--model_type` $\rightarrow$ specifies the model you want to train, choosing between `MDM` and `CycleMDM`. When using `MDM` a single stream model is used (as in the original paper) and `--single_stream` need to be specified to select which dataset configuration to use.
* `--peft [LoRA, MoE]` $\rightarrow$ you can specify which adapter to plug in the model (even both as a list). where they will be placed withing the model depends on other arguments. You should avoid same modules twice (ex. LoRA on denoising head, and also MoE on denoising head).

Other quality of life flags
1. `--train_platform_type` $\rightarrow$ to log your results, we suggest `WandBPlatform` option
2. `--eval_during_training` $\rightarrow$ perform validation loop during training
3. `--gen_during_trainig` $\rightarrow$ generate animations right before validation loop
4. ``

</details>


<details>
  <summary><b>Generate samples</b></summary>


### Motion Synthesis

```bash
python3 -m sample.generate --model_type MDM --model_path ./save/yor/action/conditioned/mdm/model --action_id 2 19 29 --num_repetitions 5
```

### Build Synthetic dataset
```
< WORK IN PROGRESS >
```

</details>



<details>
  <summary><b>Training ST-GCN</b></summary>

### Training ST-GCN evaluator

You can easily train a ST-GCN evaluator on created Few-Shot splits by running
```bash
python3 -m train.train_evaluator --fewshot_id <name_of_split>
```

To train a ST-GCN on some generated data run
```bash
< WORK IN PROGRESS >
```


<br>

...

