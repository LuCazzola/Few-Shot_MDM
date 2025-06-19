# Setup

<br>

Download the repository and submodules
```bash
git clone --recursive https://github.com/LuCazzola/Few-Shot_MDM.git
git submodule update --remote
cd Few-Shot_MDM
```

## 1. Dependancies

<br>

Download the following resources at `external/motion-diffusion-model`:
* [smpl.zip](https://drive.usercontent.google.com/download?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2&authuser=1) | [glove.zip](https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing) | [t2m.zip](https://drive.usercontent.google.com/download?id=1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb&authuser=1) | [kit.zip](https://drive.usercontent.google.com/download?id=12liZW5iyvoybXD8eOw4VanTgsMtynCuU&authuser=1)

Initialize the environment
```
conda env create -f environment.yml
conda activate mdm
```

Then, execute:
```bash
bash prep/init_mdm_dep.sh
```

## 2. Data setup

<br>

Download the datasets. They're quite light as we're working solely with small skeletons and text.
```bash
bash prep/download_data.sh
```

<br>

Now launch the following script to pre-process NTU RGB+D, making it suitable for MDM
* **NOTE**: this takes more or less <u>1 hour</u> of processing for NTU60 (PySkl version)
```bash
python3 -m scripts.setup --dataset NTU60
```

<br>

Finally, symlink the data to submodules
```bash
bash prep/link_data.sh NTU60
```

# Pre-Trained MDM

<br>

Download the pretrained MDM model on HumanML3D.
* [humanml-encoder-512-50steps](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing)

```bash
mkdir save
mv humanml-encoder-512-50steps.zip save
unzip save/humanml-encoder-512-50steps.zip -p save
rm humanml-encoder-512-50steps.zip
```

That is an updated version w.r.t one provided in the original paper, with comparable performance metrics but $\times 20$ faster!
In our work we use this as transformer encoder-based prior.