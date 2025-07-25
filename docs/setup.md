# Setup

<br>

Download the repository and submodules
```bash
git clone --recursive https://github.com/LuCazzola/Few-Shot_MDM.git
cd Few-Shot_MDM
git submodule update --remote
```

## 1. Dependancies

<br>

Download the following resources at `./external/motion-diffusion-model`:
* [smpl.zip](https://drive.usercontent.google.com/download?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2&authuser=1) | [glove.zip](https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing) | [t2m.zip](https://drive.usercontent.google.com/download?id=1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb&authuser=1) | [kit.zip](https://drive.usercontent.google.com/download?id=12liZW5iyvoybXD8eOw4VanTgsMtynCuU&authuser=1)

1. Run
```bash
sudo apt update
sudo apt install ffmpeg
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

2. Initialize the environment
```bash
conda env create -f environment.yml
conda activate mdm
```

3. Execute to organize the downloaded `.zip` files:
```bash
bash prep/init_dep.sh
```

## 2. Data setup

<br>

1. Download the datasets. They're quite light as we're working solely with small skeletons and text.
```bash
bash prep/download_data.sh
```

2. Now launch the following script to pre-process NTU RGB+D, making it suitable for MDM
* **NOTE**: this takes more or less <u>1 hour</u> of processing for NTU60 (PySkl version)
```bash
python3 -m scripts.data_prep --dataset NTU60
```

3. Finally, symlink the data to submodules
```bash
bash prep/link_data.sh NTU60
```

**NOTE**: Above instructions donwloads NTU and minimum HumanML3D requirements to sample from MDM. They do not include full HumanML3D dataset. To fully replicate our experiments, you should follow the instructions in the [official HumanML3D repository](https://github.com/EricGuo5513/HumanML3D) to obtain the full dataset, as we can't directly distribute it.

Once you've generated the dataset, you must substitute it to `./external/motion-diffusion-model/dataset/HumanML3D`.

# Pre-Trained MDM

<br>

Download the pretrained MDM model on HumanML3D and put them within the `./external/motion-diffusion-model` folder.
* [humanml_enc_512_50steps.zip](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing)

```bash
mkdir save
unzip humanml_enc_512_50steps.zip -d save/
rm humanml_enc_512_50steps.zip
```

That is an updated version w.r.t one provided in the original paper, with comparable performance metrics but $\times 20$ faster!
In our work we use this as transformer encoder-based prior.