# Setup

<br>

To setup MDM dependancies (after you already created a Conda .env) run:

```bash
git clone --recursive https://github.com/LuCazzola/Few-Shot_MDM.git
cd Few-Shot_MDM
export PYTHONPATH=$(pwd)
```

## 1. Dependancies

<br>

First, Download the following resources in the specified paths. Respectively download and move the files:
* [smpl.zip](https://drive.usercontent.google.com/download?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2&authuser=1) in `external/motion-diffusion-model/body_models`
* [t2m.zip](https://drive.usercontent.google.com/download?id=1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb&authuser=1) in `external/motion-diffusion-model`
* [kit.zip](https://drive.usercontent.google.com/download?id=12liZW5iyvoybXD8eOw4VanTgsMtynCuU&authuser=1) in `external/motion-diffusion-model`

Then, execute:
```bash
bash prep/init_mdm_dep.sh
```


## 2. Data setup

<br>

Start by downloading the datasets. They're quite light as we're working solely with small skeletons and text.
```bash
bash prep/download_data.sh
```

<br>

Now launch the following script to:
1. .
2. .
3. ...
```bash
python3 -m scripts.setup \
    --dataset NTU60
```

<br>

Finally, link the data to submodules
```bash
bash prep/link_mdm_resources.sh
```

