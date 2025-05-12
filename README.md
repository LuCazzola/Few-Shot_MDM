# Few-Shot Motion-Diffusion-Model for downstream Human Action Recognition tasks

* .

## Setup

To setup MDM dependancies (after you already created a Conda .env) run:
```
bash prep/mdm_env_init.sh
```

## Data

* **HumanML3D**: We don't need the training dataset as we don't plan on training on this dataset, so simply download text + dataset informations accordingly to ![MDM repo.](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md#download-the-pre-processed-skeletons) we work only with 3D skeletal data, so download either \texttt{NTU RGB+D [3D skeleton]} or the \texttt{NTU RGB+D 120 [3D skeleton]}.
* **NTU RGB+D**: Download NTU data from the ![PySkl repo.](https://github.com/kennymckormick/pyskl?tab=readme-ov-file#data-preparation)

Such things are automatically done by executing:
```
bash prep/data_init.sh
```


