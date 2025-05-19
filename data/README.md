# Data

In this folder you're supposed to put external datasets, take `NTU60` as reference.
Structure is as follows

```
- <Dataset>/
| - fewshot_data/        # defines lists of samples to include few-shot split
| - texts/               # lists captions associated to samples
| - class_captions.json  # explicits action-label -> [viable captions] mapping
| ...
```

### 1. Format data for MDM
Once you've converted data into SMPL as described in [Here](../modules/skel_adaptation/) you can run the following command to generate `texts/` folder, which formats captions to be processed as Motion-Diffusion-Model datasets

```bash
python3 prep_mdm_data.py \
  --dataset NTU60 \
  --smpl_data ../modules/skel_adaptation/out/forw
```

you can also augment `class_captions.json` through an LLM of your choice from hugging face:
1. it's *strongly recomended to create a separate conda environment solely for the dependancies of the following script*, as it has conflicts with MDM dependancies. Once obtained the augmented
2. Add your [identification token](https://huggingface.co/settings/tokens) from hugging face running `huggingface-cli login` on your terminal and pasting your access token.
3. Make sure you have permission to access the requested model, otherwise, make a requeste on [huggingface.co](https://huggingface.co/)

```
python3 action_2_caption.py \
  --dataset NTU60 \
  --model mistral-7b \
  --n 5 \
  --training
```

consider that the used .json within MDM model is always the one named `<dataset>/class\_captions.json`

### 2. Generate few-shot splits

Once you've defined which aciton classes you wish to use as Support-Set they can be explicited in `--class-list`. The script:
* samples different motion data within the specified action classes from `--input-root/annotations` 
  * `--shots` data instances are taken per `'_train.txt'` split
  * `--shots` $\times$ `--eval-multiplier` instances are instead taken per `'_val.txt'` split

consider that splits are defined within `--input-root` folder as `.txt` files

```
python3 gen_fewshot_split.py \
  --dataset NTU60 \
  --class-list 0 1 2 3 4 \
  --shots 10 \
  --eval-multiplier 5 \
  --seed 19
```

