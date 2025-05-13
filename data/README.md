# Data

Once you've converted data from NTU into SMPL you can generate random Few-Shot datasets using the following script

```
python3 gen_fewshot_split.py \
  --input-root ../modules/skel_adaptation/out/forw \
  --class-list 0 1 2 3 4 \
  --shots 10 \
  --eval-multiplier 5 \
  --seed 123
```