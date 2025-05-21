# Ranking

The Text-To-Motion model is trained on HumanML3D rather than NTU RGB+D 120. To estimate the likelihood that the model has encountered certain motion patterns, we avoid random sampling and instead select samples based on their theoretical *hardness*.

## Method

- We utilize the CLIP latent space by projecting:
  1. All motion captions from HumanML3D
  2. Natural language descriptions of action classes from NTU RGB+D 120
- For each NTU text description, we compute the *k* nearest neighbors among the HumanML3D captions.
- We compute a **density** score that reflects the likelihood of encountering HumanML3D samples around a central NTU sample.

This approach provides an overall sense of the *semantic alignment* between the two datasets. We interpret this alignment as a proxy for how well the Text-To-Motion model is expected to perform on a given NTU class. A higher density indicates a greater concentration of semantically related HumanML3D samples near the NTU reference, suggesting that the model is more likely to generate relevant motions for that category.

## Usage

Before executing, make sure you have text data according to ![Here](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#a-the-easy-way-text-only).
```bash
python3 rank.py \
  --texts-root ../../external/motion-diffusion-model/dataset/HumanML3D/texts
```

The ranking can be visualized / processed separatelly through
```bash
python3 process_rank.py \
  results/ntu_density_k200.json
```
