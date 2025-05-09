# Ranking

Contains intructions for the ranking strategy adoped in the selection process for low-represented categories in NTU RGB+D w.r.t.
the HumanML3D dataset.

Before executing, make sure you have text data according to !
```
python3 rank.py --texts-root ../motion-diffusion-model/dataset/HumanML3D/texts
```

The ranking can be visualized / processed separatelly through
```
python3 process_rank.py results/ntu_density_k200.json
```