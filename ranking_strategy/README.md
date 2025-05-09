# Ranking

Contains intructions for the ranking strategy adoped in the selection process for low-represented categories in NTU RGB+D w.r.t.
the HumanML3D dataset.

Before executing, please refer to "2. Get data" section in ![MDM official instructions](motion-diffusion-model/README.md)

```
python3 rank.py --texts-root path/to/textual/descriptions
```

The ranking can be visualized / processed separatelly through
```
python3 process_rank.py ...
```