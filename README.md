# Yandex Cup 2024, Music Information Retrieval, 1st place main track.

# Pipeline 
The model takes in (50, 84) image, resizes it to (224, 224), applies mean/std normalization, 2d backbone and arcface head. 

# Repro
To train a model:
```python3 train.py --config config_1.py```, 

NB: takes ~10 hours to saturate validation metric on 4090 GPU assuming there is no data loading bottleneck (almost any CPU would do)

# Results
| model | approx score LB | 
| --- | --- | 
| from scratch, patch transformer | 0.30 |
| whisper-base, take encoder, downsample pos encoding | 0.35 |
| 2d cnns, e.g. convnext_tiny | 0.45 | 
| add resize from orig resolution to (224, 224) | 0.55 |
| transformers, e.g. maxvit (works best) | 0.64 | 
| +5 models ensemble | 0.665 | 
| +query expansion | 0.680 |

# LB, main track
[comp link](https://yandex.ru/cup/ml)
![image](https://github.com/user-attachments/assets/c2020c05-9362-49c5-b77f-0a022595c2f0)

