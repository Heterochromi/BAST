## IMPORTANT!!!! This is a fork from the original BAST repo , i will be making changes to try and add classification on top of Localization, I do not really know what i am doing therefor i recommend you go back to the original repo if you wanna test/use anything with this model

## Original repo: [https://github.com/Heterochromi/BAST](https://github.com/ShengKuangCN/BAST)


# BAST

[BAST-Mamba: Binaural Audio Spectrogram Mamba Transformer for Binaural Sound Localization]()
---
## Architecture

![BAST](https://github.com/ShengKuangCN/BAST/blob/main/figure/01.BAST_architecture.png)

## Prerequisites

  pip install -r requirements.txt

## Datasets

For the audio dataset we use, please contact s.mehrkanoon@uu.nl for further information.

After downloading the dataset, please unzip files into "\data" directory. The traning and testing dataset should be under "\data\train" and '\data\test' respectively.

## Usages

### Train
We offer several training options as below
* For backbone (--backbone):
  * 'mamba': MambaVision
  * 'swin': Swin Transformer
  * 'vanilla': ViT
* For binaural integration method (--integ): 
   * 'ADD': Addition 
   * 'SUB': Subtraction
   * 'CONCAT': Concatetation
* For loss Function (--loss): 
   * 'MIX': Hybrid loss
   * 'MSE': Mean Square Error(MSE) loss
   * 'AD': Angular Distance(AD) loss
* For weights sharing (--shareweights)
* For using differentr auditory environments (--env): 
   * 'RI01': anechoic environment
   * 'RI02': lecture hall (reverberation environment)
   * 'RI': anechoic environment + lecture hall

#### example
For training BAST-Mamba with subtraction integration, hybrid loss and weights-sharing:
    
    python train_BAST.py --backbone mamba --integ SUB --loss MIX --shareweights True --env RI

### Multitask training (Classification + Localization)

Data format requirements for precomputed spectrograms:
- Each sample is a stereo log-mel spectrogram tensor of shape `[2, F, T]` saved as `.npy` or `.pt`.
- The dataset index CSV must contain the following columns:
  - `input_file`: absolute or relative path to the spectrogram file
  - `class`: class label (int or string)
  - `azimuth_deg`: azimuth in degrees (e.g., 0..360 or -180..180)
  - `elevation_deg`: elevation in degrees (e.g., -90..90)

Example minimal CSV header:

```csv
input_file,class,azimuth_deg,elevation_deg
/abs/path/spec_0001.npy,footsteps,36,-10
```

Run training with your CSV:

```bash
python train_multitask.py \
  --csv dataset_index.csv \
  --backbone vanilla \
  --integ SUB \
  --loss MIX \
  --epochs 20 \
  --batch 16 \
  --cls_weight 1.0 \
  --elev_weight 0.1
```

Notes:
- The model outputs: localization vector `[x, y]`, class logits, and a scalar elevation in degrees.
- The training script uses MIX/AD/MSE for localization, CrossEntropy for multi-class classification, and MSE for elevation.

### Test

For testing BAST-Mamba in the anechoic environment + lecture hall:
    
    python eval_BAST.py --backbone mamba --integ SUB --loss MIX --env RI

For testing BAST in the lecture hall:

    python eval_BAST.py --backbone mamba --integ SUB --loss MIX --env RI02

## Performance (BAST-Mamba)

Model | Loss | Integ. | AD
:---: | :---: | :---: | :---: 
BAST-Mamba-NSP  | MSE | Concat. | 1.61° 
BAST-Mamba-NSP  | MSE | Add. | 2.03° 
BAST-Mamba-NSP  | MSE | Sub. | 1.92° 
BAST-Mamba-NSP  | AD | Concat. |1.34° 
BAST-Mamba-NSP  | AD | Add. |1.45° 
BAST-Mamba-NSP  | AD | Sub. |1.31° 
BAST-Mamba-NSP  | Hybrid | Concat. |1.48° 
BAST-Mamba-NSP  | Hybrid | Add. |1.77° 
BAST-Mamba-NSP  | Hybrid | Sub. |1.60° 
BAST-Mamba-SP  | MSE | Concat. |1.35° 
BAST-Mamba-SP  | MSE | Add. |8.68° 
BAST-Mamba-SP  | MSE | Sub. |1.75° 
BAST-Mamba-SP  | AD | Concat. |1.10° 
BAST-Mamba-SP  | AD | Add. |21.98° 
BAST-Mamba-SP  | AD | Sub. |1.03°
BAST-Mamba-SP  | Hybrid | Concat. |1.11°
BAST-Mamba-SP  | Hybrid | Add. |10.42°
BAST-Mamba-SP  | Hybrid | Sub. |0.89°

## Citation

    @misc{https://doi.org/10.48550/arxiv.2207.03927,
      doi = {10.48550/ARXIV.2207.03927},
      url = {https://arxiv.org/abs/2207.03927},
      author = {Kuang, Sheng and van der Heijden, Kiki and Mehrkanoon, Siamak},
      title = {BAST: Binaural Audio Spectrogram Transformer for Binaural Sound Localization},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution Share Alike 4.0 International}
    }
