# Multi Task Curriculum Framework for Open-Set SSL
This is the official PyTorch implementation of [Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning](https://arxiv.org/abs/1905.02249). 

## Requirements
- Python 3.7
- PyTorch 1.1.0
- torchvision 0.3.0
- tensorboardX
- progress
- matplotlib
- numpy
- scikit-learn
- scikit-image

## Preparation
Download out-of-distributin datasets from Dropbox.

```
mkdir data
cd data
wget https://www.dropbox.com/s/7nj0sfunoqu9alu/OOD_data.zip
unzip OOD_data.zip
cd ..
```

## Usage

### Train baseline
Run
```
python run.py --gpu {GPU_ID} --n-labeled {#LABELED_SAMPLES} --data {OOD_DATASET} --method baseline
```
For example, train MixMatch with 250 labeled samples and TinyImageNet as OOD, please run:
```
python run.py --gpu 0 --n-labeled 250 --data TIN --method baseline
```
Trained model will be saved at `runs_baseline`.

### Train proposed method
Run
```
python run.py --gpu {GPU_ID} --n-labeled {#LABELED_SAMPLES} --data {OOD_DATASET} --method proposed
```
For example, train proposed method with 250 labeled samples and TinyImageNet as OOD, please run:
```
python run.py --gpu 0 --n-labeled 250 --data TIN --method proposed
```
Trained model will be saved at `runs_proposed`.

**For more details and parameters, please refer to --help option.**

## References
- [1]: Qing Yu, Daiki Ikami, Go Irie and Kiyoharu Aizawa. "Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning", in ECCV, 2020.