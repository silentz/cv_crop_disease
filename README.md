# CGIAR Computer Vision for Crop Disease

This is the project for [CGIAR Computer Vision for Crop Disease](https://zindi.africa/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease/)
&mdash; computer vision competition for identification wheat rust in images from Ethiopia and Tanzania.
It addresses problem of multiclass classification, where every image of crop is labeled as healthy or having one
of two deseases. My approach is originally based on resnet18 &mdash; pretrained convolutional neural network that
can classify images into 1000 categories.

[//]: <> (This solution ended up at -1th place in the competition.)

## Table of contents

* [Project layout](#project-layout)
* [Data overview](#data-overview)
* [Solution details](#solution-details)
* [How to run](#how-to-run)
  * [Requirements](#requirements)
  * [Dataset](#dataset)
  * [Preprocessing](#preprocessing)
  * [Training](#training)
  * [Evaluation](#evaluation)
* [Download](#download)

[//]: <> (Problem overview)

## Project layout

```
.
├── input             # Input files provided by competition.
│   ├── train         # Train images split into categories.
│   ├── test          # Test images.
│   ├── train_clean   # Preprocessed train images split into categories.
│   └── test_clean    # Preprocessed test images.
├── logdir            # Where trained model outputs are saved.
├── src               # Soltion source code.
├── config.yml        # Catalyst configuration.
└── folds.csv         # Images split into folds.
```

## Data overview

## Solution details

## How to run

### Requirements
Here is list of requirements you need to run the project. Other versions
are not tested:

- Python 3.7.6
- Pytorch 1.4.0
- Numpy 1.18.1
- Pandas 1.0.1
- OpenCV 4.2.0.32
- Albumentations 0.4.5
- Catalyst 20.2.4
- Cnn-finetune 0.6.0
- Tqdm 4.43.0

To install them you can use:
```
pip install -r requirements.txt
```

### Dataset

To run the project you also need to dowload dataset:

1. Download train dataset using following link: [train.zip](https://api.zindi.africa/v1/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease/files/train.zip).
2. Download test dataset using following link: [test.zip](https://api.zindi.africa/v1/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease/files/test.zip).
3. Unpack `train.zip` into `input/train` directory.
4. Unpack `test.zip` into `input/test` directory.

### Preprocessing



### Training

To train a model you shuold use `catalyst-dl` utility (comes after installing catalyst):
```
catalyst-dl run --config=./config.yml --verbose
```
Then you can find trained model in `logdir/checkpoints/` directory.

### Evaluation

To make predictions using trained model you use `src/inference.py` script:
```
python src/inference.py <path/to/trained/model> <path/to/test_clean> <output file name>
```

## Download
