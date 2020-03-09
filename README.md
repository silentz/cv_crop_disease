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

### Dataset

### Preprocessing

### Training

### Evaluation

## Download
