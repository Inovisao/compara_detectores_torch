#!/bin/bash

conda create --name detectores python=3.9.16 -y

conda activate detectores

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -U openmim

mim install "mmengine>=0.7.0"

mim install "mmcv>=2.0.0rc4"

cd mmdetection

pip install -e .

pip install scikit-learn

pip install funcy

pip install albumentations

pip install ultralytics

pip install supervision==0.1.0
