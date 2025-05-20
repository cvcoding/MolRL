# MolRL
Molecular Image Representation Learning through Structure Bootstrapping Self-Supervision with Hierarchical Attentive Graph Isomorphism Networks

Introduction: MolRL is a novel self-supervised pretraining deep learning framework designed specifically for learning molecular representations and predicting molecular properties. This framework addresses the challenge of acquiring labeled molecular data, which is often costly and time-consuming, by leveraging a large corpus of unlabeled molecular images. By exploiting the power of contrastive learning and hierarchical graph analysis, MolRL enables effective generalization across the vast chemical space.

PyTorch 1.12.1 Environment with Deep Learning Dependencies

# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
deepchem=2.8.0=pyhd8ed1ab_0
imagecodecs=2023.1.23=py38h6c6a46e_0
imageio=2.31.4=py38haa95532_0
pandas=2.0.3=py38h4ed8f06_0
pillow=10.0.1=py38h045eedc_0
python=3.8.18=h1aa4202_0
pytorch=1.12.1=py3.8_cuda11.3_cudnn8_0
pytorch-mutex=1.0=cuda
scikit-image=0.19.3=py38hd77b12b_1
scikit-learn=1.3.0=py38h4ed8f06_1
tensorboard=2.17.0=pyhd8ed1ab_0
torch-scatter=2.1.2=pypi_0
torchvision=0.13.1=py38_cu113

1. Data Preparation
For MoleculeNet/Freesolv (Preprocessed)
Place datasets in the following structure:
data/
  ├── freesolv/
  │   ├── train_scoffold/  # Preprocessed scaffold-split data
  │   └── test_scoffold/
  └── custom_dataset/      # For custom data (see below)

2. Reproduce Paper Results
Example: Train on Freesolv (Regression)
python train.py \
  --net vit \
  --lr 1e-4 \
  --bs 32 \
  --patch 15 \
  --data_address ../data/freesolv/train_scoffold \
  --n_epochs 50 \
  --tau 0.99 \
  --cos \
  --aug
3. Parameters Overview
Argument	Description	Default
  --lr	Learning rate	1e-4
  --bs	Batch size	32
  --patch	Patch size for graph nodes	15
  --tau	EMA decay rate for target network	0.99
  --aug	Enable image augmentations	False
  --mixup	Enable mixup augmentation	False
