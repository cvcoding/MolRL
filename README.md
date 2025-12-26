
# Molecular Representation Learning (MolRL) for Molecular Property Prediction

This repository contains the implementation of the **MolRL** framework proposed in the paper "Molecular Image Representation Learning through Structure Bootstrapping Self-Supervision with Hierarchical Attentive Graph Isomorphism Networks". MolRL is a self-supervised pre-training framework that learns molecular representations from unlabeled molecular images, enabling accurate prediction of molecular properties (classification/regression) and drug-target affinity (DTA).


## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Training Pipeline](#training-pipeline)
   - [Unsupervised Pre-training (Structure Bootstrapping Self-Supervision)](#unsupervised-pre-training-structure-bootstrapping-self-supervision)
   - [Supervised Fine-tuning on Downstream Tasks](#supervised-fine-tuning-on-downstream-tasks)
4. [Testing and Evaluation](#testing-and-evaluation)
5. [Prediction on New Datasets](#prediction-on-new-datasets)
6. [Reproducing Paper Results](#reproducing-paper-results)
7. [Custom Dataset Guide](#custom-dataset-guide)


## 1. Environment Setup
### Requirements
- Python 3.8
- PyTorch 1.12.1 with CUDA 11.3
- torchvision 0.13.1
- DeepChem 2.8.0
- RDKit (for molecular image generation)
- Other dependencies: `pillow`, `scikit-learn`, `warmup-scheduler`

### Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate molrl

# Install additional packages
pip install -r requirements.txt
pip install rdkit-pypi  # For molecular image generation
```


## 2. Data Preparation
### Dataset Structure
#### A. Unsupervised Pre-training
Use **unlabeled molecular images** (2D renderings of molecules) generated from SMILES using RDKit. Organize images in a flat folder (class labels are irrelevant for pre-training):
```
data/pretrain/
    mol1.png
    mol2.png
    ...
```

#### B. Supervised Fine-tuning
Use labeled datasets with **scaffold splitting** (as in MoleculeNet) for training/validation/testing:
```
data/downstream/
    train_scaffold/
        mol1.png
        mol2.png
        ... (with .csv labels: mol_id,property)
    val_scaffold/
        mol3.png
        ...
    test_scaffold/
        mol4.png
        ...
```
- Supported tasks: molecular property classification (e.g., BBBP, Tox21), regression (e.g., QM9), and drug-target affinity (DTA) prediction (e.g., KIBA, Davis).

### Data Preprocessing
1. Convert SMILES to 240x240 molecular images using RDKit:
   ```python
   from rdkit import Chem
   from rdkit.Chem import Draw

   mol = Chem.MolFromSmiles(smiles)
   Draw.MolToFile(mol, f"{mol_id}.png", size=(240, 240))
   ```
2. For DTA tasks, pair molecular images with target protein sequences (e.g., in FASTA format).


## 3. Training Pipeline
### A. Unsupervised Pre-training (Structure Bootstrapping Self-Supervision)
#### Objective
Pre-train the MolRL framework using **contrastive learning (BYOL)** on unlabeled molecular images to learn hierarchical graph-structured representations.

#### Command
```bash
python main.py --mode pretrain \
               --data_dir data/pretrain/ \
               --image_size 240 \
               --patch_size 15 \
               --num_epochs 600 \
               --batch_size 128 \
               --lr 1e-4 \
               --tau 0.99 \
               --save_dir checkpoints/pretrain/
```

#### Key Parameters
- `--mode pretrain`: Activate self-supervised pre-training.
- `--patch_size 15`: Split 240x240 images into 16x16 patches (256 nodes per graph).
- `--tau 0.99`: Exponential moving average weight for target network.
- `--save_dir`: Path to save pre-trained checkpoints (e.g., `molrl_pretrain.pth`).


### B. Supervised Fine-tuning on Downstream Tasks
#### Objective
Fine-tune the pre-trained MolRL on labeled tasks (classification/regression/DTA).

#### Command for Classification (e.g., BBBP)
```bash
python main.py --mode finetune \
               --data_dir data/bbbp/ \
               --image_size 240 \
               --patch_size 15 \
               --num_epochs 100 \
               --batch_size 64 \
               --lr 1e-5 \
               --task_type classification \
               --num_classes 2 \
               --load_ckpt checkpoints/pretrain/molrl_pretrain.pth
```

#### Command for Regression (e.g., QM9)
```bash
python main.py --mode finetune \
               --data_dir data/qm9/ \
               --task_type regression \
               --loss_fn mse \
               --num_targets 1 \  # Number of regression targets (e.g., 1 for energy)
               --load_ckpt checkpoints/pretrain/molrl_pretrain.pth
```


## 4. Testing and Evaluation
#### Objective
Evaluate the fine-tuned model on the test set using task-specific metrics.

#### Command
```bash
python main.py --mode test \
               --data_dir data/test_scaffold/ \
               --task_type classification \  # or "regression"/"dta"
               --load_ckpt checkpoints/finetune/molrl_finetuned.pth \
               --output_dir results/
```

#### Metrics
- **Classification**: ROC-AUC, accuracy.
- **Regression**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
- **DTA**: Mean Squared Error (MSE), Pearson Correlation Coefficient.


## 5. Prediction on New Datasets
#### Objective
Generate predictions for new molecules using the trained model.

#### Steps
1. Prepare new molecular images in `data/new_molecules/`.
2. Run prediction:
   ```bash
   python predict.py --image_dir data/new_molecules/ \
                     --model_ckpt checkpoints/finetuned_model.pth \
                     --output_csv predictions.csv \
                     --task_type classification  # or "regression"
   ```
3. Output: Predicted scores saved in CSV with molecule IDs and property predictions.


## 6. Reproducing Paper Results
### A. Pre-training on 10M Unlabeled Molecules
1. Download the pre-training dataset (SMILES from PubChem, as used in ChemBERTa).
2. Convert SMILES to images using RDKit:
   ```bash
   python scripts/convert_smiles_to_images.py --smiles_file pubchem_smiles.txt --output_dir data/pretrain/
   ```
3. Run pre-training:
   ```bash
   python main.py --mode pretrain \
                   --data_dir data/pretrain/ \
                   --num_epochs 600 \
                   --batch_size 256 \
                   --lr 5e-5 \
                   --save_dir checkpoints/pretrain_large/
   ```

### B. Fine-tuning on Classification Tasks (e.g., BBBP)
```bash
python main.py --mode finetune \
               --data_dir data/BBBP/ \
               --task_type classification \
               --num_classes 2 \
               --load_ckpt checkpoints/pretrain_large/molrl_pretrain.pth \
               --n_epochs_after 150
```

## 7. Custom Dataset Guide
### Step 1: Data Preparation
- **Images**: Generate 240x240 molecular images from SMILES using RDKit.
- **Labels**: Prepare CSV files with `mol_id` and corresponding properties.
- **Split**: Use scaffold splitting for train/val/test sets.

### Step 2: Adjust Configuration
Update `config.yaml` with dataset-specific parameters:
```yaml
dataset:
  name: custom_dataset
  task_type: classification  # or "regression"/"dta"
  num_classes: 2  # or regression targets
  data_path: data/custom/
```

### Step 3: Run Training
```bash
python main.py --config config.yaml \
               --mode finetune \
               --load_ckpt checkpoints/pretrain/molrl_pretrain.pth
```


## Model Architecture Highlights
- **Graph Structure Bootstrapping**: Learn dynamic graph structures from molecular image patches using weighted graph isomorphism networks (GINs).
- **Hierarchical Attention**: Combine local routing attention (spatial adjacency) and global graph attention (functional group analysis).
- **Self-Supervised Learning**: Use BYOL framework with dual views (anchor/learner) and data augmentation (masking, graph perturbation).


Code inspired by the paper: 
Shan D, Luo Y, Yuan H, et al. MolRL: Self-Supervised Molecular Image Representation Learning via Graph Structure Bootstrapping[J]. Pattern Recognition, 2025: 112773.

@article{shan2025molrl,
  title={MolRL: Self-Supervised Molecular Image Representation Learning via Graph Structure Bootstrapping},
  author={Shan, Dongjing and Luo, Yamei and Yuan, Hong and Mao, Jiashun and Wang, Limin},
  journal={Pattern Recognition},
  pages={112773},
  year={2025},
  publisher={Elsevier}
}

Happy molecular learning! 🔬
