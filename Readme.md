# SR4ZCT: Self-supervised Through-plane Resolution Enhancement for CT Images

This repository contains the implementation of SR4ZCT (Self-supervised Through-plane Resolution Enhancement for CT Images with Arbitrary Resolution and Overlap) with network architecture comparison and interpolation method analysis.

Paper: Shi, J., Pelt, D. M., & Batenburg, K. J. (2023, October). SR4ZCT: self-supervised through-plane resolution enhancement for CT images with arbitrary resolution and overlap. In International Workshop on Machine Learning in Medical Imaging (pp. 52-61). Cham: Springer Nature Switzerland.

## Overview

This project implements and extends the SR4ZCT method proposed by Shi et al. (2023), focusing on two key research questions:

1. **Network Architecture Comparison**: Replacing MS-DNet with UNet or ResNet for potentially better reconstruction quality
2. **Interpolation Method Enhancement**: Using cubic interpolation instead of linear interpolation for better structural detail preservation

## Features

- Complete SR4ZCT implementation with three neural network architectures:
  - MS-DNet (original baseline)
  - UNet (encoder-decoder with skip connections)
  - ResNet (residual learning approach)
- Dual interpolation methods:
  - Linear interpolation (original method)
  - Cubic interpolation (enhanced method)
- Comprehensive evaluation framework with PSNR and SSIM metrics
- Automated figure generation for paper-quality results
- 3D phantom generation and CT reconstruction pipeline

## Installation

### Dependencies

```bash
pip install torch torchvision numpy scipy scikit-image matplotlib tqdm
pip install astra-toolbox trimesh pyvista pandas pathlib
```

### Hardware Requirements

- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- 10GB+ free disk space for datasets and results

## Quick Start

### 1. Generate 3D Phantom Data

```bash
python phantom3D_adjust.py
```

This creates a 3D phantom with ellipsoids and curved vessel structures, saved as `phantom_dense.npy`.

### 2. Generate Sinograms

```bash
python sinogram3D.py
```

Creates parallel-beam sinograms for CT reconstruction.

### 3. 3D Reconstruction

```bash
python reconstruct3D.py
```

Performs FBP or SIRT reconstruction, generating the base volume for SR4ZCT training.

### 4. Create Training Datasets

**Linear Interpolation (Baseline):**
```bash
python create_4dataset_linear.py
```

**Cubic Interpolation (Enhanced):**
```bash
python create_4dataset_cubic.py
```

Both scripts generate training datasets in `sr4zct_exp_dataset/` with:
- `recon_low.npy`: Reference high-resolution volume
- `recon_low_vertical.npy`: Vertically degraded training data
- `recon_low_horizontal.npy`: Horizontally degraded training data
- `recon_low_test.npy`: Test volume for evaluation

### 5. Train Neural Networks

**MS-DNet (Baseline):**
```bash
python MSDNet.py
```

**UNet:**
```bash
python UNet.py
```

**ResNet:**
```bash
python ResNet.py
```

Each training script will:
- Train the network for 100 epochs
- Save intermediate results every 5 epochs
- Generate final enhanced volumes (`output_cor.npy`, `output_sag.npy`)
- Create training loss plots

### 6. Evaluation and Comparison

**Generate Paper-Format Results:**
```bash
python Compare_eval.py
```

**Create Comparison Figures:**
```bash
python Compare_figure.py
```

## Project Structure

```
SR4ZCT/
├── phantom3D_adjust.py          # 3D phantom generation
├── sinogram3D.py                # Sinogram generation
├── reconstruct3D.py             # CT reconstruction
├── create_4dataset_linear.py    # Linear interpolation dataset
├── create_4dataset_cubic.py     # Cubic interpolation dataset
├── MSDNet.py                    # MS-DNet implementation
├── UNet.py                      # UNet implementation
├── ResNet.py                    # ResNet implementation
├── Compare_eval.py              # Quantitative evaluation
├── Compare_figure.py            # Figure generation
├── sr4zct_exp_dataset/          # Generated training data
├── msdnet_training_results/     # MS-DNet results
├── unet_training_results/       # UNet results
├── resnet_training_results/     # ResNet results
├── figures_coronal/             # Coronal view comparisons
├── figures_sagittal/            # Sagittal view comparisons
└── output_3d/                   # CT reconstruction outputs
```

 

 
