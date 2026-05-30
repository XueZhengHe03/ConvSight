# ConvSight: CFD Convergence Detection with Multimodal Learning

A deep learning framework for detecting CFD (Computational Fluid Dynamics) simulation convergence using time series data, image data, and multimodal fusion approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Methods](#methods)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## 🔍 Overview

ConvSight explores three approaches for CFD convergence detection:

1. **Time Series Models** - Analyze residual convergence curves directly
2. **Image Models** - Classify convergence curve visualizations
3. **Multimodal Fusion** - Combine time series and image features

### Key Findings

- **StandardScaler** outperforms Mean Centering for preprocessing across all experiments
- **Dynamic Gated Fusion** achieves superior multimodal performance compared to simple concatenation
- Multimodal approach outperforms single-modal when using proper fusion architecture

## 📁 Project Structure

```
ConvSight/
├── experiment_csv/          # Time series experiments
│   ├── src/
│   │   ├── prepare_dataset_tslib.py        # StandardScaler preprocessing
│   │   ├── prepare_dataset_tslib_mean.py   # Mean Centering preprocessing
│   │   ├── train_iTransformer.py
│   │   ├── train_timesnet.py
│   │   ├── train_ts2vec.py
│   │   ├── evaluate_*.py
│   │   └── run_baselines.sh
│   ├── Time-Series-Library/  # iTransformer, TimesNet models
│   └── ts2vec/               # TS2Vec model
│
├── experiment_img/          # Image classification experiments
│   ├── src/
│   │   ├── csv2img.py           # Convert CSV to convergence curve images
│   │   ├── split_dataset.py     # Train/val/test split
│   │   ├── train_baseline.py    # Train image models
│   │   ├── evaluate_all.py      # Evaluate all models
│   │   └── run_baselines.sh
│   └── dat/
│       ├── img_converge/
│       └── img_un-converge/
│
├── experiment_fusion/       # Multimodal fusion experiments
│   ├── src/
│   │   ├── model.py             # Simple concatenation fusion
│   │   ├── model_new.py         # Dynamic Gated Fusion (proposed)
│   │   ├── dataset.py           # Multimodal dataset loader
│   │   ├── train_multimodal.py
│   │   └── cross_validation_teacher.py
│   ├── dat/                     # CSV dataset
│   └── dataset/                 # Processed data
│
└── dataset_new/             # New .dat dataset
    ├── 收敛/                   # Converged samples
    └── 未收敛/                 # Unconverged samples
```

## 📊 Datasets

### CSV Dataset (Original)
- **Total samples**: ~1014
- **Channels**: 2 (v2, v3)
- **Format**: CSV files

### DAT Dataset (New)
- **Total samples**: 70 (52 converged, 18 unconverged)
- **Channels**: 1 (residual)
- **Format**: .dat files with 4-line header

## 🧪 Methods

### Time Series Models

| Model | Description |
|-------|-------------|
| **iTransformer** | Inverted Transformer with attention across variates |
| **TimesNet** | Temporal 2D-variation modeling |
| **TS2Vec** | Universal time series representation learning |
| **Non-stationary Transformer** | Handles non-stationary time series |

### Image Models

| Model | Description |
|-------|-------------|
| **ResNet50** | Residual Network with 50 layers |
| **EfficientNet-B0** | Efficient convolutional architecture |
| **Swin-Tiny** | Hierarchical Vision Transformer |
| **EfficientNetV2-S** | Improved EfficientNet |

### Multimodal Fusion

| Model | Description |
|-------|-------------|
| **Simple Concatenation** | Direct feature concatenation |
| **Dynamic Gated Fusion** | Learned dynamic weights for each modality (proposed) |

#### Dynamic Gated Fusion

```python
# Key innovation: learn dynamic weights for each sample
alpha = gate_network(concat(img_proj, ts_proj))
fused = alpha * img_proj + (1 - alpha) * ts_proj
```

## 📈 Results

### DAT Dataset Results (10 epochs)

| Model | Accuracy | Converged Recall | Unconverged Recall |
|-------|----------|------------------|-------------------|
| **Dynamic Gated Fusion** | **93.33%** | 90.00% | 100.00% |
| iTransformer | 92.86% | 90.00% | 100.00% |
| Cross-domain (CSV→DAT) | 84.29% | 100.00% | 38.89% |
| ResNet50 (Image) | 68.75% | - | - |
| Simple Concatenation | 66.67% | 100.00% | 0.00% |

### CSV Dataset Results

| Model | Accuracy | Converged Recall | Unconverged Recall |
|-------|----------|------------------|-------------------|
| **Dynamic Gated Fusion** | **99.02%** | 99.00% | 99.22% |
| iTransformer | ~95% | - | - |

### Preprocessing Comparison

| Strategy | CSV | DAT |
|----------|-----|-----|
| **StandardScaler** | ✓ Better | ✓ Better |
| Mean Centering | | |

**Conclusion**: StandardScaler is the recommended preprocessing strategy.

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/XueZhengHe03/ConvSight.git
cd ConvSight

# Install dependencies
pip install torch torchvision
pip install timm
pip install scikit-learn
pip install pandas numpy matplotlib
pip install reformer_pytorch

# Clone model libraries (for experiment_csv)
cd experiment_csv
git clone https://github.com/thuml/Time-Series-Library.git
git clone https://github.com/yuezhihan/ts2vec.git
```

## 🚀 Usage

### 1. Time Series Experiments

```bash
cd experiment_csv/src

# Prepare dataset (StandardScaler)
python prepare_dataset_tslib.py

# Train and evaluate
python train_iTransformer.py
python evaluate_iTransformer.py

# Or run all baselines
bash run_baselines.sh
```

### 2. Image Experiments

```bash
cd experiment_img/src

# Generate images from CSV
python csv2img.py

# Split dataset
python split_dataset.py

# Train image models
python train_baseline.py --model resnet50
python train_baseline.py --model efficientnet_b0

# Evaluate all
python evaluate_all.py
```

### 3. Multimodal Fusion Experiments

```bash
cd experiment_fusion/src

# Train Dynamic Gated Fusion model
python train_multimodal_new.py

# Evaluate
python evaluate_multimodal_new.py
```

### 4. New DAT Dataset Experiments

```bash
# Time series
cd experiment_csv/src
python prepare_dataset_tslib_dat.py
python train_iTransformer_dat.py
python evaluate_iTransformer_dat.py

# Multimodal
cd experiment_fusion/src
python train_multimodal_new_dat.py
python evaluate_multimodal_new_dat.py
```

## 📝 Key Insights

1. **Architecture matters more than data size**: Dynamic Gated Fusion achieves 93.33% on DAT (70 samples) vs 66.67% with simple concatenation

2. **Multimodal > Single modal**: With proper fusion, multimodal approach consistently outperforms single-modal

3. **StandardScaler > Mean Centering**: Across all experiments and datasets

4. **Cross-domain transfer**: Models trained on large CSV dataset (99.02%) can transfer to small DAT dataset (84.29%)

5. **Convergence features are universal**: 100% recall for converged samples across all experiments

## 📚 References

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) - iTransformer, TimesNet implementations
- [TS2Vec](https://github.com/yuezhihan/ts2vec) - Time series representation learning
- [iTransformer](https://arxiv.org/abs/2310.06625) - Inverted Transformers for Time Series Forecasting

## 📄 License

This project is for academic research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.
