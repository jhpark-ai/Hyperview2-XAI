## Reproducibility Guide — EASi 2025 (ECAI Workshop)

This repository contains the official code and instructions to reproduce the results of the paper:

> From Black Box to White Box: Explainable Soil Contaminant Prediction using SHAP and Symbolic Regression  
> Junghwan Park, Seokhyun Chin, Darongsae Kwon (Equal contribution: Park & Chin; Corresponding author: Kwon)  
> EASi 2025 @ ECAI 2025

If you use this code, please cite our paper.

---

## Environment

Tested with:

- python = 3.11.0
- scikit-learn = 1.7.0
- numpy = 2.2.6
- pandas = 2.3.0
- joblib = 1.5.1
- SHAP = 0.48.0
- PySR = 1.5.9

## Data

- Place the HyperView2 dataset under a directory passed via `--dataset_dir` (examples below use `hyperview2/`).
- We assume MSI/HSI patches and metadata follow the structure expected by `train.py`.

Data layout:

```text
hyperview2/
  train/
    hsi_satellite/*.npz
    msi_satellite/*.npz
  test/
    hsi_satellite/*.npz
    msi_satellite/*.npz
train_gt.csv
```

## Workflow Overview

### 1. Initial Training by Patch Size
Run the initial training using the provided script `train.py`. Separate models are trained for each patch size, controlled by `--full_model_type`.

- **Large (5×5 < size ≤ 7×7)**
```bash
python3 train.py --n_trees 2000 --hsi_fft_ratio 0.1 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --hsi_mean_fft_ratio 0.0 --hsi_max_fft_ratio 0.0 --top_k 0 --add_extra_features True --msi_max_fft_ratio 0.2 --msi_mean_fft_ratio 0.2 --full_model_type large --use_zn_features False --dataset_dir hyperview2 --use_s_features True --use_mn_advanced_features False --use_zn_advanced_features False --use_spatial_features False --random_state 42
```

- **Medium (3×3 < size ≤ 5×5)**
```bash
python3 train.py --n_trees 200 --max_depth 20 --hsi_fft_ratio 0.2 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --hsi_mean_fft_ratio 0.0 --hsi_max_fft_ratio 0.0 --top_k 0 --add_extra_features True --msi_max_fft_ratio 0.2 --msi_mean_fft_ratio 0.2 --full_model_type medium --use_zn_features False --dataset_dir hyperview2 --use_s_features True --use_mn_advanced_features True --use_zn_advanced_features False --use_spatial_features False --random_state 42
```

- **Small (≤3×3)**
```bash
python3 train.py --n_trees 100 --max_depth 20 --hsi_fft_ratio 0.3 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --hsi_mean_fft_ratio 0.0 --hsi_max_fft_ratio 0.0 --top_k 0 --add_extra_features True --msi_max_fft_ratio 0.1 --msi_mean_fft_ratio 0.1 --full_model_type small --use_zn_features True --dataset_dir hyperview2 --use_s_features True --use_mn_advanced_features False --use_zn_advanced_features False --use_spatial_features False --random_state 42
```

### 2. SHAP Analysis and Feature Importance
Perform SHAP analysis on the trained models to identify feature importances. Execute `shap_analyze_forest.py`, which generates and saves SHAP feature-importance scores.

Example:
```bash
python shap_analyze_forest.py --full_model_type <large|medium|small>
```

### 3-1. Refined Training using SHAP Feature Selection
Retrain the models using SHAP-informed feature selection. Features are selected based on SHAP-derived importance scores.

- **Large (5×5 < size ≤ 7×7)**
```bash
python3 train.py --n_trees 2000 --hsi_fft_ratio 0.1 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --top_k 60 --add_extra_features True --msi_max_fft_ratio 0.2 --msi_mean_fft_ratio 0.2 --full_model_type large --use_zn_features False --dataset_dir hyperview2 --use_s_features True --use_mn_advanced_features False --use_zn_advanced_features False --use_spatial_features False --random_state 42
```

- **Medium (3×3 < size ≤ 5×5)**
```bash
python3 train.py --n_trees 200 --max_depth 20 --hsi_fft_ratio 0.2 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --top_k 50 --add_extra_features True --msi_max_fft_ratio 0.2 --msi_mean_fft_ratio 0.2 --full_model_type medium --use_zn_features False --dataset_dir hyperview2 --use_s_features True --use_mn_advanced_features True --use_zn_advanced_features False --use_spatial_features False --random_state 42
```

- **Small (≤3×3)**
```bash
python3 train.py --n_trees 100 --max_depth 20 --hsi_fft_ratio 0.3 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --top_k 140 --add_extra_features True --msi_max_fft_ratio 0.1 --msi_mean_fft_ratio 0.1 --full_model_type small --use_zn_features True --dataset_dir hyperview2 --use_s_features True --use_mn_advanced_features False --use_zn_advanced_features False --use_spatial_features False --random_state 42
```

### 3-2. Symbolic Regression Training
Fit symbolic expressions on the selected features.

Example:
```bash
python symbolic_regression.py --top_k 10 --full_model_type <large|medium|small>
```

## License
This project is released under the [MIT License](./LICENSE.md).
