"""
shap_analyze_random_forest
-------------------------------------------------
• RandomForestRegressor → SHAP analysis codes
• SHAP values are cached in .npz (reuse if exists)
• summary_bar / beeswarm PNGs are saved for each target (column_names)
• global importance table → CSV + bar plot
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from utils import build_feature_matrix_and_labels
import argparse

parser = argparse.ArgumentParser(
    description="Random Forest on HyperView2 Satellite Data")
parser.add_argument("--full_model_type", type=str, default='large',
                    help="large | medium | small")
args = parser.parse_args()
# -------------------------------------------------
# 0. settings
# -------------------------------------------------
model_type = args.full_model_type
DATA_ROOT = Path("hyperview2")
MODEL_PATH = Path(
    f"model/extra_trees_regressor_42_{model_type}.joblib")
SHAP_CACHE = Path(f"result/shap_values_{model_type}.npz")  # ver2
PLOT_DIR = Path("shap_plots")
PLOT_DIR.mkdir(exist_ok=True)
TARGETS = ["Fe", "Zn", "B", "Cu", "S", "Mn"]
if model_type == 'large':
    # python3 random_forest.py --n_trees 3000 --hsi_fft_ratio 0.1 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --hsi_mean_fft_ratio 0 --hsi_max_fft_ratio 0 --top_k 0 --is_full_train True --add_extra_features True --msi_max_fft_ratio 0.2 --msi_mean_fft_ratio 0.2 --full_model_type large --use_zn_features False --experiment_size_ranges False --random_state 42 --dataset_dir hyperview2 --use_s_features True
    random_state = 42
    top_k = 0
    hsi_size = [6, 6]
    msi_size = [7, 7]
    hsi_fft_ratio = 0.1
    msi_fft_ratio = 0.2
    msi_mean_fft_ratio = 0.2
    msi_max_fft_ratio = 0.2
    add_extra_features = True
    use_fe_features = True
    use_zn_features = False
    hsi_mean_fft_ratio = 0.
    hsi_max_fft_ratio = 0.
    use_s_features = True
    use_mn_advanced_features = False
    use_zn_advanced_features = False
    use_spatial_features = False

elif model_type == 'medium':
    # python3 random_forest.py --max_depth 30 --n_trees 300 --hsi_fft_ratio 0.2 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --hsi_mean_fft_ratio 0. --hsi_max_fft_ratio 0. --top_k 0 --is_full_train True --add_extra_features True --msi_max_fft_ratio 0.2 --msi_mean_fft_ratio 0.2 --full_model_type medium --use_zn_features False --experiment_size_ranges False --random_state 42 --dataset_dir hyperview2 --use_s_features True --use_mn_advanced_features True
    random_state = 42
    top_k = 0
    hsi_size = [6, 6]
    msi_size = [7, 7]
    hsi_fft_ratio = 0.2
    msi_fft_ratio = 0.2
    msi_mean_fft_ratio = 0.2
    msi_max_fft_ratio = 0.2
    add_extra_features = True
    use_fe_features = True
    use_zn_features = False
    hsi_mean_fft_ratio = 0.
    hsi_max_fft_ratio = 0.
    use_s_features = True
    use_mn_advanced_features = True
    use_zn_advanced_features = False
    use_spatial_features = False

elif model_type == 'small':
    # python3 random_forest.py --max_depth 20 --n_trees 300 --hsi_fft_ratio 0.3 --msi_fft_ratio 0.2 --hsi_size 6 6 --msi_size 7 7 --use_fe_features True --hsi_mean_fft_ratio 0. --hsi_max_fft_ratio 0. --top_k 0 --is_full_train True --add_extra_features True --msi_max_fft_ratio 0.1 --msi_mean_fft_ratio 0.1 --full_model_type small --use_zn_features True --experiment_size_ranges False --random_state 42 --dataset_dir hyperview2 --use_s_features True
    random_state = 42
    top_k = 0
    hsi_size = [6, 6]
    msi_size = [7, 7]
    hsi_fft_ratio = 0.3
    msi_fft_ratio = 0.2
    msi_mean_fft_ratio = 0.1
    msi_max_fft_ratio = 0.1
    add_extra_features = True
    use_fe_features = True
    use_zn_features = True
    hsi_mean_fft_ratio = 0.
    hsi_max_fft_ratio = 0.
    use_s_features = True
    use_mn_advanced_features = False
    use_zn_advanced_features = False
    use_spatial_features = False

# 0. load model
model = joblib.load(MODEL_PATH)
# -------------------------------------------------
# 1. load data & build feature matrix
# -------------------------------------------------
train_gt_df = pd.read_csv(DATA_ROOT / "train_gt.csv")[TARGETS]
train_gt_df[TARGETS] = np.log10(train_gt_df[TARGETS] + 1)

idx_all = np.arange(len(train_gt_df))
label_mean = train_gt_df.mean().values
label_std = train_gt_df.std().values

# reuse: build_feature_matrix_and_labels is already defined
X_train, Y_train = build_feature_matrix_and_labels(
    idx_all, train_gt_df, DATA_ROOT, label_mean, label_std, is_train=True,
    hsi_size=hsi_size,
    msi_size=msi_size,
    hsi_fft_ratio=hsi_fft_ratio,
    msi_fft_ratio=msi_fft_ratio,
    msi_mean_fft_ratio=msi_mean_fft_ratio,
    msi_max_fft_ratio=msi_max_fft_ratio,
    hsi_mean_fft_ratio=hsi_mean_fft_ratio,
    hsi_max_fft_ratio=hsi_max_fft_ratio,
    add_extra_features=add_extra_features,
    use_fe_features=use_fe_features,
    use_zn_features=use_zn_features,
    use_s_features=use_s_features,
    use_mn_advanced_features=use_mn_advanced_features,
    use_zn_advanced_features=use_zn_advanced_features,
    use_spatial_features=use_spatial_features)

# create feature names (optional)
feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]


# -------------------------------------------------
# 2. calculate SHAP values or read from cache
# -------------------------------------------------

print("[INFO] Computing SHAP values … (may take a while)")
explainer = shap.TreeExplainer(model)
# memory saving: select 5~10k samples if needed
# e.g. X_sample = shap.sample(X_train, 8000, random_state=0)
raw_vals = explainer.shap_values(X_train)

# --- ★ 3-D block ★ ---
if isinstance(raw_vals, np.ndarray) and raw_vals.ndim == 3:
    shap_vals = [raw_vals[:, :, t] for t in range(raw_vals.shape[2])]
else:
    shap_vals = raw_vals

# save cache
np.savez_compressed(
    SHAP_CACHE, **{t: v for t, v in zip(TARGETS, shap_vals)})
print(f"shap_vals shape: {len(shap_vals)}")

# -------------------------------------------------
# 3. Calculate global importance table
# -------------------------------------------------
# (N_targets, N_features) → (N_features,) via mean(|SHAP|)
abs_means = np.vstack([np.abs(v).mean(axis=0) for v in shap_vals])
global_imp = abs_means.mean(axis=0)  # mean: shape (F,)

imp_df = (
    pd.DataFrame({"feature": feature_names, "importance": global_imp})
      .sort_values("importance", ascending=False)
      .reset_index(drop=True)
)

imp_df.to_csv(
    f"result/shap_feature_importance_{model_type}.csv", index=False, encoding="utf-8")
print(
    f"[INFO] Global importance table → result/shap_feature_importance_{model_type}.csv")
