import sympy as sp
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pathlib import Path
import numpy as np
import pandas as pd
from utils import build_feature_matrix_and_labels
import joblib
from warnings import filterwarnings
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--full_model_type", type=str, default='large',
                    choices=['large', 'medium', 'small'])
parser.add_argument("--top_k", type=int, default=5)

args = parser.parse_args()

filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
save_path = f"result/symbolic_regression_{args.full_model_type}_{args.top_k}.log"
logger.addHandler(logging.FileHandler(save_path))


dataset_dir = Path("hyperview2")

full_model_type = args.full_model_type
top_k = args.top_k

target_parameters = ["Fe", "Zn", "B", "Cu", "S", "Mn"]
if full_model_type == 'large':
    random_state = 42
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

elif full_model_type == 'medium':
    random_state = 42
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

elif full_model_type == 'small':
    random_state = 42
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

# 1) read GT (train_gt.csv)
train_gt_df = pd.read_csv(
    dataset_dir / "train_gt.csv")[target_parameters]
# log transform: log10(gt + 1)
train_gt_df[target_parameters] = np.log10(
    train_gt_df[target_parameters] + 1)

N_total = len(train_gt_df)
all_indices = np.arange(N_total)

label_mean = train_gt_df.iloc[all_indices].mean().values
label_std = train_gt_df.iloc[all_indices].std().values

# --------------------------------------------------------------------------------
# 2) create feature, label matrix
# --------------------------------------------------------------------------------
X_train, Y_train = build_feature_matrix_and_labels(
    all_indices, train_gt_df, dataset_dir, label_mean, label_std,
    is_train=True, rotate_aug=True,
    hsi_size=tuple(hsi_size), msi_size=tuple(msi_size),
    hsi_fft_ratio=hsi_fft_ratio, msi_fft_ratio=msi_fft_ratio, hsi_mean_fft_ratio=hsi_mean_fft_ratio,
    msi_mean_fft_ratio=msi_mean_fft_ratio, msi_max_fft_ratio=msi_max_fft_ratio,
    hsi_max_fft_ratio=hsi_max_fft_ratio, add_extra_features=add_extra_features,
    use_fe_features=use_fe_features, use_zn_features=use_zn_features, use_s_features=use_s_features,
    use_mn_advanced_features=use_mn_advanced_features, use_zn_advanced_features=use_zn_advanced_features,
    use_spatial_features=use_spatial_features,
    full_model_type=full_model_type, target_parameters=target_parameters)
shap_vals = np.load(f"result/shap_values_{full_model_type}.npz")
train_idx = np.arange(int(len(X_train)*0.8))
test_idx = np.arange(int(len(X_train)*0.8), len(X_train))
X_test = X_train[test_idx]
Y_test = Y_train[test_idx]
X_train = X_train[train_idx]
Y_train = Y_train[train_idx]


# Order matches Y_train column order
targets = ["Fe", "Zn", "B", "Cu", "S", "Mn"]

# ------------------------------------------------------------------
# 1) SymPy mappings
# ------------------------------------------------------------------
extra_sympy = {
    "max":  sp.Max,
    "min":  sp.Min,
    "tanh": sp.tanh,
    "inv": lambda x: 1 / x,
    "cos2": lambda x: sp.cos(x)**2,
}

# ------------------------------------------------------------------
# 2) Nesting constraints
# ------------------------------------------------------------------
nested_constraints = {
    "tanh":  {"tanh": 0},
    "log":   {"log": 1},
    "square": {"square": 1, "cube": 1},
    "cube": {"square": 1, "cube": 1},
}

# ------------------------------------------------------------
# 1) PySR default parameters
# ------------------------------------------------------------
pysr_params = dict(
    # ---------------- Island model ----------------
    populations=16,        # number of islands
    population_size=100,       # individuals per island
    ncycles_per_iteration=500,      # cycles per iteration (within island)
    niterations=50,       # number of iterations (akin to generations)

    # ---------------- Operator set -----------------
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "log", "sin", "cos", "abs", "tanh", "square", "cube", "cos2(x)=cos(x)^2", "inv(x) = 1/x"
    ],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
    },
    extra_sympy_mappings=extra_sympy,
    select_k_features=None,
    nested_constraints=nested_constraints,

    # ---------------- Complexity / cost --------------
    maxsize=50,        # total number of nodes in the tree
    maxdepth=30,        # maximum tree depth
    parsimony=0.0005,

    # ---------------- Loss and constant optimization ---------
    loss="L2DistLoss()",   # MSE
    # optimize constants (default True)
    should_optimize_constants=True,

    # ---------------- Misc ----------------------
    random_state=None,      # overridden per target as 42+i
    progress=False,
    verbosity=0,
    warm_start=True,
    precision=64,        # floating-point precision
    turbo=True       # experimental acceleration
)

# ------------------------------------------------------------
# 2) Training loop (largely similar to gplearn code)
# ------------------------------------------------------------
models, equations = {}, {}

for i, tgt in enumerate(targets):
    # Select top-k features
    importance = np.abs(shap_vals[tgt]).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:top_k]

    X_train_top10 = X_train[:, top_idx]

    logger.info(f"\n### [{tgt}] PySR training started ###")
    params = pysr_params.copy()
    params["random_state"] = 42 + i

    model = PySRRegressor(**params)
    model.fit(X_train_top10, Y_train[:, i])
    logger.info(f"[{tgt}] PySR training completed")

    # PySR stores multiple expressions in a DataFrame → retrieve the best one
    best_eqn = model.get_best()              # Series: equation, score, complexity …
    expr = best_eqn["equation"]

    # Prediction and performance
    X_test_top10 = X_test[:, top_idx]
    logger.info(f"{tgt} Eq: {expr}")
    try:
        y_pred = model.predict(X_test_top10)
        r2 = r2_score(Y_test[:, i], y_pred)
    except:
        r2 = 0

    logger.info(f"R²={r2:.4f}")

    models[tgt] = model
    equations[tgt] = expr
    joblib.dump(
        model, f"model/symbolic_regression_{args.full_model_type}_{args.top_k}_{tgt}.pkl")
