import joblib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utils import build_feature_matrix_and_labels
from sklearn.ensemble import (
    ExtraTreesRegressor,
)

from warnings import filterwarnings

filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_ensemble_model(args):
    """create ensemble model"""
    base_params = {
        'random_state': args.random_state,
        'n_jobs': -1 if hasattr(args, 'n_jobs') else -1
    }

    return ExtraTreesRegressor(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        criterion='squared_error',
        **base_params
    )


def get_model_description(model_name):
    """return model description"""
    descriptions = {
        'extra_trees': 'Extra Trees',
    }
    return descriptions.get(model_name, f"Unknown model: {model_name}")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Random Forest on HyperView2 Satellite Data")
    parser.add_argument("--model", type=str, default="extra_trees",
                        choices=['extra_trees'],
                        help="ensemble model to use")
    parser.add_argument("--dataset_dir", type=str, default="hyperview2",
                        help="HyperView2 dataset directory (e.g. /workspace/data/hyperview2)")
    parser.add_argument("--use_k_fold", type=str2bool, default=True,
                        help="K-Fold cross-validation (default: holdout)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="K-Fold split count (default: 5)")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio (default: 0.15)")
    parser.add_argument("--n_trees", type=int, default=1000,
                        help="Random Forest tree count (default: 1000)")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Random Forest max depth (default: None)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state (default: 42)")
    parser.add_argument("--rotate_aug", type=str2bool, default=True,
                        help="90/180/270° patch rotation augmentation (train only)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="number of top SHAP features to use")
    parser.add_argument("--hsi_size", type=int, nargs=2, default=[6, 6],
                        help="HSI patch size (height width)")
    parser.add_argument("--msi_size", type=int, nargs=2, default=[7, 7],
                        help="MSI patch size (height width)")
    parser.add_argument("--min_samples_for_size", type=int, default=50,
                        help="minimum sample count for size-wise experiment")
    parser.add_argument("--hsi_fft_ratio", type=float, default=0.2,
                        help="HSI FFT ratio (default: 0.2)")
    parser.add_argument("--msi_fft_ratio", type=float, default=0.2,
                        help="MSI FFT ratio (default: 0.2)")
    parser.add_argument("--msi_mean_fft_ratio", type=float, default=0.2,
                        help="MSI mean FFT ratio (default: 0.2)")
    parser.add_argument("--msi_max_fft_ratio", type=float, default=0.2,
                        help="MSI max FFT ratio (default: 0.2)")
    parser.add_argument("--hsi_mean_fft_ratio", type=float, default=0.,
                        help="HSI mean FFT ratio (default: 0.2)")
    parser.add_argument("--hsi_max_fft_ratio", type=float, default=0.,
                        help="HSI max FFT ratio (default: 0.2)")
    parser.add_argument("--add_extra_features", type=str2bool, default=True,
                        help="use additional specialized features (default: True)")
    parser.add_argument("--use_fe_features", type=str2bool, default=True,
                        help="use Fe specialized features (default: True)")
    parser.add_argument("--use_zn_features", type=str2bool, default=False,
                        help="use Zn specialized features (default: False)")
    parser.add_argument("--use_s_features", type=str2bool, default=False,
                        help="use spectral index features (default: False)")
    parser.add_argument("--use_mn_advanced_features", type=str2bool, default=False,
                        help="use Mn advanced features (default: False)")
    parser.add_argument("--use_zn_advanced_features", type=str2bool, default=False,
                        help="use Zn advanced features (default: False)")
    parser.add_argument("--use_spatial_features", type=str2bool, default=False,
                        help="use spatial texture features (default: False)")
    parser.add_argument("--full_model_type", type=str, default='base',
                        help="large | medium | small | base")
    parser.add_argument("--target_parameters", type=str, nargs='+',
                        default=["Fe", "Zn", "B", "Cu", "S", "Mn"],
                        choices=["Fe", "Zn", "B", "Cu", "S", "Mn"],
                        help="Training target parameters (1~6 choices)")
    return parser.parse_args()


def full_train(args):
    np.random.seed(args.random_state)

    # check folder
    Path("result").mkdir(exist_ok=True)
    Path("model").mkdir(exist_ok=True)

    print(f"train target parameters: {args.target_parameters}")

    # 1) read GT (train_gt.csv)
    train_gt_df = pd.read_csv(
        args.dataset_dir / "train_gt.csv")[args.target_parameters]
    # log transform: log10(gt + 1)
    train_gt_df[args.target_parameters] = np.log10(
        train_gt_df[args.target_parameters] + 1)

    N_total = len(train_gt_df)
    all_indices = np.arange(N_total)

    label_mean = train_gt_df.iloc[all_indices].mean().values
    label_std = train_gt_df.iloc[all_indices].std().values

    # --------------------------------------------------------------------------------
    # 2) create feature, label matrix
    # --------------------------------------------------------------------------------
    X_train, Y_train = build_feature_matrix_and_labels(
        all_indices, train_gt_df, args.dataset_dir, label_mean, label_std,
        is_train=True, rotate_aug=args.rotate_aug,
        hsi_size=tuple(args.hsi_size), msi_size=tuple(args.msi_size),
        hsi_fft_ratio=args.hsi_fft_ratio, msi_fft_ratio=args.msi_fft_ratio, hsi_mean_fft_ratio=args.hsi_mean_fft_ratio,
        msi_mean_fft_ratio=args.msi_mean_fft_ratio, msi_max_fft_ratio=args.msi_max_fft_ratio,
        hsi_max_fft_ratio=args.hsi_max_fft_ratio, add_extra_features=args.add_extra_features,
        use_fe_features=args.use_fe_features, use_zn_features=args.use_zn_features, use_s_features=args.use_s_features,
        use_mn_advanced_features=args.use_mn_advanced_features, use_zn_advanced_features=args.use_zn_advanced_features,
        use_spatial_features=args.use_spatial_features,
        full_model_type=args.full_model_type, target_parameters=args.target_parameters)

    if args.top_k != 0:
        shap_feature_importance = pd.read_csv(
            f"result/shap_feature_importance_{args.full_model_type}.csv")
        top_common_idx = shap_feature_importance['feature'].apply(
            lambda x: int(x[5:])).iloc[:args.top_k].to_list()

        X_train = X_train[:, top_common_idx]

    print(
        f"  [Data shapes] X_train: {X_train.shape}, Y_train: {Y_train.shape if Y_train is not None else None}")

    # 3) define and train ensemble model
    # --------------------------------------------------------------------------------
    model_desc = get_model_description(args.model)
    print(f"  Using {model_desc}")
    model = create_ensemble_model(args)
    print(f"  Training {args.model} …")
    model.fit(X_train, Y_train)

    # --------------------------------------------------------------------------------
    # 4) calculate validation score (MSE per parameter & average)
    # --------------------------------------------------------------------------------
    Y_pred_val = model.predict(X_train)  # shape = (N_selected_params)

    # base mse for all parameters
    all_basemse = {
        "Fe": 3.43226857e+03, "Zn": 4.27235370e+00, "B": 5.73451430e-02,
        "Cu": 2.02348293e-01, "S": 1.83215779e+02, "Mn": 4.42813125e+02
    }
    # extract basemse for selected parameters
    basemse = np.array([all_basemse[param]
                       for param in args.target_parameters])

    Y_pred_val = Y_pred_val * label_std + label_mean
    Y_pred_val = 10**Y_pred_val - 1
    Y_train = Y_train * label_std + label_mean
    Y_train = 10**Y_train - 1

    # since log scale is used, calculate MSE directly
    mse_per_param = ((Y_pred_val - Y_train) **
                     2).mean(axis=0)  # shape = (n_selected_params,)
    mse_per_param = mse_per_param/basemse
    avg_mse = mse_per_param.mean()

    print(f"  Val Score per parameter: {mse_per_param}")
    print(f"  Val average Score:       {avg_mse:.6f}")

    test_X, _ = build_feature_matrix_and_labels(
        np.array([]), None, args.dataset_dir, label_mean, label_std,
        is_train=False, rotate_aug=False,
        hsi_size=tuple(args.hsi_size), msi_size=tuple(args.msi_size),
        hsi_fft_ratio=args.hsi_fft_ratio, msi_fft_ratio=args.msi_fft_ratio, hsi_mean_fft_ratio=args.hsi_mean_fft_ratio,
        msi_mean_fft_ratio=args.msi_mean_fft_ratio, msi_max_fft_ratio=args.msi_max_fft_ratio,
        hsi_max_fft_ratio=args.hsi_max_fft_ratio, add_extra_features=args.add_extra_features,
        use_fe_features=args.use_fe_features, use_zn_features=args.use_zn_features, use_s_features=args.use_s_features,
        use_mn_advanced_features=args.use_mn_advanced_features, use_zn_advanced_features=args.use_zn_advanced_features,
        use_spatial_features=args.use_spatial_features,
        full_model_type=args.full_model_type, target_parameters=args.target_parameters)
    if args.top_k != 0:
        shap_feature_importance = pd.read_csv(
            f"result/shap_feature_importance_{args.full_model_type}.csv")
        top_common_idx = shap_feature_importance['feature'].apply(
            lambda x: int(x[5:])).iloc[:args.top_k].to_list()

        test_X = test_X[:, top_common_idx]

    Y_test = model.predict(test_X)  # shape = (N_test, n_selected_params)
    Y_test = Y_test * label_std + label_mean
    Y_test = 10**Y_test - 1

    submission = pd.DataFrame(Y_test, columns=args.target_parameters)
    submission.to_csv(
        f"result/submission_{args.random_state}_{args.full_model_type}.csv", index_label="sample_index")
    filename = f'model/{args.model}_regressor_{args.random_state}_{args.full_model_type}.joblib'
    joblib.dump(model, filename)


if __name__ == "__main__":
    # argparse → Path object
    args = parse_args()
    args.dataset_dir = Path(args.dataset_dir)

    full_train(args)
