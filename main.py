# -*- coding: utf-8 -*-
"""文件内容与用途：四节实验共用的数据处理、模型、统计计算和命令行入口。

用途：为论文4.2–4.5提供共享实现，保留旧41–44编号；数据和结果相对于本文件目录。
统一实验将4.4训练拟合与4.5验证选参模型分别保存，使用白酒、无目标筛选及严格OOF。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
import xgboost

# 命令行执行和实验模块导入共用同一个主模块，防止运行配置在两个模块副本中分离。
if __name__ == "__main__":
    sys.modules["main"] = sys.modules[__name__]

ROOT = Path(__file__).resolve().parent
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = None
TRAINING_PLAN = None
UNIFIED_PLAN = None
RUN_OUTPUT = None
DATASETS = [
    "AlignedData_aligned",
    "Concrete_Data",
    "parkinsons_updrs_regression",
    "wine_quality_white",
    "airfoil_self_noise",
    "superconductivity",
]
ABBR = dict(zip(DATASETS, ["AD", "CO", "PU", "WQ", "AS", "SC"]))
BASE_NAMES = ["SVM", "RandomForest", "XGBoost"]
METHODS = [
    "Proposed",
    "Meta-LR",
    "Meta-Ridge",
    "Meta-XGB",
    "Meta-RF",
    "Meta-SVM",
    "E2E-XGB",
    "E2E-RF",
    "E2E-SVM",
    "E2E-MLP",
]
METHODS += ["NN-Stacking", "Convex-Stacking"]
# 各节配置由results/configuration.json统一管理。
# 保留空映射仅用于原文件接口兼容，不参与正式选参。
EPOCH_CANDIDATES = {}
CACHE_ROOT = None
# 下列默认元数据与本轮4.4设置一致；4.5各折各方法独立验证选参。
FIXED_MLP_CONFIGS = {'AlignedData_aligned': {'hidden_dims': [1536, 1024, 512, 256], 'lr': 0.0003, 'batch_size': 512, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout': 0.0, 'weight_decay': 0.0, 'max_epochs': 3000, 'selection': 'minimum_evaluation_mode_training_MSE', 'validation_used': False, 'early_stopping': False, 'lr_scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5, 'patience': 100, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-06, 'eps': 1e-08}, 'epochs': 3000}, 'Concrete_Data': {'hidden_dims': [384, 192, 96], 'lr': 0.0008, 'batch_size': 32, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout': 0.0, 'weight_decay': 0.0, 'max_epochs': 3000, 'selection': 'minimum_evaluation_mode_training_MSE', 'validation_used': False, 'early_stopping': False, 'lr_scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5, 'patience': 100, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-06, 'eps': 1e-08}, 'epochs': 3000}, 'parkinsons_updrs_regression': {'hidden_dims': [1280, 1024, 512, 256], 'lr': 0.0004, 'batch_size': 256, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout': 0.0, 'weight_decay': 0.0, 'max_epochs': 3000, 'selection': 'minimum_evaluation_mode_training_MSE', 'validation_used': False, 'early_stopping': False, 'lr_scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5, 'patience': 100, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-06, 'eps': 1e-08}, 'epochs': 3000}, 'wine_quality_white': {'hidden_dims': [768, 512, 256, 128], 'lr': 0.0003, 'batch_size': 64, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout': 0.0, 'weight_decay': 0.0, 'max_epochs': 3000, 'selection': 'minimum_evaluation_mode_training_MSE', 'validation_used': False, 'early_stopping': False, 'lr_scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5, 'patience': 100, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-06, 'eps': 1e-08}, 'epochs': 3000}, 'airfoil_self_noise': {'hidden_dims': [512, 256, 128], 'lr': 0.0003, 'batch_size': 32, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout': 0.0, 'weight_decay': 0.0, 'max_epochs': 3000, 'selection': 'minimum_evaluation_mode_training_MSE', 'validation_used': False, 'early_stopping': False, 'lr_scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5, 'patience': 100, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-06, 'eps': 1e-08}, 'epochs': 3000}, 'superconductivity': {'hidden_dims': [2048, 1536, 768], 'lr': 0.0003, 'batch_size': 256, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout': 0.0, 'weight_decay': 0.0, 'max_epochs': 3000, 'selection': 'minimum_evaluation_mode_training_MSE', 'validation_used': False, 'early_stopping': False, 'lr_scheduler': {'name': 'ReduceLROnPlateau', 'mode': 'min', 'factor': 0.5, 'patience': 100, 'threshold': 0.0001, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 1e-06, 'eps': 1e-08}, 'epochs': 3000}}
# 4.1/4.2 共用隐藏层、学习率和批大小；训练轮数与最终损失曲线一致。
FIT_CONFIGS = {'AlignedData_aligned': {'lr': 0.0004, 'batch_size': 512, 'epochs': 2000}, 'Concrete_Data': {'lr': 0.0008, 'batch_size': 64, 'epochs': 2000}, 'parkinsons_updrs_regression': {'lr': 0.0004, 'batch_size': 256, 'epochs': 2000}, 'wine_quality_white': {'lr': 0.0003, 'batch_size': 32, 'epochs': 2000}, 'airfoil_self_noise': {'lr': 0.0003, 'batch_size': 32, 'epochs': 2000}, 'superconductivity': {'lr': 0.0003, 'batch_size': 256, 'epochs': 4000}}
# 纯Sigmoid比较有独立结构，避免调4.3的网络时意外改变4.1/4.2。
FIT_HIDDEN_DIMS = {'AlignedData_aligned': [1024, 768, 384, 192], 'Concrete_Data': [256, 128, 64], 'parkinsons_updrs_regression': [1280, 1024, 512, 256], 'wine_quality_white': [512, 496, 192], 'airfoil_self_noise': [512, 256, 128], 'superconductivity': [2048, 1024, 512]}


import importlib
import itertools
import math
import sys
from typing import Iterable
from scipy.stats import friedmanchisquare, rankdata

# 直接运行主程序时让四个实验共享同一份模块状态和模型类。
if __name__ == "__main__":
    sys.modules["main"] = sys.modules[__name__]


def feature_columns(frame):
    """选择实际输入与目标；AD使用匿名列，SC使用官方完整81个输入。"""
    if "y" in frame:
        expected = [f"x{i}" for i in range(1, len(frame.columns))] + ["y"]
        if list(frame.columns) != expected:
            raise ValueError("匿名数据列必须依次为 x1, x2, ..., y")
        return list(frame.columns[:-1]), "y"
    target = "critical_temp" if "critical_temp" in frame else frame.columns[-1]
    excluded = {target}
    return [column for column in frame if column not in excluded], target


def grouped_data(name):
    """按完全相同输入分组，目标取组内均值，保留首次出现顺序及原行映射。"""
    raw = pd.read_csv(ROOT / "data" / f"{name}.csv", encoding="utf-8-sig")
    columns, target = feature_columns(raw)
    if not np.isfinite(raw[columns + [target]].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} 含缺失或非有限值")
    groups = raw.groupby(columns, sort=False, dropna=False)
    ids = groups.ngroup().to_numpy()
    frame = raw.loc[~raw.duplicated(columns), columns + [target]].reset_index(drop=True)
    summary = raw.assign(group_id=ids).groupby("group_id", sort=True)[target].agg(["mean", "size", "min", "max"])
    frame[target] = summary["mean"].to_numpy()
    audit = dict(raw_rows=len(raw), grouped_rows=len(frame), features=len(columns),
                 duplicate_input_rows=len(raw)-len(frame), conflicting_groups=int((summary["min"] != summary["max"]).sum()),
                 rule="exact_input_group_mean_target_first_appearance", data_sha256=sha256(ROOT / "data" / f"{name}.csv"))
    return frame, raw, ids, audit


def run_path(output, fold):
    """优先复用指定的历史模型目录，否则使用结果目录内的缓存。"""
    return Path(str(CACHE_ROOT) + str(fold)) if CACHE_ROOT else Path(output) / ".cache" / f"fold{fold}"


def synchronized_time():
    """等显卡任务结束后读计时器，避免异步计算少计耗时。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def fit_convex(predictions, target):
    """枚举三权重单纯形的所有面，求非负且和为1的全局最小二乘解。"""
    best = None
    # 只有三个基模型，枚举七个非空面可同时覆盖边界最优解，无迭代收敛误差。
    for size in range(1, 4):
        for active in itertools.combinations(range(3), size):
            p = predictions[:, active]
            gram, rhs = p.T @ p, p.T @ target
            system = np.block([[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            w = np.linalg.lstsq(system, np.r_[rhs, 1.0], rcond=None)[0][:size]
            if np.min(w) < -1e-8 or not np.isclose(w.sum(), 1, atol=1e-7):
                continue
            w = np.maximum(w, 0); w /= w.sum()
            weights = np.zeros(3); weights[list(active)] = w
            mse = float(np.mean((predictions @ weights - target)**2))
            if best is None or mse < best[0]:
                best = mse, weights
    if best is None:
        raise RuntimeError("凸组合求解失败")
    return dict(weights=best[1], mse=best[0], solver="enumerate_all_simplex_faces", intercept=0.0)


def split_indices(n, outer_fold):
    """固定随机种子生成五组互斥测试集，每组使用约80%训练、20%测试。"""
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(n)))
    return folds[int(outer_fold)]


def init_run(run_dir, outer_fold, datasets):
    """记录数据、全部代码、参数与环境指纹；条件变化时拒绝复用训练缓存。"""
    run_dir = Path(run_dir)
    protocol = dict(
        seed=SEED,
        version=4,
        evaluation_status="fivefold_fixed_configuration_evaluation",
        deduplication="exact_input_group_mean_target_before_outer_split",
        epoch_candidates=EPOCH_CANDIDATES,
        outer_split="KFold5_outer_80_20",
        outer_fold=outer_fold,
        inner_folds=10,
        preprocessing_scope="each_base_OOF_training_fold_and_meta_training_scope",
        checkpoint_selection="minimum_training_eval_MSE",
        optimizer="AdamW",
        mlp=FIXED_MLP_CONFIGS,
        pure_mlp=FIT_CONFIGS,
        pure_hidden_dims=FIT_HIDDEN_DIMS,
        code={p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))},
        data={name: sha256(ROOT / "data" / f"{name}.csv") for name in DATASETS},
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            torch=torch.__version__,
            sklearn=sklearn.__version__,
            xgboost=xgboost.__version__,
        ),
    )
    if TRAINING_PLAN is not None:
        protocol.update(
            version=5,
            evaluation_status="fivefold_single_holdout_selection_within_each_training_pool",
            preprocessing_scope="each_base_OOF_training_fold_and_meta_training_scope",
            checkpoint_selection="minimum_validation_MSE_then_fixed_epoch_endpoint_refit",
            local_training_plan=TRAINING_PLAN,
            device=str(DEVICE),
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    if UNIFIED_PLAN is not None:
        protocol.update(version=7, target_screening="none", ranking_metric="original_unit_MSE",
                        evaluation_status="unified_fit_and_validation_selected_followup_experiment",
                        separate_neural_model_scopes=["section43", "section45_models"],
                        frozen_unified_plan=UNIFIED_PLAN)
    path = run_dir / "protocol.json"
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError("数据、代码、参数、数据集选择或环境已变化，请使用新的 --output 目录")
        return run_dir
    write_json(path, protocol)
    return run_dir


def log(message):
    """输出带时间的即时进度，便于查看长时间实验日志。"""
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(message)
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def sha256(path):
    """分块计算文件哈希，用于识别数据、代码和模型是否发生变化。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """先写临时文件再原子替换，避免中断留下半个配置或完成标记。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    """按 UTF-8 读取实验元数据。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, frame):
    """以带 BOM 的 UTF-8 原子保存数据表，兼容 Windows 表格软件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def save_joblib(path, obj):
    """压缩并原子保存 sklearn 模型或预处理器。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    joblib.dump(obj, temporary, compress=3)
    os.replace(temporary, path)


def set_seed(seed=SEED):
    """固定 Python、NumPy 和 PyTorch 随机数，限制 CPU 线程并关闭不确定的卷积选择。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_base_models():
    """返回正式实验使用的三种第一层算法及其冻结参数。"""
    return {
        "SVM": SVR(C=100, gamma="auto"),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
        "XGBoost": XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=6, random_state=SEED, verbosity=0, n_jobs=1
        ),
    }


def make_meta_models():
    """生成五种二层基线，非线性模型与相应第一层算法使用相同固定参数。"""
    bases = make_base_models()
    return {
        "Meta-LR": LinearRegression(),
        "Meta-Ridge": Ridge(alpha=1.0, random_state=SEED),
        "Meta-XGB": bases["XGBoost"],
        "Meta-RF": bases["RandomForest"],
        "Meta-SVM": bases["SVM"],
    }


def clean_training(frame, columns, target):
    """保留全部有效目标，分别拟合当前训练范围的输入和目标标准化器。"""
    x = frame[columns].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    if len(y) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("训练样本不足十个或存在非有限数据")
    kept = np.arange(len(y))
    sx, sy = StandardScaler(), StandardScaler()
    xs, ys = sx.fit_transform(x), sy.fit_transform(y[:, None]).ravel()
    prep = dict(scaler_x=sx, scaler_y=sy, feature_names=columns, target=target,
                iqr_bounds=None, sigma_bounds=None)
    return kept, xs, ys, prep




def predict_base_feature_matrix(groups, x, prep=None):
    """对测试或新样本取每种基算法十个模型的均值；输出三列且不使用目标标签。"""
    result = []
    for name in BASE_NAMES:
        if len(groups[name]) != 10:
            raise ValueError(f"{name} 必须恰好有十个折模型")
        if isinstance(groups[name][0], dict):
            if prep is None:
                raise ValueError("独立OOF流水线需要元模型预处理器以统一预测尺度")
            original = np.column_stack([predict_fold_pipeline(m, x) for m in groups[name]]).mean(axis=1)
            result.append(prep["scaler_y"].transform(original[:, None]).ravel())
        else:
            result.append(np.column_stack([m.predict(x) for m in groups[name]]).mean(axis=1))
    return np.column_stack(result)


def prepare_dataset(name, run_dir):
    """统一使用逐OOF训练子折独立标准化的完整80%训练池。"""
    return prepare_local_dataset(name, run_dir)


class AdaptiveMLP(nn.Module):
    """论文4.4/4.5的回归网络：全连接、BatchNorm、LeakyReLU、Dropout。"""

    def __init__(self, input_dim, hidden_dims, dropout):
        """根据冻结参数创建隐藏层及标量回归输出层。"""
        super().__init__()
        layers = []
        for size in hidden_dims:
            layers += [nn.Linear(input_dim, size), nn.BatchNorm1d(size), nn.LeakyReLU(), nn.Dropout(dropout)]
            input_dim = size
        self.net = nn.Sequential(*layers, nn.Linear(input_dim, 1))

    def forward(self, x):
        """将输入特征映射为一维预测。"""
        return self.net(x).squeeze(-1)


def combine_nn_stacking_outputs(network_output, base_predictions):
    """按论文公式计算三个无约束权重乘基预测之和，再加样本修正项。"""
    if network_output.ndim != 2 or network_output.shape[1] != len(BASE_NAMES) + 1:
        raise ValueError("NN-Stacking网络必须输出三个权重和一个修正项")
    if base_predictions.ndim != 2 or base_predictions.shape[1] != len(BASE_NAMES):
        raise ValueError("NN-Stacking必须接收三列基模型预测")
    weights = network_output[:, : len(BASE_NAMES)]
    correction = network_output[:, len(BASE_NAMES)]
    return (weights * base_predictions).sum(dim=1) + correction


class UNNSWithCorrection(nn.Module):
    """NN-Stacking基线：只由原始特征生成样本权重与修正项，再组合三列基预测。"""

    def __init__(self, raw_input_dim, hidden_dims, dropout):
        """沿用当前数据集的隐藏层、BatchNorm、LeakyReLU和Dropout配置。"""
        super().__init__()
        layers = []
        input_dim = raw_input_dim
        for size in hidden_dims:
            layers += [nn.Linear(input_dim, size), nn.BatchNorm1d(size), nn.LeakyReLU(), nn.Dropout(dropout)]
            input_dim = size
        self.weight_network = nn.Sequential(*layers, nn.Linear(input_dim, len(BASE_NAMES) + 1))

    def forward(self, raw_features, base_predictions):
        """权重网络只读取原始特征；基预测只在最终线性组合处参与计算。"""
        return combine_nn_stacking_outputs(self.weight_network(raw_features), base_predictions)


class PureSigmoidMLP(nn.Module):
    """论文4.2/4.3的纯全连接 Sigmoid 网络，不包含 Dropout 和 BatchNorm。"""

    def __init__(self, input_dim, hidden_dims):
        """按给定隐藏层大小建立纯 Sigmoid 网络。"""
        super().__init__()
        layers = []
        for size in hidden_dims:
            layers += [nn.Linear(input_dim, size), nn.Sigmoid()]
            input_dim = size
        self.net = nn.Sequential(*layers, nn.Linear(input_dim, 1))

    def forward(self, x):
        """返回每个样本的标准化目标预测值。"""
        return self.net(x).squeeze(-1)


def predict_mlp(model, x, batch_size):
    """在评估模式下分批预测，禁用 Dropout 并复用 BatchNorm 统计量。"""
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            features = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            result.append(model(features).cpu().numpy())
    return np.concatenate(result)


def evaluate_loss(model, x, y, batch_size):
    """按最终实验口径分批以 float32 计算完整训练集 MSELoss，供权重选择与绘图。"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            total += nn.functional.mse_loss(model(xb), yb).item() * len(xb)
    return total / len(x)


def predict_nn_stacking(model, raw_features, base_predictions, batch_size):
    """分批计算NN-Stacking预测，保持权重网络与三列基预测的输入边界。"""
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(raw_features), batch_size):
            raw = torch.as_tensor(
                raw_features[start : start + batch_size], dtype=torch.float32, device=DEVICE
            )
            base = torch.as_tensor(
                base_predictions[start : start + batch_size], dtype=torch.float32, device=DEVICE
            )
            result.append(model(raw, base).cpu().numpy())
    return np.concatenate(result)


def evaluate_nn_stacking_loss(model, raw_features, base_predictions, y, batch_size):
    """在完整训练集上计算NN-Stacking的标准化MSE，供同一检查点规则选择。"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            raw = raw_features[start : start + batch_size]
            base = base_predictions[start : start + batch_size]
            target = y[start : start + batch_size]
            total += nn.functional.mse_loss(model(raw, base), target).item() * len(target)
    return total / len(y)


def train_nn_stacking(raw_features, base_predictions, y, config, progress_name="NN-Stacking", checkpoint_dir=None, validation=None):
    """使用原始输入产生样本权重及校正项，严格保持NN-Stacking既定预测公式。"""
    return train_local_neural(raw_features, y, config, progress_name, checkpoint_dir, validation, base_predictions)


def train_mlp(x, y, config, pure=False, progress_name="MLP", checkpoint_dir=None, validation=None):
    """按Sigmoid或LeakyReLU配置训练；选择准则由拟合、验证或固定终点模式决定。"""
    return train_local_neural(x, y, dict(config, pure_sigmoid=pure), progress_name, checkpoint_dir, validation)


def predict_fold_pipeline(bundle, raw_features):
    """用某个OOF子折自己的输入及目标标准化器预测，并返回原目标单位。"""
    prep = bundle["preprocess"]
    x = prep["scaler_x"].transform(np.asarray(raw_features, dtype=float))
    prediction = bundle["model"].predict(x)
    return prep["scaler_y"].inverse_transform(np.asarray(prediction).reshape(-1, 1)).ravel()


def train_fold_pipelines(frame, row_ids, destination):
    """逐OOF子折独立拟合清洗和标准化，保存完整模型流水线及行号来源。"""
    columns, target = feature_columns(frame)
    row_ids = np.asarray(row_ids, dtype=int)
    splits = list(KFold(10, shuffle=True, random_state=SEED).split(frame))
    groups, predictions = {}, []
    fold_ids = np.full(len(frame), -1, dtype=int)
    persist_models = not (UNIFIED_PLAN is not None and "tuning" in Path(destination).parts)
    for base_name, estimator in make_base_models().items():
        models, oof = [], np.full(len(frame), np.nan)
        seen = np.zeros(len(frame), dtype=int)
        for fold, (tr, va) in enumerate(splits):
            path = Path(destination) / base_name / f"fold_{fold:02d}.joblib"
            if persist_models and path.exists():
                bundle = joblib.load(path)
                np.testing.assert_array_equal(bundle["holdout_row_ids"], row_ids[va])
                np.testing.assert_array_equal(bundle["preprocessing_fit_row_ids"], row_ids[tr])
            else:
                kept, x, y, prep = clean_training(frame.iloc[tr], columns, target)
                fitted = clone(estimator).fit(x, y)
                bundle = dict(
                    model=fitted, preprocess=prep,
                    preprocessing_fit_row_ids=row_ids[tr],
                    model_fit_row_ids=row_ids[tr[kept]],
                    holdout_row_ids=row_ids[va],
                    protocol="per_OOF_training_fold_preprocessing_original_unit_prediction",
                )
                if persist_models:
                    save_joblib(path, bundle)
                else:
                    write_json(path.with_suffix(".provenance.json"), dict(
                        model_weights_persisted=False, base_model=base_name, fold=fold,
                        preprocessing_fit_row_ids=row_ids[tr].tolist(),
                        model_fit_row_ids=row_ids[tr[kept]].tolist(), holdout_row_ids=row_ids[va].tolist(),
                        scaler_x_mean=prep["scaler_x"].mean_.tolist(),
                        scaler_x_scale=prep["scaler_x"].scale_.tolist(),
                        scaler_y_mean=prep["scaler_y"].mean_.tolist(),
                        scaler_y_scale=prep["scaler_y"].scale_.tolist()))
            if np.intersect1d(bundle["preprocessing_fit_row_ids"], row_ids[va]).size:
                raise ValueError("OOF预处理使用了对应留出行")
            oof[va] = predict_fold_pipeline(bundle, frame.iloc[va][columns].to_numpy(dtype=float))
            fold_ids[va] = fold
            seen[va] += 1
            models.append(bundle)
            log(f"BASE {base_name} fold {fold + 1}/10 pipeline ready; rows={len(tr)}/{len(va)}")
        if not (seen == 1).all() or not np.isfinite(oof).all():
            raise ValueError("完整OOF预测缺失、重复或非有限")
        groups[base_name] = models
        predictions.append(oof)
    return groups, np.column_stack(predictions), fold_ids


def prepare_local_dataset(name, run_dir):
    """准备完整80%最终训练池和测试划分，基模型预处理严格限制在各自OOF训练子折。"""
    folder = Path(run_dir) / "datasets" / name
    folder.mkdir(parents=True, exist_ok=True)
    ready = folder / "cache_complete.json"
    if ready.exists():
        return folder, dict(np.load(folder / "train_arrays.npz"))
    start = synchronized_time()
    frame, raw, group_ids, audit = grouped_data(name)
    columns, target = feature_columns(frame)
    outer_fold = read_json(Path(run_dir) / "protocol.json")["outer_fold"]
    train_ids, test_ids = split_indices(len(frame), outer_fold)
    train, test = frame.iloc[train_ids].reset_index(drop=True), frame.iloc[test_ids].reset_index(drop=True)
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError("最终训练池与测试行交叉")
    write_csv(folder / "train.csv", train)
    write_csv(folder / "test.csv", test)
    write_csv(folder / "group_mapping.csv", pd.DataFrame(dict(raw_row=np.arange(len(raw)), group_id=group_ids)))
    write_json(folder / "data_audit.json", audit)
    kept, x, y, prep = clean_training(train, columns, target)
    save_joblib(folder / "preprocess.joblib", prep)
    preprocessing_seconds = synchronized_time() - start
    base_start = synchronized_time()
    groups, original_oof, raw_fold_ids = train_fold_pipelines(train, train_ids, folder / "base_models")
    oof = prep["scaler_y"].transform(original_oof[kept].reshape(-1, 1)).reshape(len(kept), 3)
    arrays = dict(x=x, y=y, y_orig=train[target].to_numpy(dtype=float)[kept], oof=oof,
                  train_ids=train_ids[kept], outer_train_ids=train_ids, test_ids=test_ids,
                  fold_ids=raw_fold_ids[kept])
    np.savez_compressed(folder / "train_arrays.npz", **arrays)
    write_csv(folder / "oof_features_all_rows.csv", pd.DataFrame(original_oof, columns=BASE_NAMES).assign(row_id=train_ids, fold=raw_fold_ids))
    write_csv(folder / "oof_features.csv", pd.DataFrame(oof, columns=BASE_NAMES).assign(row_id=arrays["train_ids"], fold=arrays["fold_ids"]))
    write_csv(folder / "split_rows.csv", pd.DataFrame(dict(row_id=np.concatenate([train_ids, test_ids]), split=["train"] * len(train_ids) + ["test"] * len(test_ids))))
    write_json(ready, dict(raw_rows=len(raw), grouped_rows=len(frame), preprocessing_seconds=preprocessing_seconds,
                          base_fit_seconds=synchronized_time() - base_start, train_before_clean=len(train),
                          train_after_clean=len(kept), test_rows=len(test), features=len(columns), folds=10,
                          pipeline_preprocessing="per_base_OOF_training_fold", test_filtered=False))
    del groups
    log(f"{name} FINAL CACHE READY train={len(kept)} test={len(test)}")
    return folder, arrays


def prepare_validation_arrays(name, folder):
    """只从80%训练池分出一次验证集，重新拟合64%中的完整Stacking流程并保存隔离证据。"""
    destination = Path(folder) / "tuning"
    destination.mkdir(parents=True, exist_ok=True)
    ready = destination / "arrays_complete.json"
    if ready.exists():
        return destination, dict(np.load(destination / "validation_arrays.npz"))
    train = pd.read_csv(Path(folder) / "train.csv", encoding="utf-8-sig")
    columns, target = feature_columns(train)
    full = dict(np.load(Path(folder) / "train_arrays.npz"))
    outer_fold = read_json(Path(folder).parents[1] / "protocol.json")["outer_fold"]
    specification = TRAINING_PLAN["tuning_split"]
    seed = int(specification["random_state_base"]) + int(outer_fold)
    tr, va = train_test_split(np.arange(len(train)), test_size=specification["validation_fraction_of_training_pool"], random_state=seed)
    ids = full["outer_train_ids"]
    if np.intersect1d(ids[tr], ids[va]).size or np.intersect1d(ids[va], full["test_ids"]).size:
        raise ValueError("单验证集划分交叉")
    fit, validation = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
    kept, x, y, prep = clean_training(fit, columns, target)
    groups, raw_oof, fold_ids = train_fold_pipelines(fit, ids[tr], destination / "base_models")
    oof = prep["scaler_y"].transform(raw_oof[kept].reshape(-1, 1)).reshape(len(kept), 3)
    validation_raw = validation[columns].to_numpy(dtype=float)
    vx = prep["scaler_x"].transform(validation_raw)
    vbase = predict_base_feature_matrix(groups, validation_raw, prep)
    arrays = dict(x=x, y=y, oof=oof, vx=vx, voof=vbase,
                  y_orig=fit[target].to_numpy(dtype=float)[kept], vy_orig=validation[target].to_numpy(dtype=float),
                  target_mean=np.array(prep["scaler_y"].mean_[0]), target_scale=np.array(prep["scaler_y"].scale_[0]),
                  fit_ids=ids[tr[kept]], fit_before_clean_ids=ids[tr], validation_ids=ids[va], test_ids=full["test_ids"],
                  fold_ids=fold_ids[kept])
    np.savez_compressed(destination / "validation_arrays.npz", **arrays)
    save_joblib(destination / "preprocess.joblib", prep)
    write_csv(destination / "fit.csv", fit)
    write_csv(destination / "validation.csv", validation)
    write_csv(destination / "split_rows.csv", pd.DataFrame(dict(row_id=np.concatenate([ids[tr], ids[va]]), split=["fit"] * len(tr) + ["validation"] * len(va))))
    write_json(ready, dict(dataset=name, outer_fold=outer_fold, random_state=seed, fit_before_clean=len(tr),
                          fit_after_clean=len(kept), validation_rows=len(va), validation_filtered=False,
                          test_used=False, validation_sha256=sha256(destination / "validation.csv")))
    del groups
    log(f"{name} VALIDATION READY fit={len(kept)} validation={len(va)} seed={seed}")
    return destination, arrays


def predict_neural_tensors(model, x, base=None, batch_size=512):
    """在GPU上批量推理并一次返回结果，验证时关闭Dropout且不更新批归一化统计。"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            current = model(x[start:start + batch_size]) if base is None else model(x[start:start + batch_size], base[start:start + batch_size])
            predictions.append(current)
    return torch.cat(predictions).cpu().numpy()


def train_local_neural(x, y, config, progress_name, checkpoint_dir=None, validation=None, base_predictions=None):
    """统一执行训练拟合、验证早停或固定轮数重训，分别保存对应准则下的权重。"""
    set_seed()
    is_nn = base_predictions is not None
    pure = bool(config.get("pure_sigmoid", False))
    fitting = validation is None and config.get("selection") == "training_eval_mse"
    xs = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    ys = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    bases = torch.as_tensor(base_predictions, dtype=torch.float32, device=DEVICE) if is_nn else None
    model = (UNNSWithCorrection(x.shape[1], config["hidden_dims"], config["dropout"]) if is_nn
             else (PureSigmoidMLP(x.shape[1], config["hidden_dims"]) if pure
                   else AdaptiveMLP(x.shape[1], config["hidden_dims"], config["dropout"]))).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
                                 betas=tuple(config.get("betas", [0.9, 0.999])), eps=config.get("eps", 1e-8))
    batch = min(int(config["batch_size"]), len(y))
    starts = list(range(0, len(y), batch))
    if len(y) - starts[-1] == 1 and len(starts) > 1:
        starts.pop()
    ends = starts[1:] + [len(y)]
    if not pure and any(end - start < 2 for start, end in zip(starts, ends)):
        raise ValueError("批归一化训练批不能只含一个样本")
    scheduler = None
    if fitting and config.get("lr_scheduler"):
        specification = dict(config["lr_scheduler"])
        specification.pop("name", None)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **specification)
    evaluation_batch = max(batch, 512)
    vx = torch.as_tensor(validation["x"], dtype=torch.float32, device=DEVICE) if validation is not None else None
    vb = torch.as_tensor(validation["base"], dtype=torch.float32, device=DEVICE) if validation is not None and is_nn else None
    best_validation, patience_reference = float("inf"), float("inf")
    best_training = float("inf")
    best_epoch, bad_epochs, best = 0, 0, None
    curves = []
    start_time = synchronized_time()
    stop_reason = "maximum_epochs" if validation is not None or fitting else "fixed_epoch_endpoint"
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(y), device=DEVICE)
        total = torch.zeros((), device=DEVICE)
        for lower, upper in zip(starts, ends):
            indices = permutation[lower:upper]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xs[indices]) if not is_nn else model(xs[indices], bases[indices])
            loss = nn.functional.mse_loss(prediction, ys[indices])
            loss.backward()
            optimizer.step()
            total += loss.detach() * (upper - lower)
        train_batch_mse = float(total.item() / len(y))
        if not np.isfinite(train_batch_mse):
            raise FloatingPointError("神经训练出现非有限损失")
        validation_mse, eval_mse = None, None
        if validation is not None:
            scaled = predict_neural_tensors(model, vx, vb, evaluation_batch)
            original = np.asarray(scaled, dtype=np.float64) * validation["target_scale"] + validation["target_mean"]
            validation_mse = float(np.mean((original - validation["y_orig"]) ** 2))
            if not np.isfinite(validation_mse):
                raise FloatingPointError("候选验证预测出现非有限误差")
            if validation_mse < best_validation:
                best_validation, best_epoch = validation_mse, epoch
                best = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if validation_mse <= patience_reference * (1 - config["patience_relative_improvement"]):
                patience_reference, bad_epochs = validation_mse, 0
            else:
                bad_epochs += 1
        elif fitting or epoch % 10 == 0 or epoch == int(config["epochs"]):
            scaled = predict_neural_tensors(model, xs, bases, evaluation_batch)
            eval_mse = float(np.mean((np.asarray(scaled, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))
            if not np.isfinite(eval_mse):
                raise FloatingPointError("完整训练集的评估误差非有限")
            if fitting and eval_mse < best_training:
                best_training, best_epoch = eval_mse, epoch
                best = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if scheduler is not None:
                scheduler.step(eval_mse)
        curves.append(dict(epoch=epoch, train_batch_mse=train_batch_mse, eval_mse=eval_mse,
                           validation_mse=validation_mse, learning_rate=optimizer.param_groups[0]["lr"], patience_count=bad_epochs))
        should_stop = validation is not None and epoch >= config["min_epochs"] and bad_epochs >= config["patience_epochs"]
        if epoch == 1 or epoch % 50 == 0 or should_stop or epoch == int(config["epochs"]):
            elapsed = synchronized_time() - start_time
            log(f"{progress_name} epoch {epoch}/{config['epochs']} validation={validation_mse} best_epoch={best_epoch} elapsed={elapsed:.1f}s")
            if checkpoint_dir:
                write_csv(Path(checkpoint_dir) / "curve_in_progress.csv", pd.DataFrame(curves))
                write_json(Path(checkpoint_dir) / "progress.json", dict(epoch=epoch, max_epochs=config["epochs"],
                           best_epoch=best_epoch, best_validation_MSE=None if validation is None else best_validation,
                           elapsed_seconds=elapsed, updated_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        if should_stop:
            stop_reason = "validation_patience"
            break
    if validation is not None or fitting:
        model.load_state_dict(best)
    else:
        best_epoch = epoch
    final_scaled = predict_neural_tensors(model, xs, bases, evaluation_batch)
    final_train_mse = float(np.mean((np.asarray(final_scaled, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))
    info = dict(config=deepcopy(config), pure=pure, optimizer="AdamW", input_dim=x.shape[1],
                best_epoch=best_epoch, stopped_epoch=epoch, best_eval_loss=final_train_mse,
                best_validation_MSE=best_validation if validation is not None else None,
                actual_batch_size=batch, maximum_training_batch=max(b - a for a, b in zip(starts, ends)),
                evaluation_batch_size=evaluation_batch, train_samples=len(y),
                validation_samples=len(validation["y_orig"]) if validation is not None else 0,
                fit_seconds=synchronized_time() - start_time, selection=config["selection"], stop_reason=stop_reason,
                n_parameters=sum(parameter.numel() for parameter in model.parameters()),
                learning_rate_scheduler=scheduler is not None,
                checkpoint_criterion="training_MSE" if fitting else ("validation_MSE" if validation is not None else "fixed_endpoint"))
    if is_nn:
        info.update(raw_input_dim=x.shape[1], base_model_order=BASE_NAMES, correction=True,
                    model_type="nn_stacking_unns_correction", weights_constraint="none")
    return model, info, pd.DataFrame(curves)


def select_mlp_configs(name, folder):
    """在每折单验证集分别搜索三种神经方法，统一实验使用预先抽定的六候选。"""
    destination = Path(folder) / "tuning"
    ready = destination / "selection.json"
    if ready.exists():
        return read_json(ready)["selected"]
    destination, arrays = prepare_validation_arrays(name, folder)
    selected, all_rows = {}, []
    candidates = (UNIFIED_PLAN["datasets"][name]["paper45"]["candidates"]
                  if UNIFIED_PLAN is not None else TRAINING_PLAN["mlp_candidates"])
    for method in TRAINING_PLAN["neural_methods"]:
        rows = []
        for candidate in candidates:
            current = destination / method / candidate["candidate_id"]
            complete = current / "complete.json"
            if complete.exists():
                result = read_json(complete)
                rows.append(result)
                all_rows.append(result)
                continue
            current.mkdir(parents=True, exist_ok=True)
            config = dict(candidate, **TRAINING_PLAN["fixed_network"])
            config.update(epochs=config["max_epochs"], local_validation_protocol=True, selection="validation_MSE",
                          use_lr_scheduler=False)
            features = arrays["x"] if method in ["E2E-MLP", "NN-Stacking"] else np.column_stack([arrays["x"], arrays["oof"]])
            validation_features = arrays["vx"] if method in ["E2E-MLP", "NN-Stacking"] else np.column_stack([arrays["vx"], arrays["voof"]])
            validation = dict(x=validation_features, y_orig=arrays["vy_orig"], target_mean=float(arrays["target_mean"]),
                              target_scale=float(arrays["target_scale"]))
            log(f"{name} TUNING {method} {candidate['candidate_id']} START")
            try:
                if method == "NN-Stacking":
                    validation["base"] = arrays["voof"]
                    model, info, curve = train_nn_stacking(features, arrays["oof"], arrays["y"], config,
                                      f"{name} {method} {candidate['candidate_id']}", current, validation)
                    prediction = predict_nn_stacking(model, arrays["vx"], arrays["voof"], info["evaluation_batch_size"])
                else:
                    model, info, curve = train_mlp(features, arrays["y"], config,
                                      progress_name=f"{name} {method} {candidate['candidate_id']}", checkpoint_dir=current, validation=validation)
                    prediction = predict_mlp(model, validation_features, info["evaluation_batch_size"])
                original = np.asarray(prediction, dtype=np.float64) * float(arrays["target_scale"]) + float(arrays["target_mean"])
                checked_score = float(np.mean((original - arrays["vy_orig"]) ** 2))
                if not np.isclose(checked_score, info["best_validation_MSE"], rtol=1e-6, atol=1e-9):
                    raise RuntimeError("候选最佳模型的验证误差与记录不一致")
                if UNIFIED_PLAN is not None:
                    write_json(current / "model.json", dict(info, weights_persisted=False))
                else:
                    save_mlp(current, model, info)
                write_csv(current / "loss_curve.csv", curve)
                write_csv(current / "validation_predictions.csv", pd.DataFrame(dict(row_id=arrays["validation_ids"], y_true=arrays["vy_orig"], y_pred=original)))
                result = dict(method=method, candidate_id=candidate["candidate_id"], status="complete",
                              validation_MSE=checked_score, best_epoch=info["best_epoch"], stopped_epoch=info["stopped_epoch"],
                              n_parameters=info["n_parameters"], fit_seconds=info["fit_seconds"], config=config,
                              validation_predictions_sha256=sha256(current / "validation_predictions.csv"))
                del model
            except FloatingPointError as error:
                result = dict(method=method, candidate_id=candidate["candidate_id"], status="failed_nonfinite", error=str(error), config=config)
                log(f"{name} TUNING {method} {candidate['candidate_id']} FAILED {error}")
            write_json(complete, result)
            rows.append(result)
            all_rows.append(result)
            write_json(destination / "progress.json", dict(dataset=name, completed_candidates=len(all_rows),
                       expected_candidates=len(TRAINING_PLAN["neural_methods"]) * len(candidates),
                       current_method=method, last_candidate=candidate["candidate_id"]))
            log(f"{name} TUNING {method} {candidate['candidate_id']} DONE {result.get('validation_MSE')}")
        successful = [row for row in rows if row["status"] == "complete"]
        if not successful:
            raise RuntimeError(f"{name} {method} 所有候选失败")
        minimum = min(row["validation_MSE"] for row in successful)
        tied = [row for row in successful if row["validation_MSE"] <= minimum * (1 + TRAINING_PLAN["relative_tie_tolerance"])]
        winner = min(tied, key=lambda row: (row["n_parameters"], row["best_epoch"], row["candidate_id"]))
        config = dict(winner["config"], epochs=int(winner["best_epoch"]), selection="fixed_endpoint")
        selected[method] = dict(config=config, candidate_id=winner["candidate_id"], best_epoch=winner["best_epoch"],
                                validation_MSE=winner["validation_MSE"], validation_rows=len(arrays["validation_ids"]),
                                n_parameters=winner["n_parameters"])
        write_json(destination / method / "selected.json", selected[method])
    write_json(ready, dict(dataset=name, selected=selected, candidates=all_rows, test_used=False,
                          criterion="single_holdout_original_scale_validation_MSE", plan_revision=TRAINING_PLAN["revision"]))
    log(f"{name} THREE METHODS INDEPENDENTLY SELECTED")
    return selected






def save_mlp(folder, model, info):
    """使用文件句柄保存 PyTorch 权重，兼容含中文的 Windows 路径。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / "model.pt.pending"
    with pending.open("wb") as stream:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream)
    os.replace(pending, folder / "model.pt")
    write_json(folder / "model.json", info)


def load_mlp(folder):
    """从本目录的结构参数及权重重新创建 MLP，供独立预测及重载核验。"""
    folder = Path(folder)
    info = read_json(folder / "model.json")
    c = info["config"]
    if info.get("model_type") == "nn_stacking_unns_correction":
        model = UNNSWithCorrection(info["raw_input_dim"], c["hidden_dims"], c["dropout"]).to(DEVICE)
    else:
        model = (
            PureSigmoidMLP(info["input_dim"], c["hidden_dims"])
            if info["pure"]
            else AdaptiveMLP(info["input_dim"], c["hidden_dims"], c["dropout"])
        ).to(DEVICE)
    with (folder / "model.pt").open("rb") as stream:
        model.load_state_dict(torch.load(stream, map_location=DEVICE, weights_only=True))
    model.eval()
    return model, info


def metrics(y, p):
    """在原目标尺度计算MSE及辅助指标，沿用MAPE的近零保护规则。"""
    if not np.isfinite(p).all():
        raise ValueError("预测包含非有限值")
    safe = np.where(np.abs(y) < 1e-8, 1e-8, y)
    return dict(
        MSE=float(np.mean((y - p) ** 2)),
        R2=float(r2_score(y, p)),
        MAPE=float(np.mean(np.abs((y - p) / safe)) * 100),
        MAE=float(np.mean(np.abs(y - p))),
        RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
    )


def prediction_row(name, method, split, feature_set, ids, y, scaled_pred, prep, output, fit_seconds=0.0):
    """保存逐样本预测并生成可反算指标行，明确训练与测试的输入口径。"""
    # 先转为float64再逆标准化及写CSV，避免float32最短小数导致CSV反算与内存指标不同。
    scaled_pred = np.asarray(scaled_pred, dtype=np.float64)
    pred = prep["scaler_y"].inverse_transform(scaled_pred.reshape(-1, 1)).ravel()
    write_csv(output, pd.DataFrame(dict(row_id=ids, y_true=y, y_pred=pred, pred_scaled=scaled_pred)))
    y_scaled = prep["scaler_y"].transform(np.asarray(y).reshape(-1, 1)).ravel()
    return dict(
        dataset=name,
        method=method,
        split=split,
        feature_set=feature_set,
        n=len(y),
        **metrics(y, pred),
        MSELoss=float(np.mean((y_scaled - scaled_pred) ** 2)),
        fit_seconds=fit_seconds,
    )


def mlp_config(name, pure=False):
    """从当前冻结配置返回Sigmoid或LeakyReLU参数副本，防止跨小节误用。"""
    if UNIFIED_PLAN is None:
        raise ValueError("请先加载results/configuration.json")
    settings = deepcopy(UNIFIED_PLAN["datasets"][name]["paper42_43" if pure else "paper44"])
    settings.update(epochs=settings.get("epochs", settings.get("max_epochs")), selection="training_eval_mse",
                    unified_protocol=True, local_validation_protocol=True, use_lr_scheduler=False)
    return settings


def load_base_groups(folder):
    """加载第一层三类模型，每类恰好十个，与训练缓存的折顺序一致。"""
    return {
        name: [joblib.load(Path(folder) / "base_models" / name / f"fold_{k:02d}.joblib") for k in range(10)]
        for name in BASE_NAMES
    }


# 旧41--44编号对应论文4.2--4.5；报告主指标与训练、验证所用的MSE分别管理。
PRIMARY_METRICS = {"41": "RMSE", "42": "MSE", "43": "RMSE", "44": "RMSE"}
SECTION_METRICS = {
    section: [metric] + [other for other in ["MSE", "R2", "MAPE", "MAE", "RMSE", "MSELoss"] if other != metric]
    for section, metric in PRIMARY_METRICS.items()
}
RANK_METRICS = {section: [metric] for section, metric in PRIMARY_METRICS.items()}


_NULL_CACHE = {}


def method_key(frame: pd.DataFrame, section: str) -> pd.Series:
    """为4.2把模型与输入方式合成唯一名称，其余实验直接使用方法名。"""
    if section == "42":
        return frame["method"].astype(str) + "|" + frame["feature_set"].astype(str)
    return frame["method"].astype(str)


def summarize_values(frame: pd.DataFrame, section: str) -> pd.DataFrame:
    """按数据集与方法计算五折均值和样本标准差，并保留原始方法/输入名称。"""
    group = ["dataset", "method_key"]
    numeric = SECTION_METRICS[section]
    result = frame.groupby(group)[numeric].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(part for part in column if part) for column in result.columns]
    return result.rename(columns={"dataset_": "dataset", "method_key_": "method_key"})


def rank_values(frame: pd.DataFrame, section: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """先在每个数据集和80:20划分内排名，再汇总为每数据集及全部六数据集平均排名。"""
    ranked = frame.copy()
    rank_columns = []
    for metric in RANK_METRICS[section]:
        column = f"{metric}_rank"
        ranked[column] = ranked.groupby(["dataset", "outer_fold"])[metric].rank(
            method="average", ascending=metric != "R2"
        )
        rank_columns.append(column)
    ranked["mean_rank"] = ranked[rank_columns].mean(axis=1)
    by_dataset = ranked.groupby(["dataset", "method_key"], as_index=False)[
        rank_columns + ["mean_rank"]
    ].mean()
    overall = by_dataset.groupby("method_key", as_index=False)[rank_columns + ["mean_rank"]].mean()
    overall["overall_rank"] = overall["mean_rank"].rank(method="min")
    return by_dataset, overall.sort_values(["overall_rank", "mean_rank", "method_key"])


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """对一组原始P值执行逐步拒绝式Holm多重比较校正。"""
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for position, index in enumerate(order):
        running = max(running, (count - position) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def primary_metric(section):
    """返回论文对应小节的报告主指标；不会改变训练损失或验证选参准则。"""
    return PRIMARY_METRICS[str(section)]


def validate_metrics(frame: pd.DataFrame, section: str) -> None:
    """拒绝重复、缺折、缺方法、非有限指标，防止静默平均不完整实验。"""
    expected = (
        ["MLP", "LR", "Ridge"]
        if section == "41"
        else (
            [f"{m}|{f}" for m in ("MLP", "LR", "Ridge") for f in ("base_only", "base_plus_raw")]
            if section == "42"
            else METHODS
        )
    )
    if set(frame.dataset) != set(DATASETS) or set(frame.outer_fold) != set(range(5)):
        raise ValueError(f"4.{section[-1]} 数据集或折号不完整")
    if not frame.method_key.equals(method_key(frame, section)):
        raise ValueError("method_key与方法/输入名称不一致")
    if len(frame) != len(DATASETS) * 5 * len(expected):
        raise ValueError("数据集、折号和方法的组合不完整")
    if frame.duplicated(["dataset", "outer_fold", "method_key"]).any():
        raise ValueError("出现重复的数据集-折-方法记录")
    for _, group in frame.groupby(["dataset", "outer_fold"]):
        if set(group.method_key) != set(expected):
            raise ValueError("某个折的方法不完整")
    if not np.isfinite(frame[SECTION_METRICS[section]].to_numpy()).all():
        raise ValueError("存在缺失或非有限指标")


def rank_statistics(ranks: np.ndarray) -> tuple[float, float, float]:
    """由逐行名次计算含并列修正的Friedman Q、Kendall W及共同的秩和离差平方和。"""
    blocks, methods = ranks.shape
    centered = ranks - (methods + 1) / 2
    denominator = float(np.square(centered).sum())
    score = float(np.square(centered.sum(axis=0)).sum())
    if denominator == 0:
        return 0.0, float("nan"), 0.0
    return (methods - 1) * score / denominator, score / (blocks * denominator), score


def permutation_tail(ranks: np.ndarray, draws: int, seed: int) -> dict:
    """逐块置换方法标签；小空间穷举，完全同序且无并列时精确计数，其余使用固定种子的Monte Carlo。"""
    blocks, methods = ranks.shape
    _, concordance, observed = rank_statistics(ranks)
    space = math.factorial(methods) ** (blocks - 1)
    no_ties = all(len(np.unique(row)) == methods for row in ranks)
    if not np.isfinite(concordance):
        return {
            "p": 1.0,
            "calibration": "all_tied_no_information",
            "draws": 0,
            "seed": seed,
            "extreme_count": 0,
            "permutation_space": str(space),
        }
    if no_ties and np.isclose(concordance, 1.0, rtol=0, atol=1e-12):
        return {
            "p": 1.0 / space,
            "calibration": "exact_maximum_no_ties",
            "draws": 0,
            "seed": seed,
            "extreme_count": 1,
            "permutation_space": str(space),
        }
    # 共同重标记不会改变秩和平方，因此固定第一行，其余行独立置换。
    canonical = np.sort(ranks, axis=1) - (methods + 1) / 2
    key = (tuple(map(tuple, canonical)), draws, seed)
    exact = space <= 200000
    if key not in _NULL_CACHE:
        if exact:
            permutations = list(itertools.permutations(range(methods)))
            scores = []
            for choices in itertools.product(permutations, repeat=blocks - 1):
                sums = canonical[0].copy()
                for row, order in zip(canonical[1:], choices):
                    sums += row[list(order)]
                scores.append(float(np.square(sums).sum()))
            null = np.asarray(scores)
        else:
            rng = np.random.default_rng(seed)
            pieces = []
            for start in range(0, draws, 10000):
                size = min(10000, draws - start)
                sums = np.broadcast_to(canonical[0], (size, methods)).copy()
                for row in canonical[1:]:
                    order = np.argsort(rng.random((size, methods)), axis=1)
                    sums += row[order]
                pieces.append(np.square(sums).sum(axis=1))
            null = np.concatenate(pieces)
        _NULL_CACHE[key] = np.sort(null)
    null = _NULL_CACHE[key]
    extreme = int(len(null) - np.searchsorted(null, observed - 1e-10, side="left"))
    p_value = extreme / len(null) if exact else (extreme + 1) / (len(null) + 1)
    return {
        "p": p_value,
        "calibration": "exact_enumeration" if exact else "monte_carlo_plus_one",
        "draws": len(null),
        "seed": seed,
        "extreme_count": extreme,
        "permutation_space": str(space),
        "mc_resolution": None if exact else 1 / (len(null) + 1),
        "mc_standard_error_approx": None if exact else math.sqrt(p_value * (1 - p_value) / (len(null) + 1)),
    }


def statistical_test(frame: pd.DataFrame, section: str, metric: str, draws: int, seed: int) -> dict:
    """目标一：以六数据集五折均值形成的名次检验算法总体差异；不用小样本卡方近似作结论。"""
    means = frame.groupby(["dataset", "method_key"], as_index=False)[metric].mean()
    means["rank"] = means.groupby("dataset")[metric].rank(method="average", ascending=metric != "R2")
    table = means.pivot(index="dataset", columns="method_key", values="rank").reindex(DATASETS)
    methods = list(table.columns)
    statistic, _, _ = rank_statistics(table.to_numpy())
    permutation = permutation_tail(table.to_numpy(), draws, seed)
    scipy_q, approximate_p = friedmanchisquare(*[table[name].to_numpy() for name in methods])
    if not np.isclose(statistic, scipy_q):
        raise RuntimeError("Friedman Q与SciPy独立公式不一致")
    omnibus = {
        "section": section,
        "metric": metric,
        "primary_metric": metric == primary_metric(section),
        "unit": "six dataset-level means; each mean contains five 80:20 splits",
        "methods": methods,
        "n_datasets": len(DATASETS),
        "friedman_statistic": float(statistic),
        "friedman_p": permutation["p"],
        "chi_square_p_reference_only": float(approximate_p),
        "permutation": permutation,
        "null_hypothesis": "method labels exchangeable within each independent dataset block",
        "dataset_rank_matrix": table.to_dict(orient="index"),
    }
    return omnibus


def split_concordance(
    frame: pd.DataFrame, section: str, metric: str, draws: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """目标二：逐数据集量化五折方法排名一致性；置换P值仅为忽略折间依赖的条件探索性参照。"""
    concordance_rows, dispersion_rows = [], []
    for dataset in DATASETS:
        table = (
            frame.loc[frame.dataset == dataset]
            .pivot(index="outer_fold", columns="method_key", values=metric)
            .sort_index()
        )
        values = table.to_numpy()
        ranks = rankdata(-values if metric == "R2" else values, axis=1, method="average")
        q_value, w_value, _ = rank_statistics(ranks)
        permutation = permutation_tail(ranks, draws, seed)
        concordance_rows.append(
            {
                "section": section,
                "dataset": dataset,
                "abbr": ABBR[dataset],
                "metric": metric,
                "primary_metric": metric == primary_metric(section),
                "n_folds": 5,
                "n_methods": len(table.columns),
                "kendall_w": w_value,
                "rank_q": q_value,
                "p_conditional": permutation["p"],
                "calibration": permutation["calibration"],
                "draws": permutation["draws"],
                "seed": seed,
                "extreme_count": permutation["extreme_count"],
                "permutation_space": permutation["permutation_space"],
                "inference_status": "exploratory_only_overlapping_training_folds",
                "null_hypothesis": "independent random method-label orders conditional on each fold rank multiset",
            }
        )
        for column, method in enumerate(table.columns):
            scores, positions = values[:, column], ranks[:, column]
            dispersion_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "abbr": ABBR[dataset],
                    "metric": metric,
                    "primary_metric": metric == primary_metric(section),
                    "method_key": method,
                    "mean": float(scores.mean()),
                    "sample_sd": float(scores.std(ddof=1)),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                    "rank_mean": float(positions.mean()),
                    "rank_sample_sd": float(positions.std(ddof=1)),
                    "rank_min": float(positions.min()),
                    "rank_max": float(positions.max()),
                    **{f"fold_{fold}_value": float(scores[fold]) for fold in range(5)},
                    **{f"fold_{fold}_rank": float(positions[fold]) for fold in range(5)},
                }
            )
    return concordance_rows, dispersion_rows


def save_section_summary(frame, section, output, write_fold_records=True):
    """保存主指标汇总和排名；仅重汇总时保留逐折原始CSV的精确字节。"""
    summary = Path(output) / "summary"
    if write_fold_records:
        write_csv(summary / f"section{section}_all_folds.csv", frame)
    write_csv(summary / f"section{section}_mean_std.csv", summarize_values(frame, section))
    by_dataset, overall = rank_values(frame, section)
    write_csv(summary / f"section{section}_dataset_average_ranks.csv", by_dataset)
    write_csv(summary / f"section{section}_overall_average_ranks.csv", overall)




def collect_section(section, output, datasets):
    """收集本次运行各数据集五折结果；缺折时保留缓存，完整时才生成汇总。"""
    frames = []
    for fold in range(5):
        for name in datasets:
            path = run_path(output, fold) / "datasets" / name / f"section{section}" / "metrics.csv"
            if not path.exists():
                return False
            frame = pd.read_csv(path, encoding="utf-8-sig")
            frame["outer_fold"] = fold
            frame["method_key"] = method_key(frame, section)
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if datasets != DATASETS:
        return False
    validate_metrics(frame, section)
    write_csv(Path(output) / "summary" / f"section{section}_all_folds.csv", frame)
    if section == "41":
        for name in datasets:
            curves = []
            for fold in range(5):
                path = run_path(output, fold) / "datasets" / name / "section41" / "loss_curve.csv"
                curve = pd.read_csv(path, encoding="utf-8-sig")
                curve.insert(0, "outer_fold", fold)
                curves.append(curve)
            write_csv(Path(output) / "curves" / f"{name}.csv", pd.concat(curves, ignore_index=True))
    return True


def output_directory(value):
    """检查复现输出路径，保护仓库中已发布的数据、代码和最终结果不被覆盖。"""
    path = Path(value).resolve()
    if path == ROOT or path in ROOT.parents:
        raise ValueError("输出目录不能是仓库根目录或它的上级目录")
    if ROOT in path.parents:
        relative = path.relative_to(ROOT).parts
        if relative[0] != "results":
            raise ValueError("仓库内输出请使用 results/reproduced；其他输出请指定仓库外目录")
    return path




def markdown_table(frame, columns, labels=None):
    """将指定数据列写成可编辑Markdown表格，缺测值明确留空且不伪装成零。"""
    names = labels or columns
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.8g}")
            else:
                rendered.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)




def unified_active():
    """判断当前是否执行四节共享数据、独立拟合及验证选参的新版计划。"""
    return UNIFIED_PLAN is not None


def configure_unified(plan_path, output):
    """加载已经冻结的新计划，并适配现有严格OOF和单验证集公共接口。"""
    global UNIFIED_PLAN, TRAINING_PLAN, RUN_OUTPUT, CACHE_ROOT
    UNIFIED_PLAN = read_json(plan_path)
    if UNIFIED_PLAN["status"] != "frozen_for_execution":
        raise ValueError("必须使用已授权并冻结的执行计划")
    DATASETS[:] = list(UNIFIED_PLAN["datasets"])
    ABBR.update({name: spec["abbreviation"] for name, spec in UNIFIED_PLAN["datasets"].items()})
    specification = UNIFIED_PLAN["paper45_selection"]
    network = dict(max_epochs=specification["max_epochs"], min_epochs=specification["min_epochs_before_stopping"],
                   patience_epochs=specification["patience_epochs"],
                   patience_relative_improvement=specification["patience_relative_improvement"],
                   unified_protocol=True, betas=UNIFIED_PLAN["optimizer"]["betas"],
                   eps=UNIFIED_PLAN["optimizer"]["eps"])
    TRAINING_PLAN = dict(neural_methods=UNIFIED_PLAN["neural_methods"],
                         tuning_split=UNIFIED_PLAN["tuning_split"], fixed_network=network,
                         relative_tie_tolerance=specification["relative_tie_tolerance"],
                         revision=UNIFIED_PLAN["plan_version"], unified_sections=True,
                         target_screening="none")
    CACHE_ROOT = None
    RUN_OUTPUT = output_directory(output)
    RUN_OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, spec in UNIFIED_PLAN["datasets"].items():
        if sha256(ROOT / spec["expected_active_data_path"]) != spec["source_sha256"]:
            raise ValueError(f"数据指纹不一致：{name}")
        candidates = spec["paper45"]["candidates"]
        if len(candidates) != specification["candidate_count_per_dataset_method_fold"]:
            raise ValueError("候选数量与冻结预算不符")
        grid = specification["independent_grid"]
        for candidate in candidates:
            if candidate["dropout"] not in grid["dropout"] or candidate["weight_decay"] not in grid["weight_decay"]:
                raise ValueError("候选不属于独立参数网格")
    for section in ["41", "42", "43", "44"]:
        RANK_METRICS[section] = [primary_metric(section)]
    set_seed()
    return RUN_OUTPUT


def unified_model_folder(folder, method):
    """定位4.5专属神经权重或同条件固定基线，拒绝越出当前数据集缓存的引用。"""
    folder = Path(folder).resolve()
    source = folder / "section45_models" / method
    pointer = source / "model_source.json"
    if pointer.exists():
        if method in UNIFIED_PLAN["neural_methods"]:
            raise ValueError("4.5神经模型不得引用4.4拟合权重")
        source = (folder / read_json(pointer)["relative_folder"]).resolve()
        if not source.is_relative_to(folder):
            raise ValueError("固定基线引用越出当前数据集目录")
    return source


def train_unified_job(name, fold, output):
    """完成一个数据集一折的四节训练部分，始终不计算外层测试预测。"""
    global LOG_FILE
    from train_4_1 import run_section41
    from ablation_4_2 import run_section42
    from train_4_3 import run_section43

    run = init_run(run_path(output, fold), fold, DATASETS)
    LOG_FILE = run / f"{name}_train.log"
    folder = run / "datasets" / name
    marker = folder / "unified_training_complete.json"
    if marker.exists():
        return
    log(f"UNIFIED TRAIN START {name} fold={fold}")
    run_section41(name, run)
    run_section42(name, run)
    run_section43(name, run, validation_selected=False)
    run_section43(name, run, validation_selected=True)
    write_json(marker, dict(status="complete", dataset=name, outer_fold=fold,
                            paper42=True, paper43=True, paper44=True, paper45_final_models=True,
                            test_evaluated=False, completed_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"UNIFIED TRAIN COMPLETE {name} fold={fold}; test_evaluated=False")


def check_unified_frozen_sources(output):
    """核对本轮代码、数据、配置指纹，阻止中途变更后继续使用已有缓存。"""
    frozen = read_json(Path(output) / "config/frozen_files_sha256.json")
    for relative, expected in frozen.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"冻结文件发生变化：{relative}")


def freeze_unified_models(output):
    """所有三十个训练任务及540个候选完成后，冻结正式模型并允许测试。"""
    output = Path(output)
    check_unified_frozen_sources(output)
    candidates = 0
    model_hashes = {}
    for fold in range(5):
        for name in DATASETS:
            folder = run_path(output, fold) / "datasets" / name
            if not (folder / "unified_training_complete.json").exists():
                raise ValueError(f"尚未完成训练：{name} fold={fold}")
            records = list((folder / "tuning").glob("*/*/complete.json"))
            if len(records) != 18:
                raise ValueError("每个数据集每折必须完成十八条候选记录")
            candidates += len(records)
            arrays = dict(np.load(folder / "train_arrays.npz"))
            validation = dict(np.load(folder / "tuning/validation_arrays.npz"))
            np.testing.assert_array_equal(arrays["train_ids"], arrays["outer_train_ids"])
            np.testing.assert_array_equal(validation["fit_ids"], validation["fit_before_clean_ids"])
            for left, right in [(arrays["train_ids"], arrays["test_ids"]),
                                (validation["fit_ids"], validation["validation_ids"]),
                                (validation["validation_ids"], arrays["test_ids"])]:
                if np.intersect1d(left, right).size:
                    raise ValueError("训练、验证或测试行相互重叠")
            for method in METHODS:
                source = unified_model_folder(folder, method)
                if not (folder / "section45_models" / method / "complete.json").exists():
                    raise ValueError("4.5模型尚未完成")
                for p in source.glob("model.*"):
                    model_hashes[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256(p)
            for p in (folder / "base_models").glob("*/*.joblib"):
                model_hashes[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256(p)
            model_hashes[str((folder / "preprocess.joblib").relative_to(ROOT)).replace("\\", "/")] = sha256(folder / "preprocess.joblib")
    if candidates != 540:
        raise ValueError("全部候选记录数不是540")
    result = dict(status="all_final_models_frozen_before_test", created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  candidate_records=candidates, training_jobs=30, neural_final_configs=90,
                  model_files_sha256=model_hashes)
    write_json(output / "config/test_gate.json", result)
    return result


def check_unified_curve(folder, pure=False, validation_refit=False):
    """核对拟合模型确实保存最低训练误差，或验证重训确实使用选定轮数终点。"""
    info = read_json(Path(folder) / "model.json")
    curve_path = Path(folder).parent / "loss_curve.csv" if pure else Path(folder) / "loss_curve.csv"
    curve = pd.read_csv(curve_path, encoding="utf-8-sig", float_precision="round_trip")
    if validation_refit:
        if info["best_epoch"] != info["config"]["epochs"] or info["stopped_epoch"] != info["config"]["epochs"]:
            raise ValueError("验证选参后的重训未使用固定轮数终点")
    else:
        if info["stopped_epoch"] != info["config"]["epochs"]:
            raise ValueError("训练拟合实验未完成固定预算")
        selected = curve.loc[curve.epoch == info["best_epoch"], "eval_mse"].iloc[0]
        if not np.isclose(selected, curve.eval_mse.min(), rtol=1e-10, atol=1e-12):
            raise ValueError("拟合实验保存的不是训练MSE最低轮次")
        if not np.isclose(info["best_eval_loss"], selected, rtol=1e-6, atol=1e-10):
            raise ValueError("拟合模型训练MSE与曲线不一致")
    if pure != info["pure"]:
        raise ValueError("Sigmoid与LeakyReLU模型类型不符")
    return info


def verify_unified_results(output):
    """独立反算全部最终预测及候选验证MSE，核对行号、选参、最低拟合和固定终点。"""
    output = Path(output)
    check_unified_frozen_sources(output)
    gate = read_json(output / "config/test_gate.json")
    for relative, expected in gate["model_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"测试后正式模型发生变化：{relative}")
    checked, candidate_count, failed, model_curves = 0, 0, 0, 0
    maximum = {key: 0.0 for key in ["MSE", "R2", "MAPE", "MAE", "RMSE", "MSELoss"]}
    params = []
    for fold in range(5):
        for name in DATASETS:
            folder = run_path(output, fold) / "datasets" / name
            a = dict(np.load(folder / "train_arrays.npz"))
            v = dict(np.load(folder / "tuning/validation_arrays.npz"))
            prep = joblib.load(folder / "preprocess.joblib")
            train_ids, test_ids = split_indices(len(a["outer_train_ids"]) + len(a["test_ids"]), fold)
            np.testing.assert_array_equal(a["train_ids"], train_ids)
            np.testing.assert_array_equal(a["test_ids"], test_ids)
            np.testing.assert_array_equal(np.sort(np.r_[v["fit_ids"], v["validation_ids"]]), np.sort(train_ids))
            np.testing.assert_array_equal(v["fit_ids"], v["fit_before_clean_ids"])
            if prep["iqr_bounds"] is not None or prep["sigma_bounds"] is not None:
                raise ValueError("统一实验仍含目标值筛选")
            seen = set()
            for scope, methods in [("section41", ["MLP", "LR", "Ridge"]),
                                   ("section42_base_only", ["MLP", "LR", "Ridge"]),
                                   ("section43", METHODS), ("section45_models", UNIFIED_PLAN["neural_methods"]),
                                   ("section44", METHODS)]:
                metric_table = pd.read_csv(folder / scope / "metrics.csv", encoding="utf-8-sig", float_precision="round_trip")
                for method in methods:
                    p = folder / scope / method / "predictions.csv"
                    if p in seen:
                        continue
                    seen.add(p)
                    predictions = pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")
                    ids = a["test_ids"] if scope == "section44" else a["train_ids"]
                    np.testing.assert_array_equal(predictions.row_id, ids)
                    expected_y = (pd.read_csv(folder / "test.csv", encoding="utf-8-sig", float_precision="round_trip")[prep["target"]].to_numpy()
                                  if scope == "section44" else a["y_orig"])
                    np.testing.assert_allclose(predictions.y_true, expected_y, rtol=1e-12, atol=1e-12)
                    row = metric_table.loc[metric_table.method == method].iloc[0]
                    recomputed = metrics(predictions.y_true.to_numpy(), predictions.y_pred.to_numpy())
                    recomputed["MSELoss"] = float(np.mean(((predictions.y_true.to_numpy() - prep["scaler_y"].mean_[0]) /
                                                          prep["scaler_y"].scale_[0] - predictions.pred_scaled.to_numpy()) ** 2))
                    for key, value in recomputed.items():
                        maximum[key] = max(maximum[key], abs(value - float(row[key])))
                        if not np.isclose(value, float(row[key]), rtol=1e-10, atol=1e-10):
                            raise ValueError(f"指标反算失败：{name}/{fold}/{scope}/{method}/{key}")
                        checked += 1
            selection = read_json(folder / "tuning/selection.json")["selected"]
            for method in UNIFIED_PLAN["neural_methods"]:
                records = []
                for candidate in UNIFIED_PLAN["datasets"][name]["paper45"]["candidates"]:
                    current = folder / "tuning" / method / candidate["candidate_id"]
                    record = read_json(current / "complete.json")
                    candidate_count += 1
                    if record["status"] != "complete":
                        failed += 1
                        continue
                    prediction = pd.read_csv(current / "validation_predictions.csv", encoding="utf-8-sig", float_precision="round_trip")
                    np.testing.assert_array_equal(prediction.row_id, v["validation_ids"])
                    np.testing.assert_allclose(prediction.y_true, v["vy_orig"], rtol=1e-12, atol=1e-12)
                    mse = float(np.mean((prediction.y_true.to_numpy() - prediction.y_pred.to_numpy()) ** 2))
                    if not np.isclose(mse, record["validation_MSE"], rtol=1e-12, atol=1e-12):
                        raise ValueError("候选验证MSE反算失败")
                    if sha256(current / "validation_predictions.csv") != record["validation_predictions_sha256"]:
                        raise ValueError("候选验证预测指纹变化")
                    if (current / "model.pt").exists():
                        raise ValueError("候选不应永久保存重复权重")
                    records.append(record)
                minimum = min(row["validation_MSE"] for row in records)
                tied = [row for row in records if row["validation_MSE"] <= minimum * (1 + TRAINING_PLAN["relative_tie_tolerance"])]
                winner = min(tied, key=lambda row: (row["n_parameters"], row["best_epoch"], row["candidate_id"]))
                chosen = selection[method]
                if chosen["candidate_id"] != winner["candidate_id"] or chosen["best_epoch"] != winner["best_epoch"]:
                    raise ValueError("独立复算的候选选择不一致")
                for scope, refit in [("section43", False), ("section45_models", True)]:
                    info = check_unified_curve(folder / scope / method, validation_refit=refit)
                    model_curves += 1
                    cfg = info["config"]
                    params.append(dict(dataset=name, outer_fold=fold, method=method,
                                       paper_section="4.5" if refit else "4.4", hidden_dims=",".join(map(str, cfg["hidden_dims"])),
                                       lr=cfg["lr"], dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
                                       batch_size=cfg["batch_size"], best_epoch=info["best_epoch"], stopped_epoch=info["stopped_epoch"],
                                       candidate_id=chosen["candidate_id"] if refit else "fixed_fit",
                                       validation_MSE=chosen["validation_MSE"] if refit else None,
                                       fit_seconds=info["fit_seconds"]))
            for scope in ["section41", "section42_base_only"]:
                check_unified_curve(folder / scope / "MLP", pure=True)
                model_curves += 1
    if candidate_count != 540 or len(params) != 180 or model_curves != 240:
        raise ValueError("候选或最终神经模型数量不完整")
    write_csv(output / "summary/all_neural_parameters.csv", pd.DataFrame(params))
    write_csv(output / "summary/paper45_selected_parameters.csv", pd.DataFrame(params).query("paper_section == '4.5'"))
    result = dict(status="passed", final_metric_checks=checked, candidate_records=candidate_count,
                  failed_candidates=failed, candidate_validation_MSE_independently_checked=candidate_count - failed,
                  neural_model_curves_checked=model_curves, max_independent_metric_errors=maximum,
                  separate_fit_and_validation_models_verified=True, all_valid_targets_retained=True,
                  final_model_hashes_unchanged=True, github_modified=False)
    write_json(output / "verification.json", result)
    return result


def plot_unified_fit(output):
    """绘制六个数据集的三种神经方法五折训练拟合曲线，保留各自原单位纵轴。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for ax, name in zip(axes.flat, DATASETS):
        for method in UNIFIED_PLAN["neural_methods"]:
            curves = []
            for fold in range(5):
                folder = run_path(output, fold) / "datasets" / name
                curve = pd.read_csv(folder / "section43" / method / "loss_curve.csv", encoding="utf-8-sig")
                scale = joblib.load(folder / "preprocess.joblib")["scaler_y"].scale_[0]
                curves.append(np.sqrt(curve.eval_mse.to_numpy() * scale ** 2))
            values = np.stack(curves)
            epochs = np.arange(1, values.shape[1] + 1)
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(epochs, mean, label=method, linewidth=1)
            ax.fill_between(epochs, np.maximum(0, mean - sd), mean + sd, alpha=0.12)
        ax.set(title=ABBR[name], xlabel="Epoch", ylabel="Training RMSE (original units)")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    destination = Path(output) / "figures"
    destination.mkdir(exist_ok=True)
    figure.savefig(destination / "paper44_training_curves.png", dpi=180)
    plt.close(figure)


def summarize_unified_results(output):
    """四节完成并核验后，按照论文各节主指标输出排名、参数和中文实验汇报。"""
    output = Path(output)
    for section in ["41", "42", "43", "44"]:
        if not collect_section(section, output, DATASETS):
            raise ValueError(f"第{section}节尚未完成")
        frame = pd.read_csv(output / "summary" / f"section{section}_all_folds.csv", encoding="utf-8-sig")
        save_section_summary(frame, section, output)
    verification = verify_unified_results(output)
    lines = ["# 四节统一实验结果", "", "白酒替代混合酒；全部数据集保留有效目标，精确同输入分组取目标均值。", "",
             "4.2、4.3为纯Sigmoid拟合与输入消融；4.4为独立充分训练拟合；4.5为单验证集选参后完整训练池重训。", "",
             "神经方法结构、学习率和批大小按数据集固定。4.5从独立20组合网格按种子20261014固定抽6组合，三个方法分别选参。", "",
             "4.2、4.4、4.5按RMSE排名，4.3按MSE排名；逐折排名后平均五折，再等权平均六数据集。训练及验证选择仍使用MSE。", ""]
    for section, title in [("43", "4.4训练拟合"), ("44", "4.5测试预测")]:
        overall = pd.read_csv(output / "summary" / f"section{section}_overall_average_ranks.csv", encoding="utf-8-sig")
        metric = primary_metric(section)
        lines.extend([f"## {title}总体{metric}排名", "", markdown_table(overall,
            ["method_key", "mean_rank", "overall_rank"], ["方法", f"{metric}平均名次", "排列位置"]), ""])
        summaries = pd.read_csv(output / "summary" / f"section{section}_mean_std.csv", encoding="utf-8-sig")
        for name in DATASETS:
            selected = summaries.loc[summaries.dataset == name].sort_values(f"{metric}_mean")
            lines.extend([f"### {ABBR[name]}", "", markdown_table(selected,
                ["method_key", f"{metric}_mean", f"{metric}_std", "R2_mean", "MAE_mean", "MSE_mean", "MAPE_mean"],
                ["方法", f"{metric}均值", f"{metric}样本SD", "R2", "MAE", "MSE", "MAPE百分比"]), ""])
    lines.extend(["## 核验和完整文件", "", json.dumps(verification, ensure_ascii=False), "",
                  "summary/section41*与section42*为论文4.2与4.3；section43*与section44*为论文4.4与4.5。", "",
                  "summary/all_neural_parameters.csv保存4.4与4.5共180组神经训练参数和最佳轮数。", "",
                  "summary/paper45_selected_parameters.csv保存4.5独立选定的90组配置。", "",
                  "figures/paper44_training_curves.png保存三种神经方法完整3000轮五折曲线。", "",
                  "4.4模型位于.cache/fold*/datasets/*/section43；4.5神经模型位于section45_models，固定基线引用4.4同条件模型。", "",
                  "候选权重不重复落盘；候选参数、验证预测和曲线保留，最终权重和全部正式预测保留。", ""])
    report = output / "实验结果汇报.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    plot_unified_fit(output)
    result = dict(status="complete", completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  report=str(report), report_sha256=sha256(report), verification=verification,
                  paper_sections=["4.2", "4.3", "4.4", "4.5"], github_modified=False,
                  original_outputs_overwritten=False)
    write_json(output / "completion.json", result)
    log(f"UNIFIED ALL COMPLETE {report}")
    return result




def main(default_section="all"):
    """五个原入口共用命令行；默认在新目录运行四节，不覆盖仓库所附正式结果。"""
    parser = argparse.ArgumentParser(description="论文4.2/4.4/4.5按RMSE报告，4.3按MSE报告；训练及验证选择仍使用MSE")
    parser.add_argument("--section", choices=["all", "41", "42", "43", "44"], default=default_section)
    parser.add_argument("--dataset", choices=["all"] + DATASETS, default="all")
    parser.add_argument("--fold", choices=["all", "0", "1", "2", "3", "4"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    parser.add_argument("--unified-plan", type=Path, default=ROOT / "results/configuration.json")
    parser.add_argument("--stage", "--unified-stage", choices=["all", "train", "freeze", "test", "summarize"], default="all")
    args = parser.parse_args()
    output = configure_unified(args.unified_plan, args.output)
    if args.stage == "summarize" and not (output / ".cache").exists():
        publication_analysis(output)
        return
    if output.resolve() == (ROOT / "results").resolve():
        raise ValueError("results是已发布结果；训练请指定独立复现目录")
    prepare_release_output(output, args.unified_plan)
    check_unified_frozen_sources(output)
    names = DATASETS if args.dataset == "all" else [args.dataset]
    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]
    if args.stage in ["all", "train"]:
        for fold in folds:
            run = init_run(run_path(output, fold), fold, DATASETS)
            for name in names:
                if args.section in ["all", "44"]:
                    train_unified_job(name, fold, output)
                elif args.section == "41":
                    from train_4_1 import run_section41
                    run_section41(name, run)
                elif args.section == "42":
                    from ablation_4_2 import run_section42
                    run_section42(name, run)
                else:
                    from train_4_3 import run_section43
                    run_section43(name, run, validation_selected=False)
        if args.stage == "train" or args.section not in ["all", "44"] or len(names) != 6 or len(folds) != 5:
            return
    if args.stage in ["all", "freeze"]:
        freeze_unified_models(output)
    if args.stage in ["all", "test"]:
        from predict_4_4 import run_section44
        for fold in folds:
            for name in names:
                run_section44(name, run_path(output, fold))
    if args.stage in ["all", "summarize"]:
        summarize_unified_results(output)
        collect_publication_curves(output)
        publication_analysis(output)



def publication_analysis(output, permutations=999999, concordance_permutations=199999):
    """仅用已保存逐折记录按论文主指标重汇总、统计及绘图，保留原始记录且不训练模型。"""
    output = Path(output)
    omnibus, concordances, dispersions = [], [], []
    source_hashes = {}
    for section in ["41", "42", "43", "44"]:
        path = output / "summary" / f"section{section}_all_folds.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
        validate_metrics(frame, section)
        source_hashes[path.relative_to(output).as_posix()] = sha256(path)
        save_section_summary(frame, section, output, write_fold_records=False)
        metric = primary_metric(section)
        omnibus.append(statistical_test(frame, section, metric, permutations, 20261016))
        concordance, dispersion = split_concordance(frame, section, metric, concordance_permutations, 20261017)
        concordances.extend(concordance)
        dispersions.extend(dispersion)
    for row, adjusted in zip(omnibus, holm_adjust([row["friedman_p"] for row in omnibus])):
        row.update(p_holm=float(adjusted), significant_0_05=bool(adjusted < 0.05), multiplicity_family="four_section_primary_metric_tests")
    agreement = pd.DataFrame(concordances)
    agreement["p_conditional_holm"] = holm_adjust(agreement.p_conditional)
    agreement["multiplicity_family"] = "24_section_primary_metric_conditional_tests"
    write_json(output / "statistics/friedman.json", omnibus)
    write_csv(output / "statistics/kendall_concordance.csv", agreement)
    write_csv(output / "statistics/split_dispersion.csv", pd.DataFrame(dispersions))
    for path in sorted((output / "curves").rglob("*.csv")):
        source_hashes[str(path.relative_to(output)).replace("\\", "/")] = sha256(path)
    verification = dict(primary_metrics=PRIMARY_METRICS, training_and_validation_metric="MSE", input_sha256=source_hashes,
               friedman_tests=4, concordance_tests=24, primary_p_adjustment="Holm", seed=20261016,
               inference_unit="six datasets; five-fold means within each dataset",
                curve_aggregation="sqrt per trial first, then mean and sample SD across five trials",
                no_retraining=True, source_records_unchanged=True,
                concordance_status="descriptive; conditional P-values exploratory because training folds overlap")
    from train_4_1 import plot_fivefold_curves
    plot_fivefold_curves(output / "curves", output)
    plot_publication_fit(output)
    for relative, digest in source_hashes.items():
        if sha256(output / relative) != digest:
            raise RuntimeError(f"原始逐折记录或曲线发生变化：{relative}")
    write_json(output / "statistics/verification.json", verification)
    log("已按论文各节RMSE/MSE口径生成统计及RMSE曲线；未进行参数搜索或重新训练")
    return verification


def save_paper_figure_rows(figure, axes, destination, prefix):
    """将六子图按每行两图导出为矢量PDF，供论文自然分页使用。"""
    from matplotlib.transforms import Bbox
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for row in range(3):
        box = Bbox.union([ax.get_tightbbox(renderer) for ax in axes[row]])
        box = box.expanded(1.008, 1.025).transformed(figure.dpi_scale_trans.inverted())
        figure.savefig(Path(destination) / f"{prefix}_row{row+1}.pdf", bbox_inches=box, pad_inches=0.02)


def plot_publication_fit(output):
    """逐折将原量纲MSE开平方后，再绘制4.4的RMSE均值与样本标准差。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        figure, axes = plt.subplots(3, 2, figsize=(10.2, 8.7), constrained_layout=True)
        for ax, name in zip(axes.flat, DATASETS):
            table = pd.read_csv(output / "curves/paper44" / f"{name}.csv", encoding="utf-8-sig")
            for method, color in zip(TRAINING_PLAN["neural_methods"], ["#2166ac", "#d95f02", "#2b8c6b"]):
                current = table.loc[table.method == method]
                pivot = np.sqrt(current.pivot(index="epoch", columns="outer_fold", values="MSE"))
                assert pivot.shape == (3000, 5) and not pivot.isna().any().any()
                mean, sd = pivot.mean(axis=1), pivot.std(axis=1, ddof=1)
                ax.plot(pivot.index, mean, color=color, lw=1.1, label=method)
                ax.fill_between(pivot.index, mean-sd, mean+sd, where=(mean-sd)>0, color=color, alpha=0.12)
            ax.set_title(ABBR[name])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training RMSE")
            ax.set_yscale("log")
            ax.grid(alpha=0.18)
            ax.legend(fontsize=7, loc="upper right")
        for extension in ["pdf", "png"]:
            figure.savefig(output / "figures" / f"paper44_training_curves.{extension}", dpi=240)
        save_paper_figure_rows(figure, axes, output / "figures", "neural_rmse")
        plt.close(figure)


def prepare_release_output(output, plan_path):
    """为新复现目录生成可移植指纹；已发布结果不包含本机缓存或模型路径。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in [*sorted(ROOT.glob("*.py")), *[ROOT / "data" / f"{name}.csv" for name in DATASETS], Path(plan_path)]}
    path = output / "config/frozen_files_sha256.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("复现代码、数据或参数发生变化；请使用新的 --output 目录")
    else:
        write_json(path, manifest)
        write_json(output / "config/frozen_plan.json", read_json(plan_path))


def collect_publication_curves(output):
    """从完整训练缓存导出原目标单位曲线，逐折转换后再汇总，避免混用标准化尺度。"""
    output = Path(output)
    for name in DATASETS:
        pure, fitting = [], []
        for fold in range(5):
            folder = run_path(output, fold) / "datasets" / name
            scale2 = joblib.load(folder / "preprocess.joblib")["scaler_y"].scale_[0] ** 2
            curve = pd.read_csv(folder / "section41/loss_curve.csv", encoding="utf-8-sig")
            for key in ["eval_mse", "train_batch_mse", "LR_mse", "Ridge_mse", "best_mlp_mse"]:
                if key in curve:
                    curve[key] *= scale2
            pure.append(curve.assign(outer_fold=fold, dataset=name, target_units="original"))
            for method in TRAINING_PLAN["neural_methods"]:
                curve = pd.read_csv(folder / "section43" / method / "loss_curve.csv", encoding="utf-8-sig")
                fitting.append(pd.DataFrame(dict(epoch=curve.epoch, MSE=curve.eval_mse * scale2,
                                                method=method, outer_fold=fold, dataset=name)))
        write_csv(output / "curves" / f"{name}.csv", pd.concat(pure, ignore_index=True))
        write_csv(output / "curves/paper44" / f"{name}.csv", pd.concat(fitting, ignore_index=True))


if __name__ == "__main__":
    main()
