"""文件内容与用途：论文4.5只加载验证选参后冻结的模型；固定基线复用拟合权重，评价相同测试行。"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from main import (
    METHODS,
    BASE_NAMES,
    load_base_groups,
    predict_base_feature_matrix,
    write_csv,
    load_mlp,
    predict_nn_stacking,
    predict_mlp,
    prediction_row,
    log,
    write_json,
    read_json,
    synchronized_time,
    predict_fold_pipeline,
    unified_active,
    unified_model_folder,
)


def run_section44(name, run_dir):
    """统一实验加载4.5专属模型，旧协议兼容4.3模型；测试前必须完成全模型冻结。"""
    folder = Path(run_dir) / "datasets" / name
    model_scope = "section45_models" if unified_active() else "section43"
    if not all((folder / model_scope / m / "complete.json").exists() for m in METHODS):
        raise ValueError(f"{name} 请先完成 {model_scope}")
    if unified_active() and not (Path(run_dir).parents[1] / "config/test_gate.json").exists():
        raise ValueError("全部4.5模型冻结前禁止执行正式测试")
    metrics_path = folder / "section44" / "metrics.csv"
    existing_rows = {}
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path, encoding="utf-8-sig")
        existing_rows = {str(row.method): row.to_dict() for _, row in existing.iterrows()}
    prep = joblib.load(folder / "preprocess.joblib")
    test = pd.read_csv(folder / "test.csv", encoding="utf-8-sig")
    ids = np.load(folder / "train_arrays.npz")["test_ids"]
    raw_features = test[prep["feature_names"]].to_numpy(dtype=float)
    x = prep["scaler_x"].transform(raw_features)
    groups = load_base_groups(folder)
    base_start = synchronized_time()
    pipelines = isinstance(groups[BASE_NAMES[0]][0], dict)
    base = predict_base_feature_matrix(groups, raw_features, prep) if pipelines else predict_base_feature_matrix(groups, x)
    base_seconds = synchronized_time() - base_start
    # 独立核对均值；原目标单位下使用浮点误差界，避免小标准差放大float32舍入而误报。
    mean_checks = []
    for j, n in enumerate(BASE_NAMES):
        if pipelines:
            values = np.column_stack([predict_fold_pipeline(m, raw_features) for m in groups[n]])
            independent = values.astype(np.float64).sum(axis=1) / 10
            actual = prep["scaler_y"].inverse_transform(base[:, j].astype(np.float64)[:, None]).ravel()
            epsilon = np.finfo(values.dtype).eps
            # 32个舍入误差项覆盖十项求和、取均值及目标尺度转换；保留逐点核验。
            bound = 32 * epsilon * (np.max(np.abs(values), axis=1) + abs(prep["scaler_y"].mean_[0]) + abs(prep["scaler_y"].scale_[0])) + 1e-12
            difference = np.abs(actual - independent)
            if np.any(difference > bound):
                raise ValueError(f"{n} 十模型均值超出原单位浮点误差界")
            mean_checks.append(dict(base_model=n, source_dtype=str(values.dtype),
                                   maximum_original_difference=float(difference.max()),
                                   maximum_roundoff_bound=float(bound.max()),
                                   rows=len(values), failures=int(np.count_nonzero(difference > bound))))
        else:
            expected = sum(m.predict(x) for m in groups[n]) / 10
            np.testing.assert_allclose(base[:, j], expected, rtol=1e-6, atol=1e-6)
    if mean_checks:
        write_csv(folder / "section44" / "base_mean_verification.csv", pd.DataFrame(mean_checks))
    del groups
    joint = np.column_stack([x, base])
    y = test[prep["target"]].to_numpy(dtype=float)
    rows = []
    write_csv(
        folder / "section44" / "base_mean_features.csv",
        pd.DataFrame(base, columns=BASE_NAMES).assign(row_id=ids),
    )
    for method in METHODS:
        prediction_path = folder / "section44" / method / "predictions.csv"
        if method in existing_rows and prediction_path.exists():
            rows.append(existing_rows[method])
            continue
        src = unified_model_folder(folder, method) if unified_active() else folder / "section43" / method
        features = x if method.startswith("E2E") else joint
        start = synchronized_time()
        if method == "NN-Stacking":
            model, info = load_mlp(src)
            p = predict_nn_stacking(model, x, base, info["actual_batch_size"])
        elif method in ["Proposed", "E2E-MLP"]:
            model, info = load_mlp(src)
            p = predict_mlp(model, features, info["actual_batch_size"])
        elif method == "Convex-Stacking":
            p = base @ np.asarray(read_json(src / "model.json")["weights"])
            model = None
        else:
            model = joblib.load(src / "model.joblib")
            p = model.predict(features)
        prediction_seconds = synchronized_time() - start
        feature_set = "raw" if method.startswith("E2E") else "raw_plus_fold_mean"
        if method == "NN-Stacking":
            feature_set = "raw_to_weights_plus_fold_mean"
        if method == "Convex-Stacking":
            feature_set = "fold_mean_convex_weights"
        row = prediction_row(name, method, "test", feature_set, ids, y, p, prep, prediction_path)
        if method in ["Proposed", "E2E-MLP", "NN-Stacking"] and (src / "selection_used.json").exists():
            selection = read_json(src / "selection_used.json")
            row.update(selected_epochs=selection["config"]["epochs"], candidate_id=selection["candidate_id"],
                       validation_MSE=selection["validation_MSE"])
        row.update(prediction_seconds=prediction_seconds, shared_base_prediction_seconds=0.0 if method.startswith("E2E") else base_seconds)
        rows.append(row)
        log(f'{name} 4.4 {method} R2={row["R2"]:.6f} RMSE={row["RMSE"]:.6f}')
        del model
    write_csv(folder / "section44" / "metrics.csv", pd.DataFrame(rows))
    write_json(
        folder / "section44" / "complete.json",
        dict(methods=METHODS, source="reloaded_saved_models", mean_verified=True),
    )


if __name__ == "__main__":
    from main import main

    main("44")
