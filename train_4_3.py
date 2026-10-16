"""文件内容与用途：论文4.4独立拟合及4.5验证选参重训，分目录保存两类权重和训练指标。"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from main import (
    METHODS,
    prepare_dataset,
    make_meta_models,
    make_base_models,
    read_json,
    log,
    train_nn_stacking,
    mlp_config,
    predict_nn_stacking,
    save_mlp,
    write_csv,
    load_mlp,
    train_mlp,
    predict_mlp,
    save_joblib,
    write_json,
    prediction_row,
    fit_convex,
    select_mlp_configs,
    unified_active,
)


def run_section43(name, run_dir, validation_selected=False):
    """分别运行训练拟合或验证选参重训；非神经固定模型在两种模式间复用。"""
    folder, a = prepare_dataset(name, run_dir)
    prep = joblib.load(folder / "preprocess.joblib")
    local_protocol = "local_training_plan" in read_json(Path(run_dir) / "protocol.json")
    unified = unified_active()
    model_scope = "section45_models" if unified and validation_selected else "section43"
    selected_configs = select_mlp_configs(name, folder) if local_protocol and (not unified or validation_selected) else {}
    joint = np.column_stack([a["x"], a["oof"]])
    estimators = make_meta_models()
    base = make_base_models()
    estimators.update({"E2E-XGB": base["XGBoost"], "E2E-RF": base["RandomForest"], "E2E-SVM": base["SVM"]})
    rows = []
    for method in METHODS:
        dest = folder / model_scope / method
        complete = dest / "complete.json"
        if complete.exists():
            rows.append(read_json(complete)["row"])
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if unified and validation_selected and method not in ["Proposed", "E2E-MLP", "NN-Stacking"]:
            source = folder / "section43" / method
            record = read_json(source / "complete.json")
            write_json(dest / "model_source.json", dict(relative_folder=f"section43/{method}", reused_fixed_model=True))
            write_json(complete, dict(record, reused_fixed_model=True, source_model_scope="section43"))
            rows.append(record["row"])
            continue
        x = a["x"] if method.startswith("E2E") else joint
        feature_set = "raw" if method.startswith("E2E") else "raw_plus_oof"
        if method == "NN-Stacking":
            feature_set = "raw_to_weights_plus_oof"
        log(f"{name} {model_scope} {method} START")
        start = time.perf_counter()
        config = dict(selected_configs[method]["config"]) if method in selected_configs else mlp_config(name)
        if method in selected_configs:
            write_json(dest / "selection_used.json", selected_configs[method])
        if method == "NN-Stacking":
            model, info, curve = train_nn_stacking(
                a["x"],
                a["oof"],
                a["y"],
                config,
                progress_name=f"{name} {method}",
                checkpoint_dir=dest,
            )
            p = predict_nn_stacking(model, a["x"], a["oof"], info["actual_batch_size"])
            save_mlp(dest, model, info)
            write_csv(dest / "loss_curve.csv", curve)
            loaded, loaded_info = load_mlp(dest)
            reloaded = predict_nn_stacking(loaded, a["x"], a["oof"], loaded_info["actual_batch_size"])
            np.testing.assert_allclose(p, reloaded, rtol=1e-6, atol=1e-6)
            del loaded
        elif method == "Convex-Stacking":
            original = a["oof"] * prep["scaler_y"].scale_[0] + prep["scaler_y"].mean_[0]
            convex = fit_convex(original, a["y_orig"])
            convex["weights"] = convex["weights"].tolist()
            write_json(dest / "model.json", convex)
            p = a["oof"] @ np.asarray(convex["weights"])
            np.testing.assert_allclose(p, a["oof"] @ np.asarray(read_json(dest / "model.json")["weights"]))
            feature_set = "oof_convex_weights"
            model = None
            write_csv(dest / "weights.csv", pd.DataFrame({"base_model": ["SVM", "RandomForest", "XGBoost"], "weight": convex["weights"]}))
        elif method in ["Proposed", "E2E-MLP"]:
            model, info, curve = train_mlp(
                x, a["y"], config, progress_name=f"{name} {method}", checkpoint_dir=dest
            )
            p = predict_mlp(model, x, info["actual_batch_size"])
            save_mlp(dest, model, info)
            write_csv(dest / "loss_curve.csv", curve)
            loaded, loaded_info = load_mlp(dest)
            np.testing.assert_allclose(
                p, predict_mlp(loaded, x, loaded_info["actual_batch_size"]), rtol=1e-6, atol=1e-6
            )
            del loaded
        else:
            shared_linear = folder / "section41" / ("LR" if method == "Meta-LR" else "Ridge") / "model.joblib"
            if unified and method in ["Meta-LR", "Meta-Ridge"] and shared_linear.exists():
                model = joblib.load(shared_linear)
            else:
                model = estimators[method].fit(x, a["y"])
            p = model.predict(x)
            save_joblib(dest / "model.joblib", model)
            np.testing.assert_allclose(
                p, joblib.load(dest / "model.joblib").predict(x), rtol=1e-10, atol=1e-10
            )
            params = {
                k: ("NaN" if isinstance(v, float) and np.isnan(v) else v)
                for k, v in model.get_params().items()
            }
            write_json(dest / "model.json", dict(params=params, input_dim=x.shape[1]))
        row = prediction_row(
            name,
            method,
            "train",
            feature_set,
            a["train_ids"],
            a["y_orig"],
            p,
            prep,
            dest / "predictions.csv",
            time.perf_counter() - start,
        )
        if method in selected_configs:
            row.update(selected_epochs=config["epochs"], candidate_id=selected_configs[method]["candidate_id"],
                       validation_MSE=selected_configs[method]["validation_MSE"])
        write_json(complete, dict(row=row, model_reloaded_train_predictions_verified=True))
        rows.append(row)
        log(f'{name} {model_scope} {method} DONE MSE={row["MSE"]:.6f}')
        del model
    write_csv(folder / model_scope / "metrics.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    from main import main

    main("43")
