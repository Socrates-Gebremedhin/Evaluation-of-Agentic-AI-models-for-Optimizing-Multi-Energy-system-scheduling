"""
Import preprocessors and models from preprocessor/ and forecast_models/ (sibling packages).
"""
from pathlib import Path
import sys

_here = Path(__file__).resolve().parent
_parent = _here.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import itertools
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def set_seed(seed=42):
    """Set seeds for reproducibility (NumPy, Python random, PyTorch, CUDA)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _native_params(params):
    """Convert numpy/pandas scalars in a params dict to native Python types (for model constructors)."""
    if not params:
        return params
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (np.integer, np.int64, np.int32)):
            out[k] = tuple(int(x) for x in v) if isinstance(v, tuple) else [int(x) for x in v]
        else:
            out[k] = v
    return out


def evaluate_model(pipeline, df, metric_fn, model_params=None, eval_set="val"):
    """
    Evaluate a model pipeline using train/val/test split and a given metric.

    eval_set: "val" (default) or "test". Use "test" for final comparison of models.

    Supports:
      - "forecast": models that use df directly (e.g. SARIMA, Persistence)
      - "tabular": models that use X/y dataframes (e.g. XGBoost)
      - "tensor": models that take (X, y) arrays (e.g. LSTM)
    """
    model_params = _native_params(model_params or {})
    model_type = pipeline.get("model_type", "forecast")

    train_df, val_df, test_df = pipeline["splitter"](df)
    eval_df = test_df if eval_set == "test" else val_df

    X_train, y_train, X_eval, y_eval = None, None, None, None
    for step in pipeline.get("preprocessing", []):
        step.fit(train_df)
        res_train = step.transform(train_df)
        res_eval = step.transform(eval_df)
        if isinstance(res_train, tuple):
            X_train, y_train = res_train
            X_eval, y_eval = res_eval
            break
        train_df = res_train
        eval_df = res_eval

    model = pipeline["model"](**model_params)
    warnings.simplefilter("ignore", ConvergenceWarning)

    start_time = time.perf_counter()
    try:
        if model_type == "forecast":
            model.fit(train_df)
            horizon = len(eval_df)
            exog_col = getattr(model, "exog_col", None) or pipeline.get("exog_col")
            if exog_col and exog_col in eval_df.columns:
                exog_future = eval_df[[exog_col]]
                y_pred = model.predict(horizon, exog=exog_future)
                target_col = getattr(model, "target_col", None) or pipeline.get("target_col")
                y_true = eval_df[target_col].values if target_col else eval_df.iloc[:, 0].values
            else:
                y_pred = model.predict(horizon)
                y_true = eval_df.squeeze().values

        elif model_type == "tabular":
            target_col = pipeline["target_col"]
            X_train_tab = train_df.drop(columns=[target_col])
            y_train_tab = train_df[target_col]
            X_eval_tab = eval_df.drop(columns=[target_col])
            y_true = eval_df[target_col].values
            if X_train_tab.shape[1] == 0:
                raise ValueError(
                    "Tabular pipeline has no features: preprocessing is empty and "
                    "the only column is the target. Add at least one preprocessor "
                    "(e.g. LagFeatures, CalendarFeatures) to create input features."
                )
            model.fit(X_train_tab, y_train_tab)
            y_pred = model.predict(X_eval_tab)

        elif model_type == "tensor":
            if X_train is None or y_train is None:
                raise ValueError(
                    "Preprocessing for tensor model must return (X, y) tuples"
                )
            model.fit(X_train, y_train, X_val=X_eval, y_val=y_eval)
            y_pred = model.predict_from_sequences(X_eval)
            y_true = y_eval

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    except Exception:
        print("Full error traceback:")
        traceback.print_exc()
        raise

    # ---------- Train-set predictions (for overfit/underfit check) ----------
    y_train_true, y_train_pred = None, None
    if model_type == "forecast":
        if hasattr(model, "get_train_predictions"):
            pred_series = model.get_train_predictions()
            target_col = getattr(model, "target_col", None) or pipeline.get("target_col")
            if target_col and target_col in train_df.columns:
                y_train_true = train_df.loc[pred_series.index, target_col].values
            else:
                y_train_true = train_df.loc[pred_series.index].squeeze().values
            y_train_pred = np.asarray(pred_series.values, dtype=float)
        else:
            # Persistence: one-step-ahead in-sample (pred[t] = true[t-1])
            arr = train_df.squeeze().values
            if len(arr) > 1:
                y_train_true = np.asarray(arr[1:], dtype=float)
                y_train_pred = np.asarray(arr[:-1], dtype=float)
    elif model_type == "tabular":
        y_train_true = np.asarray(y_train_tab.values, dtype=float)
        y_train_pred = np.asarray(model.predict(X_train_tab), dtype=float)
    elif model_type == "tensor":
        y_train_true = np.asarray(y_train, dtype=float)
        y_train_pred = model.predict_from_sequences(X_train)
        if isinstance(y_train_pred, torch.Tensor):
            y_train_pred = y_train_pred.detach().cpu().numpy()
    

    # Inverse transforms so y_true and y_pred are in original units (same scale for metric).
    inv_kw = {}
    if model_type in ("tabular", "tensor"):
        tc = pipeline.get("target_col")
        if tc is not None:
            inv_kw["target_col"] = tc
    for step in reversed(pipeline.get("preprocessing", [])):
        if hasattr(step, "inverse"):
            if model_type == "tensor":
                if isinstance(y_pred, torch.Tensor):
                    y_pred_np = y_pred.detach().cpu().numpy()
                    y_pred_np = step.inverse(y_pred_np, **inv_kw)
                    y_pred = torch.tensor(y_pred_np, dtype=torch.float32)
                else:
                    y_pred = step.inverse(y_pred, **inv_kw)
                y_true = step.inverse(np.asarray(y_true), **inv_kw)
            elif model_type in ("forecast", "tabular"):
                y_pred = step.inverse(y_pred, **inv_kw)
                y_true = step.inverse(np.asarray(y_true), **inv_kw)

    # Inverse transform train outputs when present
    if y_train_true is not None and y_train_pred is not None:
        for step in reversed(pipeline.get("preprocessing", [])):
            if hasattr(step, "inverse"):
                if model_type == "tensor":
                    y_train_pred = step.inverse(np.asarray(y_train_pred), **inv_kw)
                    y_train_true = step.inverse(np.asarray(y_train_true), **inv_kw)
                else:
                    y_train_pred = step.inverse(np.asarray(y_train_pred), **inv_kw)
                    y_train_true = step.inverse(np.asarray(y_train_true), **inv_kw)
        y_train_true = np.asarray(y_train_true)
        y_train_pred = np.asarray(y_train_pred)

    elapsed_sec = time.perf_counter() - start_time

    # Eval (val/test) metrics
    if isinstance(metric_fn, dict):
        out = {name: fn(y_true, y_pred) for name, fn in metric_fn.items()}
    else:
        out = {"metric": metric_fn(y_true, y_pred)}

    out["elapsed_sec"] = elapsed_sec

    # Train metrics (for overfit/underfit)
    if y_train_true is not None and y_train_pred is not None:
        if isinstance(metric_fn, dict):
            for name, fn in metric_fn.items():
                out[f"train_{name}"] = fn(y_train_true, y_train_pred)
        else:
            out["train_metric"] = metric_fn(y_train_true, y_train_pred)
    else:
        if isinstance(metric_fn, dict):
            for name in metric_fn:
                out[f"train_{name}"] = float("nan")
        else:
            out["train_metric"] = float("nan")

    if hasattr(model, "history") and model.history is not None:
        out["history"] = model.history

    return out


def get_test_forecast(pipeline, df, model_params=None):
    """
    Get test-set predictions for a pipeline. Returns a DataFrame with DatetimeIndex
    and columns "actual" and "forecast" (in original units), for saving as CSV to
    send to optimizers.

    Uses the same split, preprocessing, fit, and inverse logic as evaluate_model(eval_set="test").
    """
    model_params = _native_params(model_params or {})
    model_type = pipeline.get("model_type", "forecast")

    train_df, val_df, test_df = pipeline["splitter"](df)
    eval_df = test_df

    X_train, y_train, X_eval, y_eval = None, None, None, None
    eval_index = None
    for step in pipeline.get("preprocessing", []):
        step.fit(train_df)
        res_train = step.transform(train_df)
        res_eval = step.transform(eval_df)
        if isinstance(res_train, tuple):
            X_train, y_train = res_train
            X_eval, y_eval = res_eval
            seq_len = getattr(step, "seq_len", None)
            eval_index = eval_df.index[seq_len:] if seq_len is not None else eval_df.index[len(eval_df) - len(y_eval):]
            break
        train_df = res_train
        eval_df = res_eval

    model = pipeline["model"](**model_params)
    warnings.simplefilter("ignore", ConvergenceWarning)

    try:
        if model_type == "forecast":
            model.fit(train_df)
            horizon = len(eval_df)
            exog_col = getattr(model, "exog_col", None) or pipeline.get("exog_col")
            if exog_col and exog_col in eval_df.columns:
                exog_future = eval_df[[exog_col]]
                y_pred = model.predict(horizon, exog=exog_future)
                target_col = getattr(model, "target_col", None) or pipeline.get("target_col")
                y_true = eval_df[target_col].values if target_col else eval_df.iloc[:, 0].values
            else:
                y_pred = model.predict(horizon)
                y_true = eval_df.squeeze().values
            eval_index = eval_df.index

        elif model_type == "tabular":
            target_col = pipeline["target_col"]
            X_train_tab = train_df.drop(columns=[target_col])
            y_train_tab = train_df[target_col]
            X_eval_tab = eval_df.drop(columns=[target_col])
            y_true = eval_df[target_col].values
            model.fit(X_train_tab, y_train_tab)
            y_pred = model.predict(X_eval_tab)
            eval_index = eval_df.index

        elif model_type == "tensor":
            if X_train is None or y_train is None:
                raise ValueError("Preprocessing for tensor model must return (X, y) tuples")
            model.fit(X_train, y_train, X_val=X_eval, y_val=y_eval)
            y_pred = model.predict_from_sequences(X_eval)
            y_true = y_eval

        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    except Exception:
        traceback.print_exc()
        raise

    y_pred = np.asarray(y_pred)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_true = np.asarray(y_true, dtype=float)

    inv_kw = {}
    if model_type in ("tabular", "tensor"):
        tc = pipeline.get("target_col")
        if tc is not None:
            inv_kw["target_col"] = tc
    for step in reversed(pipeline.get("preprocessing", [])):
        if hasattr(step, "inverse"):
            if model_type == "tensor":
                y_pred = step.inverse(y_pred, **inv_kw)
                y_true = step.inverse(y_true, **inv_kw)
            else:
                y_pred = step.inverse(y_pred, **inv_kw)
                y_true = step.inverse(y_true, **inv_kw)

    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    if eval_index is None or len(eval_index) != len(y_true):
        eval_index = pd.RangeIndex(len(y_true))
    out = pd.DataFrame({"actual": y_true, "forecast": y_pred}, index=eval_index)
    out.index.name = "datetime"
    return out


def best_model_test_forecast(configs, comparison_df, primary_metric="rmse"):
    """
    Get the test-set forecast from the best model in comparison_df.

    configs: same list as for compare_models_on_test: (name, data, pipelines_list, results_df, best_hp).
    comparison_df: DataFrame returned by compare_models_on_test (sorted by primary_metric).
    primary_metric: column name to pick best model (default "rmse"; lower is better).

    Returns:
        best_name: str
        forecast_df: DataFrame with index=datetime, columns "actual", "forecast".
    """
    best_row = comparison_df.sort_values(by=primary_metric, ascending=True).iloc[0]
    best_name = best_row["model"]
    for name, data, pipelines_list, results_df, best_hp in configs:
        if name != best_name:
            continue
        best_pipeline_row = results_df.iloc[0]
        pipe_idx = int(best_pipeline_row["pipeline_idx"])
        if pipe_idx < 1 or pipe_idx > len(pipelines_list):
            raise IndexError(f"Best model '{best_name}': invalid pipeline_idx {pipe_idx}")
        pipeline = pipelines_list[pipe_idx - 1]
        forecast_df = get_test_forecast(pipeline, data, model_params=best_hp)
        return best_name, forecast_df
    raise ValueError(f"Best model '{best_name}' not found in configs.")


def evaluate_model_rolling(
    pipeline, df, metric_fn, model_params=None, window_size=24, use_test=False
):
    """
    Rolling-window evaluation (e.g. 24h blocks).

    - Uses the same split / preprocessing / model logic as evaluate_model.
    - Computes the metric on non-overlapping windows of length `window_size`
      over the chosen evaluation segment (validation by default, or test if use_test=True).
    - Returns (metric_mean, metric_std) over all windows.
    """
    model_params = model_params or {}
    model_type = pipeline.get("model_type", "forecast")

    train_df, val_df, test_df = pipeline["splitter"](df)
    eval_df = test_df if use_test else val_df

    X_train, y_train, X_val, y_val = None, None, None, None
    # Note: preprocessing is always fit on train_df, eval_df goes through transform only
    for step in pipeline.get("preprocessing", []):
        step.fit(train_df)
        res_train = step.transform(train_df)
        res_val = step.transform(eval_df)
        if isinstance(res_train, tuple):
            X_train, y_train = res_train
            X_val, y_val = res_val
            break
        train_df = res_train
        eval_df = res_val

    model = pipeline["model"](**(model_params or {}))
    warnings.simplefilter("ignore", ConvergenceWarning)

    try:
        if model_type == "forecast":
            model.fit(train_df)
            horizon = len(eval_df)
            exog_col = getattr(model, "exog_col", None) or pipeline.get("exog_col")
            if exog_col and exog_col in eval_df.columns:
                exog_future = eval_df[[exog_col]]
                y_pred = model.predict(horizon, exog=exog_future)
                target_col = getattr(model, "target_col", None) or pipeline.get("target_col")
                y_true = eval_df[target_col].values if target_col else eval_df.iloc[:, 0].values
            else:
                y_pred = model.predict(horizon)
                y_true = eval_df.squeeze().values

        elif model_type == "tabular":
            target_col = pipeline["target_col"]
            X_train_tab = train_df.drop(columns=[target_col])
            y_train_tab = train_df[target_col]
            X_val_tab = eval_df.drop(columns=[target_col])
            y_true = eval_df[target_col].values
            model.fit(X_train_tab, y_train_tab)
            y_pred = model.predict(X_val_tab)

        elif model_type == "tensor":
            if X_train is None or y_train is None:
                raise ValueError(
                    "Preprocessing for tensor model must return (X, y) tuples"
                )
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            y_pred = model.predict_from_sequences(X_val)
            y_true = y_val

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    except Exception:
        print("Full error traceback (rolling):")
        traceback.print_exc()
        raise

    # Inverse transforms so y_true and y_pred are in original units (same as evaluate_model).
    inv_kw = {}
    if model_type in ("tabular", "tensor"):
        tc = pipeline.get("target_col")
        if tc is not None:
            inv_kw["target_col"] = tc
    for step in reversed(pipeline.get("preprocessing", [])):
        if hasattr(step, "inverse"):
            if model_type == "tensor":
                if isinstance(y_pred, torch.Tensor):
                    y_pred_np = y_pred.detach().cpu().numpy()
                    y_pred_np = step.inverse(y_pred_np, **inv_kw)
                    y_pred = torch.tensor(y_pred_np, dtype=torch.float32)
                else:
                    y_pred = step.inverse(y_pred, **inv_kw)
                y_true = step.inverse(np.asarray(y_true), **inv_kw)
            elif model_type in ("forecast", "tabular"):
                y_pred = step.inverse(y_pred, **inv_kw)
                y_true = step.inverse(np.asarray(y_true), **inv_kw)

    # Rolling windows over y_true / y_pred
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if isinstance(metric_fn, dict):
        if window_size is None or window_size <= 0 or window_size > n:
            return {name: (fn(y_true, y_pred), 0.0) for name, fn in metric_fn.items()}
        out = {}
        for name, fn in metric_fn.items():
            vals = []
            for start in range(0, n - window_size + 1, window_size):
                end = start + window_size
                vals.append(fn(y_true[start:end], y_pred[start:end]))
            vals = np.asarray(vals, dtype=float)
            out[name] = (float(vals.mean()), float(vals.std()))
        return out

    if window_size is None or window_size <= 0 or window_size > n:
        metrics = [metric_fn(y_true, y_pred)]
    else:
        metrics = []
        for start in range(0, n - window_size + 1, window_size):
            end = start + window_size
            m = metric_fn(y_true[start:end], y_pred[start:end])
            metrics.append(m)
    metrics = np.asarray(metrics, dtype=float)
    return float(metrics.mean()), float(metrics.std())


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, eps=1e-10, min_denom_frac=1e-6):
    """
    Mean absolute percentage error (%). Scale-invariant; use with RMSE for full picture.
    Points where |y_true| is below a scale-relative threshold are excluded so MAPE
    does not explode when true values are near zero (denom would be tiny).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_y = np.abs(y_true)
    # Only use points where denominator is meaningful (avoid 0 or near-zero true values)
    scale = np.median(abs_y)
    threshold = max(eps, min_denom_frac * scale) if scale > 0 else eps
    valid = abs_y >= threshold
    if not np.any(valid):
        return float("nan")
    y_t, y_p = y_true[valid], y_pred[valid]
    denom = abs_y[valid]  # already >= threshold
    return float(np.mean(np.abs((y_t - y_p) / denom)))


def r2(y_true, y_pred):
    """R² (coefficient of determination). Fraction of variance explained."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mbe(y_true, y_pred):
    """
    Mean bias error: mean(pred - true).
    Positive = overprediction on average, negative = underprediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def _preprocessing_name(pipe) -> str:
    """Return a readable description of the preprocessing steps."""
    steps = pipe.get("preprocessing", [])
    if not steps:
        return ""
    names = [type(step).__name__ for step in steps]
    return " + ".join(names)


def compare_models_on_test(configs, metric_fn=None):
    """
    Evaluate each model on the test set and return a comparison DataFrame.

    configs: list of tuples (model_name, data, pipelines_list, results_df, best_hyperparameters).
      - data: the DataFrame or Series passed to run_pipelines_hyperparam (e.g. df_elec or series_elec).
      - pipelines_list: same list of pipelines used for that model (e.g. pipelines_persistence).
      - results_df: the sorted results DataFrame from run_pipelines_hyperparam (best row first).
      - best_hyperparameters: the dict returned by run_pipelines_hyperparam for that model.
    metric_fn: dict of name -> callable (e.g. {"rmse": rmse, "mape": mape}), or None for {"rmse": rmse, "mape": mape}.

    Returns: DataFrame with columns "model" and one column per metric (e.g. rmse, mape).
    """
    if metric_fn is None:
        metric_fn = {"rmse": rmse, "mape": mape}
    rows = []
    for name, data, pipelines_list, results_df, best_hp in configs:
        best_row = results_df.iloc[0]
        pipe_idx = int(best_row["pipeline_idx"])
        if pipe_idx < 1 or pipe_idx > len(pipelines_list):
            raise IndexError(
                f"Model '{name}': best row has pipeline_idx={pipe_idx} but pipelines_list has "
                f"{len(pipelines_list)} pipeline(s). Re-run run_pipelines_hyperparam with the same "
                "pipelines list so results_df and pipelines_list match."
            )
        pipeline = pipelines_list[pipe_idx - 1]
        scores = evaluate_model(
            pipeline, data, metric_fn, model_params=best_hp, eval_set="test"
        )
        rows.append({"model": name, **scores})
    return pd.DataFrame(rows)


def run_pipelines_hyperparam(df, pipelines, metric_fn, primary_metric=None, sort_ascending=True):
    """
    Run pipelines over hyperparameter grid; return sorted results and best config.

    metric_fn: callable(y_true, y_pred) -> float, or dict of name -> callable for multiple metrics.
    primary_metric: when metric_fn is dict, sort by this key (default: first key). Ignored for single metric.
    sort_ascending: True = lower is better (e.g. RMSE, MAPE); False = higher is better (e.g. R²).

    Returns:
        results_df: DataFrame sorted by primary_metric (best first).
        best_hyperparameters: dict of hyperparam name -> value for the best row.
        best_preprocessing: dict with key "preprocessing" -> string of step names for the best row.
    """
    is_multi = isinstance(metric_fn, dict)
    metric_names = list(metric_fn.keys()) if is_multi else ["metric"]
    primary = primary_metric if is_multi and primary_metric is not None else (metric_names[0] if is_multi else "metric")

    results = []
    for pipe_idx, pipe in enumerate(pipelines, start=1):
        print(f"\nRunning pipeline {pipe_idx}")
        hyperparams = pipe.get("hyperparams", {})
        preproc_str = _preprocessing_name(pipe)

        if not hyperparams:
            score = evaluate_model(pipe, df, metric_fn)
            row = {"pipeline_idx": pipe_idx, "preprocessing": preproc_str}
            row.update(score)
            results.append(row)
            continue

        keys, values = zip(*hyperparams.items())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            score = evaluate_model(
                pipeline=pipe, df=df, metric_fn=metric_fn, model_params=params
            )
            row = {"pipeline_idx": pipe_idx, "preprocessing": preproc_str, **params}
            row.update(score)
            results.append(row)

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by=primary, ascending=sort_ascending).reset_index(drop=True)
    best_row = df_out.iloc[0]
    exclude = {"pipeline_idx", "preprocessing", "metric_std", "elapsed_sec", "history"} | set(metric_names)
    exclude |= {k for k in df_out.columns if isinstance(k, str) and k.startswith("train_")}
    best_hyperparameters = {
        k: best_row[k] for k in df_out.columns
        if k not in exclude and not (isinstance(k, str) and k.endswith("_std"))
    }
    best_preprocessing = {"preprocessing": best_row["preprocessing"]}
    return df_out, best_hyperparameters, best_preprocessing


def run_pipelines_hyperparam_rolling(
    df, pipelines, metric_fn, window_size=24, use_test=False, primary_metric=None, sort_ascending=True
):
    """
    Rolling-window runner. Returns (results_df, best_hyperparameters, best_preprocessing).
    When metric_fn is a dict, each metric gets columns name and name_std (e.g. rmse, rmse_std).
    """
    is_multi = isinstance(metric_fn, dict)
    metric_names = list(metric_fn.keys()) if is_multi else ["metric"]
    primary = primary_metric if is_multi and primary_metric is not None else (metric_names[0] if is_multi else "metric")

    results = []
    for pipe_idx, pipe in enumerate(pipelines, start=1):
        print(f"\nRunning pipeline {pipe_idx} (rolling)")
        hyperparams = pipe.get("hyperparams", {})
        preproc_str = _preprocessing_name(pipe)

        if not hyperparams:
            scores = evaluate_model_rolling(
                pipe, df, metric_fn, window_size=window_size, use_test=use_test
            )
            row = {"pipeline_idx": pipe_idx, "preprocessing": preproc_str}
            if is_multi:
                for name, (m, s) in scores.items():
                    row[name] = m
                    row[f"{name}_std"] = s
            else:
                row["metric"], row["metric_std"] = scores
            results.append(row)
            continue

        keys, values = zip(*hyperparams.items())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            scores = evaluate_model_rolling(
                pipeline=pipe,
                df=df,
                metric_fn=metric_fn,
                model_params=params,
                window_size=window_size,
                use_test=use_test,
            )
            row = {"pipeline_idx": pipe_idx, "preprocessing": preproc_str, **params}
            if is_multi:
                for name, (m, s) in scores.items():
                    row[name] = m
                    row[f"{name}_std"] = s
            else:
                row["metric"], row["metric_std"] = scores
            results.append(row)

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by=primary, ascending=sort_ascending).reset_index(drop=True)
    best_row = df_out.iloc[0]
    exclude = {"pipeline_idx", "preprocessing", "metric_std"}
    for n in metric_names:
        exclude.add(n)
        exclude.add(f"{n}_std")
    exclude |= {k for k in df_out.columns if isinstance(k, str) and k.startswith("train_")}
    best_hyperparameters = {k: best_row[k] for k in df_out.columns if k not in exclude}
    best_preprocessing = {"preprocessing": best_row["preprocessing"]}
    return df_out, best_hyperparameters, best_preprocessing

def format_for_csv(df, decimal_places=2):
    out = df.copy()
    for col in out.select_dtypes(include=["number"]).columns:
        if out[col].dtype in ("int64", "int32"):
            out[col] = out[col].apply(lambda x: f"{int(x):,}")
        else:
            out[col] = out[col].apply(lambda x: f"{x:,.{decimal_places}f}")
    return out

__all__ = [
    "best_model_test_forecast",
    "compare_models_on_test",
    "evaluate_model",
    "evaluate_model_rolling",
    "get_test_forecast",
    "mape",
    "mbe",
    "r2",
    "rmse",
    "run_pipelines_hyperparam",
    "run_pipelines_hyperparam_rolling",
    "set_seed",
    "format_for_csv",
]
