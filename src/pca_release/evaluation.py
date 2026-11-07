from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction arrays must have the same shape")
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    balanced_accuracy = np.nanmean([sensitivity, specificity])
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else np.nan
    return {
        "N": int(truth.size),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "BACC": float(balanced_accuracy),
        "F1": float(f1),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    region: str,
) -> pd.DataFrame:
    rows = []
    for model_name, y_pred in predictions.items():
        rows.append({"Region": region, "Model": model_name, **binary_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)
