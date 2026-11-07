#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pca_release.config import load_config, resolve_path
from pca_release.evaluation import evaluate_predictions
from pca_release.io import read_csv, select_subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stored PCA-ASPECTS region classifiers.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    paths = config["paths"]
    subject_column = config["columns"]["subject"]
    features = config["columns"]["features"]
    regions = config["regions"]
    model_names = config["models"]

    truth_columns = [region["truth_column"] for region in regions]
    frame = read_csv(resolve_path(root, paths["feature_table"]), [subject_column, *features, *truth_columns])
    test_ids = read_csv(resolve_path(root, paths["test_ids"]), [subject_column])
    test = select_subjects(frame, test_ids, subject_column)
    if test.empty:
        raise ValueError("No feature-table rows matched the test IDs")

    model_dir = resolve_path(root, paths["model_dir"])
    output_dir = resolve_path(root, paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    design = test[features].to_numpy(float)
    tables = []

    for region in regions:
        key = region["key"]
        truth = test[region["truth_column"]].to_numpy(int)
        if region.get("binarize_truth", False):
            truth = (truth > 0).astype(int)
        predictions = {}
        for model_name in model_names:
            model_path = model_dir / f"{key}_{model_name}.pkl"
            if model_path.exists():
                predictions[model_name] = np.asarray(joblib.load(model_path).predict(design), dtype=int)
        if predictions:
            tables.append(evaluate_predictions(truth, predictions, key))

    if not tables:
        raise FileNotFoundError(f"No configured model files were found in {model_dir}")
    results = pd.concat(tables, ignore_index=True)
    results.to_csv(output_dir / "region_performance.csv", index=False)
    print(f"Wrote {output_dir / 'region_performance.csv'}")


if __name__ == "__main__":
    main()
