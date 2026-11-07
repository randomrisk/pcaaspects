#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pca_release.config import load_config, resolve_path
from pca_release.io import read_csv
from pca_release.training import prepare_training_data, train_region_models, write_model_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCA-ASPECTS regional classifiers.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    paths = config["paths"]
    columns = config["columns"]
    subject_column = columns["subject"]
    features = columns["features"]
    regions = config["regions"]
    random_state = int(config["training"]["random_state"])
    model_dir = args.model_dir or resolve_path(root, paths["training_model_dir"])
    output_dir = args.output_dir or resolve_path(root, paths["training_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_columns = [region["truth_column"] for region in regions]
    frame = read_csv(
        resolve_path(root, paths["feature_table"]),
        [subject_column, *features, *truth_columns],
    )
    train_ids = read_csv(resolve_path(root, paths["train_ids"]), [subject_column])
    training = prepare_training_data(frame, train_ids, subject_column, features, regions)
    results = train_region_models(
        training,
        features,
        regions,
        model_dir,
        cv_splits=args.cv_splits,
        random_state=random_state,
        n_jobs=args.jobs,
    )
    results.to_csv(output_dir / "cross_validation_results.csv", index=False)
    parameters = {
        f"{row.region}_{row.model}": row.best_parameters
        for row in results.itertuples(index=False)
    }
    with (output_dir / "best_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2)
    write_model_manifest(model_dir, model_dir / "MODEL_MANIFEST.csv")
    print(f"Wrote {len(results)} models to {model_dir}")
    print(f"Wrote training results to {output_dir}")


if __name__ == "__main__":
    main()
