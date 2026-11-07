from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def model_specs(random_state: int = 42) -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "LDA": (LinearDiscriminantAnalysis(), {}),
        "QDA": (QuadraticDiscriminantAnalysis(), {}),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 13],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "SVM": (
            SVC(probability=True),
            {
                "model__kernel": ["linear", "rbf"],
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", "auto"],
            },
        ),
        "RF": (
            RandomForestClassifier(random_state=random_state),
            {
                "model__n_estimators": [100, 250, 500],
                "model__max_depth": [None, 4, 8, 10],
                "model__max_features": ["sqrt", None],
            },
        ),
        "MLP": (
            MLPClassifier(max_iter=2000, random_state=random_state),
            {
                "model__hidden_layer_sizes": [(32,), (64,), (32, 16)],
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001, 0.01],
            },
        ),
    }


def prepare_training_data(
    frame: pd.DataFrame,
    train_ids: pd.DataFrame,
    subject_column: str,
    features: list[str],
    regions: list[dict],
) -> pd.DataFrame:
    required = [subject_column, *features, *(region["truth_column"] for region in regions)]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature-table columns: {missing}")
    if subject_column not in train_ids.columns:
        raise ValueError(f"Missing train-ID column: {subject_column}")
    if train_ids[subject_column].duplicated().any():
        raise ValueError("Training IDs must be unique")
    selected = frame[frame[subject_column].isin(set(train_ids[subject_column]))].copy()
    if selected.empty:
        raise ValueError("No feature-table rows matched the training IDs")
    if selected[subject_column].duplicated().any():
        raise ValueError("The feature table must contain one row per training subject")
    if selected[required].isna().any().any():
        raise ValueError("Training columns contain missing values")
    design = selected[features].to_numpy(float)
    if not np.isfinite(design).all():
        raise ValueError("Training features contain non-finite values")
    return selected.reset_index(drop=True)


def train_region_models(
    frame: pd.DataFrame,
    features: list[str],
    regions: list[dict],
    model_dir: Path,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    model_dir.mkdir(parents=True, exist_ok=True)
    design = frame[features].to_numpy(float)
    rows = []
    for region in regions:
        key = region["key"]
        truth = frame[region["truth_column"]].to_numpy(int)
        if region.get("binarize_truth", False):
            truth = (truth > 0).astype(int)
        classes, counts = np.unique(truth, return_counts=True)
        if classes.size != 2 or set(classes) != {0, 1}:
            raise ValueError(f"Region {key} must contain binary classes 0 and 1")
        if counts.min() < cv_splits:
            raise ValueError(f"Region {key} has fewer than {cv_splits} samples in its smallest class")
        cross_validation = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state,
        )
        for name, (estimator, grid) in model_specs(random_state).items():
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="balanced_accuracy",
                cv=cross_validation,
                n_jobs=n_jobs,
                error_score="raise",
                refit=True,
            )
            search.fit(design, truth)
            filename = f"{key}_{name}.pkl"
            joblib.dump(search.best_estimator_, model_dir / filename)
            rows.append(
                {
                    "region": key,
                    "model": name,
                    "best_cv_balanced_accuracy": float(search.best_score_),
                    "best_parameters": search.best_params_,
                    "filename": filename,
                }
            )
    return pd.DataFrame(rows)


def write_model_manifest(model_dir: Path, output: Path) -> None:
    rows = []
    for path in sorted(model_dir.glob("*.pkl")):
        model = joblib.load(path)
        rows.append(
            {
                "filename": path.name,
                "estimator": f"{type(model).__module__}.{type(model).__name__}",
                "n_features_in": int(model.n_features_in_),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)
