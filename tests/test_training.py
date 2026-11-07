import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pca_release.training as training_module
from pca_release.training import model_specs, prepare_training_data, train_region_models


def test_model_specs_are_complete():
    assert set(model_specs()) == {"LDA", "QDA", "KNN", "SVM", "RF", "MLP"}


def test_prepare_training_data_selects_ids():
    features = ["feature"]
    regions = [{"key": "region", "truth_column": "truth", "binarize_truth": True}]
    frame = pd.DataFrame(
        {
            "subject": ["a", "b"],
            "feature": [1.0, 2.0],
            "truth": [0, 2],
        }
    )
    train_ids = pd.DataFrame({"subject": ["b"]})
    selected = prepare_training_data(frame, train_ids, "subject", features, regions)
    assert selected["subject"].tolist() == ["b"]


def test_prepare_training_data_rejects_duplicate_ids():
    frame = pd.DataFrame({"subject": ["a"], "feature": [1.0], "truth": [0]})
    train_ids = pd.DataFrame({"subject": ["a", "a"]})
    regions = [{"key": "region", "truth_column": "truth", "binarize_truth": True}]
    with pytest.raises(ValueError, match="unique"):
        prepare_training_data(frame, train_ids, "subject", ["feature"], regions)


def test_train_region_models_writes_fitted_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        training_module,
        "model_specs",
        lambda random_state=42: {"LDA": (LinearDiscriminantAnalysis(), {})},
    )
    frame = pd.DataFrame(
        {
            "feature": [0.0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.6],
            "truth": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    regions = [{"key": "region", "truth_column": "truth", "binarize_truth": False}]
    results = train_region_models(
        frame,
        ["feature"],
        regions,
        tmp_path,
        cv_splits=2,
        n_jobs=1,
    )
    assert results["filename"].tolist() == ["region_LDA.pkl"]
    assert (tmp_path / "region_LDA.pkl").is_file()
