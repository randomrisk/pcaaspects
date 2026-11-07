import numpy as np

from pca_release.evaluation import binary_metrics


def test_binary_metrics_counts_and_rates():
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FN"] == 1
    assert metrics["BACC"] == 0.5
