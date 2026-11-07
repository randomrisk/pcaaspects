# Pretrained models

This directory contains 30 pretrained scikit-learn pipelines for five regional PCA-ASPECTS classification tasks. The pipelines were saved with scikit-learn 1.7.2 and expect six features in this exact order:

1. `PCA_qfv`
2. `thalamus_qfv`
3. `cerebellum_qfv`
4. `pons_qfv`
5. `midbrain_qfv`
6. `logVol`

`MODEL_MANIFEST.csv` records each estimator type, expected feature count, file size, and SHA-256 checksum.

Only load pickle-based model files obtained from a trusted source because deserialization can execute code. These models are for research reproducibility and are not validated for clinical use.
