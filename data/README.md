# Input data

Clinical data are not distributed with this code release.

The evaluation workflow expects:

- `features.csv`: one row per subject with the configured subject, feature, and truth columns.
- `test_ids.csv`: one configured subject identifier per row.

The training workflow additionally expects `train_ids.csv`, with one unique subject identifier per row. The feature table must contain these columns:

```text
subject
PCA_qfv
thalamus_qfv
cerebellum_qfv
pons_qfv
midbrain_qfv
logVol
PCA
Thalamus
cerebellum
pons_y
midbrain_y
```

Column names may be changed in `configs/default.yaml`. Regional labels are binary during training; nonzero PCA, thalamus, and cerebellum labels are converted to 1.

The visualization workflow additionally expects JSON dataset indexes, NIfTI DWI and stroke volumes, registration transforms, and a PCA-ASPECTS atlas. Pass their locations through command-line arguments or update the configuration locally.

Do not commit subject-level data, identifiers, or derived clinical figures to this directory.
