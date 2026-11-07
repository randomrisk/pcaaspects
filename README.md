# PCA-ASPECTS code release

This package provides training, evaluation, visualization code, and pretrained models for automated PCA-ASPECTS scoring from acute ischemic stroke MRI using quantitative lesion features and machine learning.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Training

Place the authorized feature table and training IDs in `data/`, then run:

```bash
python scripts/train_models.py --config configs/default.yaml
```

By default, newly trained models are written to `outputs/training/models` so the distributed models are not overwritten. Cross-validation scores and selected parameters are written to `outputs/training`. Use `--model-dir` and `--output-dir` to select different destinations.

Training uses five-fold stratified cross-validation, balanced accuracy for model selection, a fixed random seed of 42, and the hyperparameter grids used for the released regional classifiers. The training table schema is documented in `data/README.md`.

## Pretrained models

The 30 regional models in `models/` were serialized with scikit-learn 1.7.2. They cover LDA, QDA, KNN, SVM, random forest, and multilayer perceptron classifiers for five anatomical regions. They require six features in this order:

```text
PCA_qfv, thalamus_qfv, cerebellum_qfv, pons_qfv, midbrain_qfv, logVol
```

Model types, feature counts, file sizes, and SHA-256 checksums are listed in `models/MODEL_MANIFEST.csv`. Load pickle files only from a trusted copy of this release.

## Evaluation

Place authorized input tables in `data/`, update `configs/default.yaml` if needed, and run:

```bash
python scripts/run_evaluation.py --config configs/default.yaml
```

Results are written to `outputs/evaluation/region_performance.csv`.

## Subject visualization

Create a filepath manifest:

```bash
python scripts/build_visualization_manifest.py \
  --subject-csv /path/to/subjects.csv \
  --index-root /path/to/mni_dataset \
  --raw-root /path/to/raw_dataset \
  --atlas /path/to/AspectsPcaAtlas_MNI182.nii.gz \
  --output outputs/subject_filepaths.csv
```

Render one subject or all manifest-ready subjects:

```bash
python scripts/visualize_subjects.py \
  --subject SUBJECT_ID \
  --manifest outputs/subject_filepaths.csv \
  --output-dir outputs/visualizations

python scripts/visualize_subjects.py \
  --all \
  --manifest outputs/subject_filepaths.csv \
  --output-dir outputs/visualizations
```

## Intended use

The code and models are provided for research reproducibility and are not validated for clinical use. Do not redistribute clinical data, subject-level outputs, or third-party assets without authorization.
