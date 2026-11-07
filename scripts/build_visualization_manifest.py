#!/usr/bin/env python3
"""Resolve visualization input paths for a subject list."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from visualize_subjects import (
    DEFAULT_ATLAS,
    DEFAULT_INDEX_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_SUBJECT_CSV,
    indexed_modality,
    indexed_path,
    load_json_indices,
    load_subjects,
    raw_path,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "subject_filepaths.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-csv", type=Path, default=DEFAULT_SUBJECT_CSV)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = load_subjects(args.subject_csv)
    indices = load_json_indices(args.index_root)
    rows = []

    for subject in subjects:
        row = {
            "subject": subject,
            "dataset_subject": "",
            "cohort": "",
            "mni_dwi": "",
            "mni_stroke": "",
            "raw_dwi": "",
            "raw_stroke": "",
            "raw_stroke_mode": "native",
            "dwi_affine": "",
            "atlas_mni": str(args.atlas),
            "status": "ready",
            "error": "",
        }
        try:
            token = subject.removeprefix("sub-")
            candidates = list(dict.fromkeys([subject, f"sub-{token.zfill(8)}"]))
            match = None
            for candidate in candidates:
                candidate_matches = [
                    (path.stem, data[candidate])
                    for path, data in indices
                    if candidate in data
                ]
                if candidate_matches:
                    candidate_matches.sort(
                        key=lambda item: (
                            "DWI/DWI_MNI.nii.gz" in item[1]
                            or "DWI_proc/DWI_MNI.nii.gz" in item[1],
                            "DWI/stroke_MNI.nii.gz" in item[1]
                            or "DWI_proc/stroke_MNI.nii.gz" in item[1],
                            len(item[1]),
                        ),
                        reverse=True,
                    )
                    cohort, entries = candidate_matches[0]
                    match = (candidate, cohort, entries)
                    break
            if match is None:
                raise KeyError(f"{subject} is absent from all JSON indices (aliases tried: {candidates})")
            dataset_subject, cohort, entries = match
            dwi_relative = indexed_modality(entries, "DWI_MNI.nii.gz")
            stroke_relative = indexed_modality(entries, "stroke_MNI.nii.gz")
            mni_dwi = indexed_path(args.index_root, cohort, dataset_subject, dwi_relative)
            mni_stroke = indexed_path(args.index_root, cohort, dataset_subject, stroke_relative)
            raw_dwi = raw_path(args.raw_root, cohort, dataset_subject, dwi_relative)
            try:
                raw_stroke = raw_path(args.raw_root, cohort, dataset_subject, stroke_relative)
                raw_stroke_mode = "native"
            except FileNotFoundError:
                raw_stroke = None
                raw_stroke_mode = "resample_mni_to_raw"
            affine = raw_dwi.with_name(f"{dataset_subject}_DWI_affine.mat")
            if not affine.exists():
                raise FileNotFoundError(affine)
            row.update(
                {
                    "dataset_subject": dataset_subject,
                    "cohort": cohort,
                    "mni_dwi": str(mni_dwi),
                    "mni_stroke": str(mni_stroke),
                    "raw_dwi": str(raw_dwi),
                    "raw_stroke": str(raw_stroke) if raw_stroke else "",
                    "raw_stroke_mode": raw_stroke_mode,
                    "dwi_affine": str(affine),
                }
            )
        except Exception as error:
            row["status"] = "missing"
            row["error"] = str(error)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(row["status"] == "ready" for row in rows)
    missing = len(rows) - ready
    print(f"Wrote {args.output}")
    print(f"subjects={len(rows)} ready={ready} missing={missing}")


if __name__ == "__main__":
    main()
