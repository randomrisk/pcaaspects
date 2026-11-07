#!/usr/bin/env python3
"""Create MNI- and raw-space DWI/stroke/PCA-ASPECTS image plates.

The dataset JSON files are the authoritative index used to associate a subject
with its cohort and MNI files. Native-space files are then resolved in the
matching cohort under the raw dataset root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pcaaspects-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "data" / "mni"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_SUBJECT_CSV = PROJECT_ROOT / "data" / "subjects.csv"
DEFAULT_ATLAS = PROJECT_ROOT / "data" / "AspectsPcaAtlas_MNI182.nii.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "visualizations"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "subject_filepaths.csv"

ATLAS_LABELS = {
    1: ("PCA-L", "#00BFC4"),
    2: ("PCA-R", "#F8766D"),
    3: ("Thalamus-L", "#7CAE00"),
    4: ("Thalamus-R", "#C77CFF"),
    5: ("Cerebellum-L", "#619CFF"),
    6: ("Cerebellum-R", "#FFB000"),
    7: ("Pons", "#F564E3"),
    8: ("Midbrain", "#8B8B8B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="Render one subject identifier.")
    parser.add_argument(
        "--all", action="store_true", help="Render every subject in --subject-csv."
    )
    parser.add_argument("--subject-csv", type=Path, default=DEFAULT_SUBJECT_CSV)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Pre-resolved file-path CSV used by --all (and by --subject when present).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--space", choices=["both", "mni", "raw"], default="both",
        help="Space(s) to render.",
    )
    parser.add_argument(
        "--mask-stroke-to-dwi",
        action="store_true",
        help="Restrict the displayed stroke mask to DWI brain support.",
    )
    parser.add_argument(
        "--native-raw-orientation",
        action="store_true",
        help="Display raw data in its native acquisition axes rather than canonical RAS axes.",
    )
    parser.add_argument(
        "--isotropic-raw",
        action="store_true",
        help="Reslice raw-space DWI, stroke, and atlas to an axis-aligned isotropic grid.",
    )
    args = parser.parse_args()
    if bool(args.subject) == bool(args.all):
        parser.error("Choose exactly one of --subject or --all.")
    return args


def normalize_subject_id(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    return value if value.startswith("sub-") else f"sub-{value}"


def load_subjects(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    available = rows[0].keys()
    key = next(
        (candidate for candidate in ("normalized_subject_id", "subject", "participant_id", "Subject_ID") if candidate in available),
        None,
    )
    if key is None:
        raise KeyError(f"No supported subject-ID column found in {path}: {list(available)}")
    return [normalize_subject_id(row[key]) for row in rows]


def load_json_indices(index_root: Path) -> list[tuple[Path, dict[str, list[str]]]]:
    indices = []
    for path in sorted(index_root.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            indices.append((path, json.load(handle)))
    if not indices:
        raise FileNotFoundError(f"No JSON indices found under {index_root}")
    return indices


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["subject"]: row for row in csv.DictReader(handle)}


def find_indexed_subject(
    subject: str, indices: list[tuple[Path, dict[str, list[str]]]]
) -> tuple[str, list[str]]:
    matches = [(path.stem, data[subject]) for path, data in indices if subject in data]
    if not matches:
        raise KeyError(f"{subject} is absent from all JSON indices")
    if len(matches) > 1:
        raise RuntimeError(f"{subject} occurs in multiple JSON indices: {matches}")
    return matches[0]


def indexed_path(index_root: Path, cohort: str, subject: str, relative: str) -> Path:
    relative_path = Path(relative)
    path = index_root / cohort / subject / relative_path.parent / f"{subject}_{relative_path.name}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def indexed_modality(entries: list[str], suffix: str) -> str:
    matches = [entry for entry in entries if Path(entry).name.lower() == suffix.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one JSON entry named {suffix}; found {matches}")
    return matches[0]


def raw_path(raw_root: Path, cohort: str, subject: str, indexed_relative: str) -> Path:
    relative = Path(indexed_relative)
    raw_name = relative.name.replace("_MNI.nii.gz", ".nii.gz")
    path = raw_root / cohort / subject / relative.parent / f"{subject}_{raw_name}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def squeeze_3d(image: sitk.Image) -> sitk.Image:
    if image.GetDimension() == 3:
        return image
    if image.GetDimension() == 4 and image.GetSize()[3] == 1:
        return image[:, :, :, 0]
    raise ValueError(f"Expected 3D image (or singleton 4D), got size {image.GetSize()}")


def read_image(path: Path, pixel_type: int) -> sitk.Image:
    return squeeze_3d(sitk.ReadImage(str(path), pixel_type))


def same_geometry(first: sitk.Image, second: sitk.Image, tolerance: float = 1e-5) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), atol=tolerance)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), atol=tolerance)
        and np.allclose(first.GetDirection(), second.GetDirection(), atol=tolerance)
    )


def resample_labels(
    labels: sitk.Image, reference: sitk.Image, transform: sitk.Transform | None = None
) -> sitk.Image:
    transform = transform or sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(
        labels,
        reference,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def load_canonical(path: Path) -> np.ndarray:
    return np.squeeze(nib.as_closest_canonical(nib.load(str(path))).get_fdata())


def canonical_from_sitk(image: sitk.Image, reference_path: Path) -> np.ndarray:
    """Orient a resampled SimpleITK label image like its reference NIfTI."""
    xyz = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    reference = nib.load(str(reference_path))
    label_image = nib.Nifti1Image(xyz, reference.affine, reference.header)
    return np.squeeze(nib.as_closest_canonical(label_image).get_fdata())


def native_from_sitk(image: sitk.Image) -> np.ndarray:
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


def isotropic_raw_arrays(
    dwi_path: Path, stroke_image: sitk.Image, atlas_image: sitk.Image
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dwi_image = nib.load(str(dwi_path))
    stroke_nib = nib.Nifti1Image(native_from_sitk(stroke_image), dwi_image.affine)
    atlas_nib = nib.Nifti1Image(native_from_sitk(atlas_image), dwi_image.affine)
    voxel_size = min(dwi_image.header.get_zooms()[:3])
    dwi_iso = resample_to_output(dwi_image, voxel_sizes=(voxel_size,) * 3, order=1)
    stroke_iso = resample_from_to(stroke_nib, dwi_iso, order=0)
    atlas_iso = resample_from_to(atlas_nib, dwi_iso, order=0)
    return (
        np.squeeze(dwi_iso.get_fdata()),
        np.squeeze(stroke_iso.get_fdata()),
        np.squeeze(atlas_iso.get_fdata()),
    )


def select_lesion_slices(stroke: np.ndarray, max_slices: int = 12) -> list[int]:
    """Select all lesion slices, evenly subsampling only when necessary."""
    indices = np.flatnonzero(np.any(stroke > 0.5, axis=(0, 1)))
    if indices.size == 0:
        raise ValueError("Stroke mask is empty")
    if indices.size <= max_slices:
        return indices.astype(int).tolist()
    positions = np.linspace(0, indices.size - 1, max_slices).round().astype(int)
    return indices[positions].astype(int).tolist()


def contrast_window(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    foreground = finite[finite > 0]
    values = foreground if foreground.size else finite
    low, high = np.percentile(values, [1, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def draw_panel(
    ax: plt.Axes,
    dwi: np.ndarray,
    stroke: np.ndarray | None,
    atlas: np.ndarray | None,
    window: tuple[float, float],
) -> None:
    dwi_view = np.rot90(dwi)
    ax.imshow(dwi_view, cmap="gray", vmin=window[0], vmax=window[1])
    if stroke is not None:
        stroke_view = np.rot90(stroke)
        lesion = np.ma.masked_where(stroke_view <= 0, stroke_view)
        ax.imshow(lesion, cmap=ListedColormap(["#FF2D2D"]), alpha=0.58)
        ax.contour((stroke_view > 0).astype(float), levels=[0.5], colors=["white"], linewidths=0.55)
    if atlas is not None:
        atlas_view = np.rot90(atlas)
        for label, (_, color) in ATLAS_LABELS.items():
            binary = (atlas_view == label).astype(float)
            if binary.max() > 0:
                ax.contour(binary, levels=[0.5], colors=[color], linewidths=1.0)
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_space(
    subject: str,
    space_name: str,
    dwi: np.ndarray,
    stroke: np.ndarray,
    atlas: np.ndarray,
    output_dir: Path,
    mask_stroke_to_dwi: bool = False,
    support_mask: np.ndarray | None = None,
    dpi: int = 300,
) -> Path:
    if not (dwi.shape == stroke.shape == atlas.shape):
        raise ValueError(
            f"Shape mismatch in {space_name}: DWI={dwi.shape}, stroke={stroke.shape}, atlas={atlas.shape}"
        )
    display_stroke = stroke.copy()
    if mask_stroke_to_dwi:
        support = support_mask > 0 if support_mask is not None else dwi > 0
        display_stroke[~support] = 0
    slices = select_lesion_slices(display_stroke)
    n_rows = len(slices)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(7.2, max(2.25, 2.05 * n_rows)),
        facecolor="white",
        squeeze=False,
    )
    titles = ["DWI", "DWI + stroke core", "DWI + stroke core + atlas"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=9.5, weight="bold", pad=6)

    for row, slice_index in enumerate(slices):
        dwi_slice = dwi[:, :, slice_index]
        stroke_slice = display_stroke[:, :, slice_index]
        atlas_slice = atlas[:, :, slice_index]
        window = contrast_window(dwi_slice)
        draw_panel(axes[row, 0], dwi_slice, None, None, window)
        draw_panel(axes[row, 1], dwi_slice, stroke_slice, None, window)
        draw_panel(axes[row, 2], dwi_slice, stroke_slice, atlas_slice, window)
        axes[row, 0].set_ylabel(
            f"slice {slice_index}",
            fontsize=7.2,
            labelpad=5,
        )

    handles = [
        Line2D(
            [0], [0], color="white", lw=5, marker="s", markerfacecolor="#FF2D2D",
            markersize=7, label="Stroke core"
        )
    ]
    handles.extend(
        Line2D([0], [0], color=color, lw=1.8, label=name)
        for name, color in ATLAS_LABELS.values()
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.2,
        bbox_to_anchor=(0.52, 0.004),
        handlelength=1.4,
        columnspacing=1.1,
    )
    fig.suptitle(f"{subject} — {space_name}", fontsize=11, weight="bold", y=0.995)
    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.90 if n_rows <= 2 else 0.965,
        bottom=min(0.17, 0.72 / max(n_rows, 1)),
        wspace=0.015,
        hspace=0.035,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{subject}_dwi_stroke_pcaaspects.png"
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output


def render_subject(
    subject: str,
    indices: list[tuple[Path, dict[str, list[str]]]],
    args: argparse.Namespace,
) -> Path:
    cohort, entries = find_indexed_subject(subject, indices)
    dwi_rel = indexed_modality(entries, "DWI_MNI.nii.gz")
    stroke_rel = indexed_modality(entries, "stroke_MNI.nii.gz")

    mni_dwi_path = indexed_path(args.index_root, cohort, subject, dwi_rel)
    mni_stroke_path = indexed_path(args.index_root, cohort, subject, stroke_rel)
    raw_dwi_path = raw_path(args.raw_root, cohort, subject, dwi_rel)
    raw_stroke_path = raw_path(args.raw_root, cohort, subject, stroke_rel)

    raw_dwi = read_image(raw_dwi_path, sitk.sitkFloat32)
    atlas_mni = read_image(args.atlas, sitk.sitkUInt8)

    affine_path = raw_dwi_path.with_name(f"{subject}_DWI_affine.mat")
    if not affine_path.exists():
        raise FileNotFoundError(affine_path)
    mni_to_raw = sitk.ReadTransform(str(affine_path))
    raw_to_mni = mni_to_raw.GetInverse()
    atlas_raw = resample_labels(atlas_mni, raw_dwi, raw_to_mni)

    mni_dwi_array = load_canonical(mni_dwi_path)
    mni_stroke_array = load_canonical(mni_stroke_path)
    mni_atlas_array = load_canonical(args.atlas)
    raw_dwi_array = load_canonical(raw_dwi_path)
    raw_stroke_array = load_canonical(raw_stroke_path)
    raw_atlas_array = canonical_from_sitk(atlas_raw, raw_dwi_path)

    mni_output = render_space(
        subject,
        "MNI space",
        mni_dwi_array,
        mni_stroke_array,
        mni_atlas_array,
        args.output_dir / "mni",
    )
    raw_output = render_space(
        subject,
        "Raw space",
        raw_dwi_array,
        raw_stroke_array,
        raw_atlas_array,
        args.output_dir / "raw",
    )
    print(f"Wrote {mni_output}")
    print(f"Wrote {raw_output}")
    return mni_output


def render_manifest_subject(
    subject: str, row: dict[str, str], args: argparse.Namespace
) -> Path:
    if row.get("status") != "ready":
        raise FileNotFoundError(row.get("error") or "Manifest row is not ready")

    mni_dwi_path = Path(row["mni_dwi"])
    mni_stroke_path = Path(row["mni_stroke"])
    raw_dwi_path = Path(row["raw_dwi"])
    affine_path = Path(row["dwi_affine"])
    atlas_path = Path(row["atlas_mni"])

    raw_dwi_image = read_image(raw_dwi_path, sitk.sitkFloat32)
    mni_dwi_image = read_image(mni_dwi_path, sitk.sitkFloat32)
    atlas_mni_image = read_image(atlas_path, sitk.sitkUInt8)
    mni_to_raw = sitk.ReadTransform(str(affine_path))
    raw_to_mni = mni_to_raw.GetInverse()
    atlas_raw = resample_labels(atlas_mni_image, raw_dwi_image, raw_to_mni)
    raw_brainmask_path = raw_dwi_path.with_name(f"{row.get('dataset_subject') or subject}_DWI_brainmask.nii.gz")
    raw_support_array = None
    mni_support_array = None
    if raw_brainmask_path.exists():
        raw_brainmask_image = read_image(raw_brainmask_path, sitk.sitkUInt8)
        if not same_geometry(raw_brainmask_image, raw_dwi_image):
            raw_brainmask_image = resample_labels(raw_brainmask_image, raw_dwi_image)
        mni_brainmask_image = resample_labels(raw_brainmask_image, mni_dwi_image, mni_to_raw)
        raw_support_array = (
            native_from_sitk(raw_brainmask_image)
            if args.native_raw_orientation
            else canonical_from_sitk(raw_brainmask_image, raw_dwi_path)
        )
        mni_support_array = canonical_from_sitk(mni_brainmask_image, mni_dwi_path)
    if row.get("raw_stroke_mode", "native") == "native":
        raw_stroke_path = Path(row["raw_stroke"])
        raw_stroke_image = read_image(raw_stroke_path, sitk.sitkUInt8)
        if not same_geometry(raw_stroke_image, raw_dwi_image):
            raw_stroke_image = resample_labels(raw_stroke_image, raw_dwi_image)
            raw_stroke_array = (
                native_from_sitk(raw_stroke_image)
                if args.native_raw_orientation
                else canonical_from_sitk(raw_stroke_image, raw_dwi_path)
            )
        else:
            raw_stroke_array = (
                np.squeeze(nib.load(str(raw_stroke_path)).get_fdata())
                if args.native_raw_orientation
                else load_canonical(raw_stroke_path)
            )
    else:
        mni_stroke_image = read_image(mni_stroke_path, sitk.sitkUInt8)
        raw_stroke_image = resample_labels(mni_stroke_image, raw_dwi_image, raw_to_mni)
        raw_stroke_array = (
            native_from_sitk(raw_stroke_image)
            if args.native_raw_orientation
            else canonical_from_sitk(raw_stroke_image, raw_dwi_path)
        )

    outputs = []
    if args.space in ("both", "mni"):
        outputs.append(
            render_space(
                subject, "MNI space", load_canonical(mni_dwi_path),
                load_canonical(mni_stroke_path), load_canonical(atlas_path),
                args.output_dir / "mni", args.mask_stroke_to_dwi, mni_support_array, args.dpi,
            )
        )
    if args.space in ("both", "raw"):
        if args.isotropic_raw:
            raw_dwi_array, raw_stroke_array, raw_atlas_array = isotropic_raw_arrays(
                raw_dwi_path, raw_stroke_image, atlas_raw
            )
            raw_support_for_plot = None
            if raw_support_array is not None:
                raw_brainmask_image = read_image(raw_brainmask_path, sitk.sitkUInt8)
                if not same_geometry(raw_brainmask_image, raw_dwi_image):
                    raw_brainmask_image = resample_labels(raw_brainmask_image, raw_dwi_image)
                _, raw_support_for_plot, _ = isotropic_raw_arrays(
                    raw_dwi_path, raw_brainmask_image, atlas_raw
                )
        else:
            raw_dwi_array = (
                np.squeeze(nib.load(str(raw_dwi_path)).get_fdata())
                if args.native_raw_orientation
                else load_canonical(raw_dwi_path)
            )
            raw_atlas_array = (
                native_from_sitk(atlas_raw)
                if args.native_raw_orientation
                else canonical_from_sitk(atlas_raw, raw_dwi_path)
            )
            raw_support_for_plot = raw_support_array
        outputs.append(
            render_space(
                subject,
                "Raw space",
                raw_dwi_array,
                raw_stroke_array,
                raw_atlas_array,
                args.output_dir / "raw",
                args.mask_stroke_to_dwi,
                raw_support_for_plot,
                args.dpi,
            )
        )
    for output in outputs:
        print(f"Wrote {output}")
    return outputs[0]


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    args = parse_args()
    manifest = load_manifest(args.manifest)
    subjects = (
        list(manifest)
        if args.all and manifest
        else load_subjects(args.subject_csv)
        if args.all
        else [normalize_subject_id(args.subject)]
    )
    indices = [] if manifest else load_json_indices(args.index_root)
    failures = []
    for subject in subjects:
        try:
            if subject in manifest:
                render_manifest_subject(subject, manifest[subject], args)
            else:
                render_subject(subject, indices, args)
        except Exception as error:
            failures.append((subject, str(error)))
            print(f"FAILED {subject}: {error}")
    if failures:
        failure_path = args.output_dir / "visualization_failures.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subject", "error"])
            writer.writerows(failures)
        raise SystemExit(f"{len(failures)} subject(s) failed; see {failure_path}")


if __name__ == "__main__":
    main()
