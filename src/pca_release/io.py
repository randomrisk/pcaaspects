from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    missing = sorted(set(required_columns or []) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")
    return frame


def select_subjects(frame: pd.DataFrame, ids: pd.DataFrame, subject_column: str) -> pd.DataFrame:
    selected = frame[frame[subject_column].isin(set(ids[subject_column]))].copy()
    return selected.reset_index(drop=True)
