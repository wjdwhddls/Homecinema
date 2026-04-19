"""Schema mapping + source filter in ccmovies_preprocess (no encoder calls)."""

import pandas as pd

from model.autoEQ.train_pseudo.ccmovies_preprocess import (
    CCMOVIES,
    VALID_SOURCES,
)


def test_ccmovies_constant_is_alphabetical():
    assert CCMOVIES == sorted(CCMOVIES)
    assert len(CCMOVIES) == 9  # 8 Blender + Valkaama


def test_valid_sources_exclude_disagreement_and_excluded():
    assert VALID_SOURCES == {"auto_agreement", "gemini_only"}


def test_filter_removes_nan_rows(tmp_path):
    # Simulate final_labels.csv shape
    df = pd.DataFrame({
        "film_id": ["agent_327", "sintel", "sintel", "spring"],
        "window_id": ["a_0", "s_0", "s_1", "sp_0"],
        "t0": [0.0, 0.0, 4.0, 0.0],
        "t1": [4.0, 4.0, 8.0, 4.0],
        "split": ["train", "test_pseudo", "test_pseudo", "train"],
        "source": ["auto_agreement", "gemini_only", "disagreement", "excluded"],
        "final_v": [0.3, -0.1, float("nan"), float("nan")],
        "final_a": [0.1, 0.4, float("nan"), float("nan")],
        "ensemble_std_v": [0.05, 0.1, 0.3, 0.0],
        "ensemble_std_a": [0.05, 0.1, 0.3, 0.0],
    })
    kept = df[df["source"].isin(VALID_SOURCES)].dropna(subset=["final_v", "final_a"])
    assert len(kept) == 2  # only auto_agreement + gemini_only with non-NaN
    assert set(kept["window_id"]) == {"a_0", "s_0"}


def test_all_ccmovies_recognized_in_metadata_csv():
    # Spot-check that CCMOVIES matches the films actually present in metadata
    # (skipped if file missing — dev-only convenience)
    from pathlib import Path
    meta_csv = Path("dataset/autoEQ/CCMovies/windows/metadata.csv")
    if not meta_csv.is_file():
        return
    df = pd.read_csv(meta_csv)
    films = sorted(df["film_id"].unique())
    assert films == CCMOVIES, (
        f"CCMOVIES constant drifted from dataset:\n"
        f"  dataset: {films}\n  constant: {CCMOVIES}"
    )
