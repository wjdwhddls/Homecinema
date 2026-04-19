"""movie_set dispatch + LOMO splitter movie_list parameter + film_split_json_ids."""

import json

from model.autoEQ.train_pseudo.ccmovies_preprocess import CCMOVIES
from model.autoEQ.train_pseudo.dataset import (
    COGNIMUSE_MOVIES,
    film_split_json_ids,
    lomo_splits_with_time_val,
)


def _fake_meta(films: list[str], n_per: int = 5) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in films:
        for i in range(n_per):
            out[f"{f}_{i:04d}"] = {
                "movie_code": f,
                "valence": 0.0,
                "arousal": 0.0,
                "t0": i * 2.0,
                "t1": i * 2.0 + 4.0,
            }
    return out


def test_lomo_splitter_accepts_ccmovies_list():
    # 9-film CCMovies LOMO with fold index valid 0..8
    meta = _fake_meta(CCMOVIES, n_per=20)
    train, val, test = lomo_splits_with_time_val(
        meta, fold=3, val_tail_ratio=0.15, gap_windows=2, movie_list=CCMOVIES
    )
    test_film = CCMOVIES[3]
    # test set should all be from chosen film
    assert all(wid.startswith(test_film) for wid in test), \
        f"test ids should start with {test_film}"
    # train/val should not contain test film
    assert not any(wid.startswith(test_film) for wid in train + val)


def test_lomo_splitter_default_is_cognimuse_backward_compat():
    meta = _fake_meta(COGNIMUSE_MOVIES, n_per=20)
    # No movie_list arg → defaults to COGNIMUSE_MOVIES, fold 0..6 accepted
    train, val, test = lomo_splits_with_time_val(meta, fold=0)
    assert all(wid.startswith(COGNIMUSE_MOVIES[0]) for wid in test)


def test_lomo_splitter_rejects_out_of_range_fold_for_movie_list():
    meta = _fake_meta(CCMOVIES, n_per=20)
    try:
        lomo_splits_with_time_val(meta, fold=9, movie_list=CCMOVIES)
    except AssertionError:
        return
    assert False, "fold=9 with 9-film list should raise AssertionError"


def test_film_split_json_ids(tmp_path):
    split = {"train": ["spring"], "val": ["sintel"], "test": ["cosmos_laundromat"]}
    split_path = tmp_path / "film_split.json"
    split_path.write_text(json.dumps(split))

    meta = _fake_meta(["spring", "sintel", "cosmos_laundromat", "other_film"], n_per=3)
    train, val, test = film_split_json_ids(meta, str(split_path))
    assert all(w.startswith("spring") for w in train)
    assert all(w.startswith("sintel") for w in val)
    assert all(w.startswith("cosmos_laundromat") for w in test)
    # other_film should be dropped (not in any split)
    assert not any(w.startswith("other_film") for w in train + val + test)
