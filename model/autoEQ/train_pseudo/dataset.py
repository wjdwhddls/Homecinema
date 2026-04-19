from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from ..train.dataset import va_to_mood  # 재사용 (7-class)
from .config import TrainCogConfig


# ---- Permanent movie-id mapping (determinism) ------------------------------
COGNIMUSE_MOVIES = ["BMI", "CHI", "CRA", "DEP", "FNE", "GLA", "LOR"]
assert COGNIMUSE_MOVIES == sorted(COGNIMUSE_MOVIES), (
    "COGNIMUSE_MOVIES must stay in alphabetical order to keep movie_id stable"
)


# ---- 4-quadrant mood centers (used only if Phase 0 reduces K to 4) ---------
MOOD_CENTERS_4Q = torch.tensor(
    [
        [0.6, 0.6],   # 0: HVHA (Joyful Activation / Power)
        [0.6, -0.4],  # 1: HVLA (Peacefulness / Tenderness)
        [-0.6, 0.6],  # 2: LVHA (Tension)
        [-0.6, -0.4], # 3: LVLA (Sadness)
    ],
    dtype=torch.float32,
)


def va_to_quadrant(valence: float, arousal: float) -> int:
    """Map V/A to nearest 4-quadrant center index (Euclidean)."""
    va = torch.tensor([valence, arousal], dtype=torch.float32)
    d = torch.cdist(va.unsqueeze(0), MOOD_CENTERS_4Q).squeeze(0)
    return int(d.argmin().item())


# ---- Dataset ---------------------------------------------------------------


class PrecomputedCogDataset(Dataset):
    """Loads pre-extracted visual/audio features + CogniMuse metadata from
    the three .pt files produced by `cognimuse_preprocess.py`.

    Schema of each __getitem__ entry:
        visual_feat   (512,)
        audio_feat    (2048,)
        valence       scalar float
        arousal       scalar float
        mood          int (0..num_mood_classes-1)
        movie_id      int (0..6)
        valence_std   scalar float  (within-window std; 0 if absent)
        arousal_std   scalar float
    """

    def __init__(
        self,
        feature_dir: str,
        split_name: str = "cognimuse",
        num_mood_classes: int = 7,
    ):
        super().__init__()
        d = Path(feature_dir)
        self.visual: dict[str, Tensor] = torch.load(
            d / f"{split_name}_visual.pt", weights_only=False
        )
        self.audio: dict[str, Tensor] = torch.load(
            d / f"{split_name}_audio.pt", weights_only=False
        )
        self.metadata: dict[str, dict] = torch.load(
            d / f"{split_name}_metadata.pt", weights_only=False
        )
        self.window_ids: list[str] = sorted(self.metadata.keys())
        self.movie_ids: list[int] = [
            int(self.metadata[w]["movie_id"]) for w in self.window_ids
        ]
        self.num_mood_classes = num_mood_classes

    def __len__(self) -> int:
        return len(self.window_ids)

    def __getitem__(self, idx: int) -> dict:
        wid = self.window_ids[idx]
        m = self.metadata[wid]
        valence = float(m["valence"])
        arousal = float(m["arousal"])
        if self.num_mood_classes == 4:
            mood = va_to_quadrant(valence, arousal)
        else:
            mood = va_to_mood(valence, arousal)
        return {
            "visual_feat": self.visual[wid],
            "audio_feat": self.audio[wid],
            "valence": torch.tensor(valence, dtype=torch.float32),
            "arousal": torch.tensor(arousal, dtype=torch.float32),
            "mood": torch.tensor(mood, dtype=torch.long),
            "movie_id": self.movie_ids[idx],
            "valence_std": torch.tensor(
                float(m.get("valence_std", 0.0)), dtype=torch.float32
            ),
            "arousal_std": torch.tensor(
                float(m.get("arousal_std", 0.0)), dtype=torch.float32
            ),
        }


# ---- LOMO splitter with time-based val holdout + temporal gap --------------


def lomo_splits_with_time_val(
    metadata: dict[str, dict],
    fold: int,
    val_tail_ratio: float = 0.15,
    gap_windows: int = 2,
    movie_list: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Leave-One-Movie-Out with time-based within-movie val holdout.

    For fold k:
      - test = every window of movie_list[k] (full movie)
      - For the remaining movies, sort windows by t0 ascending, then:
          val   = last round(N * val_tail_ratio) windows
          train = first (N - N_val - gap_windows) windows
          The gap_windows in-between are DROPPED to prevent frame/audio
          leakage from overlapping windows.
          Time buffer = stride * (gap + 1) - window.
            gap=0 → -2s (2s overlap), gap=1 → 0s (adjacent, no overlap),
            gap=2 → +2s (default: overlap removed + 2s scene-continuity buffer),
            gap≥3 sacrifices train data for marginal extra buffer.

    Args:
        movie_list: ordered list of movie codes. Defaults to COGNIMUSE_MOVIES
            for backward compatibility. Pass CCMOVIES (9 films) for the
            pseudo-label dataset.

    Returns:
        (train_ids, val_ids, test_ids) — sorted lists of window_id strings.

    Raises:
        AssertionError if any non-test movie has N - N_val - gap_windows <= 0
        (movie too short for the configured split + gap).
    """
    movies = list(movie_list) if movie_list is not None else COGNIMUSE_MOVIES
    assert 0 <= fold < len(movies), f"fold must be 0..{len(movies)-1}"
    test_code = movies[fold]

    test_ids: list[str] = []
    train_ids: list[str] = []
    val_ids: list[str] = []

    # test: full movie
    for wid, m in metadata.items():
        if m["movie_code"] == test_code:
            test_ids.append(wid)

    # Remaining movies: per-movie time-tail split
    for code in movies:
        if code == test_code:
            continue
        movie_windows = sorted(
            [(wid, float(m["t0"])) for wid, m in metadata.items() if m["movie_code"] == code],
            key=lambda x: x[1],
        )
        N = len(movie_windows)
        N_val = round(N * val_tail_ratio)
        N_train = N - N_val - gap_windows
        assert N_train > 0, (
            f"Movie {code}: N={N}, N_val={N_val}, gap={gap_windows} "
            f"→ N_train={N_train} <= 0. Reduce val_tail_ratio or gap_windows."
        )
        train_ids.extend(w for w, _ in movie_windows[:N_train])
        # drop gap_windows between train-tail and val-head
        val_ids.extend(w for w, _ in movie_windows[N_train + gap_windows :])

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


# ---- Film-split JSON based splits (alternative to LOMO) --------------------


def film_split_json_ids(
    metadata: dict[str, dict],
    split_json_path: str,
    *,
    train_key: str = "train",
    val_key: str = "val",
    test_key: str = "test",
) -> tuple[list[str], list[str], list[str]]:
    """Load film_split.json {'train': [...], 'val': [...], 'test': [...]} and
    assign windows to splits by `movie_code` membership.

    Used when `lomo_fold == -1` (single-fold mode) for CCMovies training where
    LOMO 9-fold is overkill for initial experiments.
    """
    import json
    with open(split_json_path) as f:
        split = json.load(f)
    train_films = set(split[train_key])
    val_films = set(split[val_key])
    test_films = set(split[test_key])

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for wid, m in metadata.items():
        code = str(m["movie_code"])
        if code in train_films:
            train_ids.append(wid)
        elif code in val_films:
            val_ids.append(wid)
        elif code in test_films:
            test_ids.append(wid)
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


# ---- σ-filter (train split only) --------------------------------------------


def apply_sigma_filter(
    window_ids: list[str],
    metadata: dict[str, dict],
    threshold: float,
) -> list[str]:
    """Exclude windows whose max(valence_std, arousal_std) exceeds threshold.

    No-op when threshold <= 0. Intended for the TRAIN split only — val/test
    must keep original distribution to preserve evaluation validity.
    """
    if threshold <= 0:
        return window_ids
    out: list[str] = []
    for wid in window_ids:
        m = metadata[wid]
        sigma = max(
            float(m.get("valence_std", 0.0)),
            float(m.get("arousal_std", 0.0)),
        )
        if sigma <= threshold:
            out.append(wid)
    return out


# ---- Augmentation collate (quadrant-restricted mixup + conditional label smoothing) ----


def make_train_collate_fn(config: TrainCogConfig) -> Callable:
    """Return a collate_fn implementing:

    1. **Conditional label smoothing** — when `max(valence_std, arousal_std) >
       label_smooth_sigma_threshold`, multiply V/A by `(1 - label_smooth_eps)`.
       High-uncertainty windows are pulled toward 0; confident windows stay raw.
    2. **Quadrant-restricted Mixup** — with probability `mixup_prob`, mix each
       sample with a random same-quadrant partner in the batch using
       `λ ~ Beta(α, α)` shrunk to [0.1, 0.9] to avoid label-space extremes.
       Visual/audio features + V/A + std are all interpolated; `mood` uses the
       primary sample's class (same-quadrant guarantees coarse coherence but
       not identical 7-GEMS class).

    Backward compatible: when `mixup_prob == 0` and `label_smooth_eps == 0`,
    behaves as `torch.utils.data._utils.collate.default_collate`.
    """
    eps = config.label_smooth_eps
    sigma_thr = config.label_smooth_sigma_threshold
    mix_p = config.mixup_prob
    alpha = config.mixup_alpha
    smooth_enabled = eps > 0.0 and sigma_thr > 0.0
    mix_enabled = mix_p > 0.0

    def _collate(batch: list[dict]) -> dict:
        if smooth_enabled:
            out: list[dict] = []
            for s in batch:
                vsd = float(s["valence_std"])
                asd = float(s["arousal_std"])
                if max(vsd, asd) > sigma_thr:
                    s = {**s,
                         "valence": s["valence"] * (1.0 - eps),
                         "arousal": s["arousal"] * (1.0 - eps)}
                out.append(s)
            batch = out

        if mix_enabled and len(batch) >= 2:
            quads = [va_to_quadrant(float(s["valence"]), float(s["arousal"]))
                     for s in batch]
            by_q: dict[int, list[int]] = {}
            for i, q in enumerate(quads):
                by_q.setdefault(q, []).append(i)
            mixed: list[dict] = []
            for i, s in enumerate(batch):
                partners = [j for j in by_q[quads[i]] if j != i]
                if not partners or torch.rand(1).item() >= mix_p:
                    mixed.append(s)
                    continue
                j = partners[int(torch.randint(0, len(partners), (1,)).item())]
                t = batch[j]
                lam_raw = float(torch.distributions.Beta(alpha, alpha).sample())
                lam = 0.1 + 0.8 * lam_raw  # shrink to [0.1, 0.9]
                inv = 1.0 - lam
                new = {
                    "visual_feat": s["visual_feat"] * lam + t["visual_feat"] * inv,
                    "audio_feat": s["audio_feat"] * lam + t["audio_feat"] * inv,
                    "valence": s["valence"] * lam + t["valence"] * inv,
                    "arousal": s["arousal"] * lam + t["arousal"] * inv,
                    "mood": s["mood"],
                    "movie_id": s["movie_id"],
                    "valence_std": s["valence_std"] * lam + t["valence_std"] * inv,
                    "arousal_std": s["arousal_std"] * lam + t["arousal_std"] * inv,
                }
                mixed.append(new)
            batch = mixed

        return default_collate(batch)

    return _collate


# ---- DataLoader factory ----------------------------------------------------


def create_dataloaders_cog(
    dataset: PrecomputedCogDataset,
    train_ids: list[str],
    val_ids: list[str],
    config: TrainCogConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoader from window_id lists.

    Train loader applies augmentation collate (mixup + label smoothing) when
    config flags are non-zero. Val loader uses default_collate to keep
    evaluation labels unmodified.
    """
    wid_to_idx = {w: i for i, w in enumerate(dataset.window_ids)}
    train_indices = [wid_to_idx[w] for w in train_ids if w in wid_to_idx]
    val_indices = [wid_to_idx[w] for w in val_ids if w in wid_to_idx]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=make_train_collate_fn(config),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader
