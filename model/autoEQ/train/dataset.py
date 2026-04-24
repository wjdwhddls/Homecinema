import math

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .config import TrainConfig


# --- V/A -> 7 Mood category mapping (Zentner GEMS + Eerola) ---

# 2026-04-24 FINAL-A calibration (post-hoc centroid re-placement).
# Fit: TRAIN-fit empirical prototypes from LIRIS train (4315 clips, 64 films,
# 3-seed ensemble predictions). Sadness/Tenderness use 0.5·ORIG + 0.5·EMP to
# retain recall; JoyfulActivation keeps ORIG (0 GT samples in LIRIS learning
# set). See BASE_MODEL.md "Centroid Calibration (2026-04-24)" for full
# derivation and the original values (needed to revert).
MOOD_CENTERS = torch.tensor(
    [
        [-0.3730, +0.1004],  # 0: Tension          (empirical)
        [-0.4223, -0.3225],  # 1: Sadness          (0.5·ORIG + 0.5·EMP)
        [+0.0376, -0.4225],  # 2: Peacefulness     (empirical)
        [+0.7000, +0.6000],  # 3: JoyfulActivation (ORIG — no GT data)
        [+0.1813, -0.2091],  # 4: Tenderness       (0.5·ORIG + 0.5·EMP)
        [-0.1073, +0.0906],  # 5: Power            (empirical)
        [-0.0135, -0.0855],  # 6: Wonder           (empirical)
    ],
    dtype=torch.float32,
)


def va_to_mood(valence: float, arousal: float) -> int:
    """Map V/A coordinates to nearest mood category (Euclidean distance)."""
    va = torch.tensor([valence, arousal], dtype=torch.float32)
    distances = torch.cdist(va.unsqueeze(0), MOOD_CENTERS).squeeze(0)
    return distances.argmin().item()


# --- Film-level split ---


def film_level_split(
    movie_ids: list[int],
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    test_ratio: float = 0.125,
    seed: int = 42,
) -> tuple[set[int], set[int], set[int]]:
    """Split unique movie IDs into train/val/test sets.

    Returns:
        (train_ids, val_ids, test_ids) as sets of movie IDs.
    """
    unique_ids = sorted(set(movie_ids))
    n = len(unique_ids)

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    shuffled = [unique_ids[i] for i in perm]

    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    # test gets the rest

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])

    return train_ids, val_ids, test_ids


def _va_quadrant(valence: float, arousal: float) -> str:
    """HVHA / HVLA / LVHA / LVLA (sign-based 4-quadrant label)."""
    v = "H" if valence >= 0 else "L"
    a = "H" if arousal >= 0 else "L"
    return f"{v}V{a}A"


def stratified_film_level_split(
    movie_va: dict[int, tuple[float, float]],
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    test_ratio: float = 0.125,
    seed: int = 42,
) -> tuple[set[int], set[int], set[int]]:
    """Film-level split stratified by the film's mean V/A quadrant (spec 2-3).

    Each movie is placed in one of 4 quadrants (HVHA/HVLA/LVHA/LVLA) by the
    sign of its mean valence/arousal, then split independently within each
    quadrant. Guarantees film-level disjoint train/val/test and balanced
    emotion-quadrant coverage in every split.

    Args:
        movie_va: {movie_id: (mean_valence, mean_arousal)}
    """
    del test_ratio  # test gets the remainder; kept for signature symmetry

    quadrants: dict[str, list[int]] = {"HVHA": [], "HVLA": [], "LVHA": [], "LVLA": []}
    for mid in sorted(movie_va):
        v, a = movie_va[mid]
        quadrants[_va_quadrant(v, a)].append(mid)

    train_ids: set[int] = set()
    val_ids: set[int] = set()
    test_ids: set[int] = set()

    for i, (_, mids) in enumerate(sorted(quadrants.items())):
        if not mids:
            continue
        rng = torch.Generator().manual_seed(seed + i)
        perm = torch.randperm(len(mids), generator=rng).tolist()
        shuffled = [mids[j] for j in perm]
        n = len(shuffled)

        # Always guarantee at least 1 train; reserve remainder for val/test.
        if n == 1:
            n_train, n_val = 1, 0
        else:
            n_train = max(1, min(n - 1, round(n * train_ratio)))
            remaining = n - n_train
            if remaining <= 1:
                n_val = remaining
            else:
                n_val = max(1, min(remaining - 1, round(n * val_ratio)))
        train_ids.update(shuffled[:n_train])
        val_ids.update(shuffled[n_train : n_train + n_val])
        test_ids.update(shuffled[n_train + n_val :])

    return train_ids, val_ids, test_ids


def compute_movie_va(
    movie_ids: list[int],
    valences: list[float] | Tensor,
    arousals: list[float] | Tensor,
) -> dict[int, tuple[float, float]]:
    """Aggregate per-sample V/A into per-movie mean V/A."""
    if isinstance(valences, Tensor):
        valences = valences.tolist()
    if isinstance(arousals, Tensor):
        arousals = arousals.tolist()
    sums: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for mid, v, a in zip(movie_ids, valences, arousals):
        if mid not in sums:
            sums[mid] = [0.0, 0.0]
            counts[mid] = 0
        sums[mid][0] += float(v)
        sums[mid][1] += float(a)
        counts[mid] += 1
    return {mid: (sums[mid][0] / counts[mid], sums[mid][1] / counts[mid]) for mid in sums}


# --- Synthetic dataset ---


class SyntheticAutoEQDataset(Dataset):
    """Generates synthetic data mimicking LIRIS-ACCEDE structure.

    Each sample provides pre-computed feature vectors (not raw video/audio)
    since frozen encoders are deterministic.
    """

    def __init__(
        self,
        num_clips: int = 120,
        num_films: int = 10,
        config: TrainConfig | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.config = config or TrainConfig()
        self.num_clips = num_clips
        self.num_films = num_films

        rng = torch.Generator().manual_seed(seed)
        clips_per_film = num_clips // num_films

        self.movie_ids: list[int] = []
        for film_id in range(num_films):
            self.movie_ids.extend([film_id] * clips_per_film)

        # Pre-generate all features and labels for reproducibility
        self.visual_feats = torch.randn(
            num_clips, self.config.visual_dim, generator=rng
        )
        self.audio_feats = torch.randn(
            num_clips, self.config.audio_raw_dim, generator=rng
        )
        self.valences = torch.rand(num_clips, generator=rng) * 2 - 1  # [-1, 1]
        self.arousals = torch.rand(num_clips, generator=rng) * 2 - 1  # [-1, 1]

        # Mood labels derived from V/A (same as spec)
        self.moods = torch.tensor(
            [
                va_to_mood(self.valences[i].item(), self.arousals[i].item())
                for i in range(num_clips)
            ],
            dtype=torch.long,
        )

        # Congruence labels: initially all congruent (negative sampler changes this)
        self.cong_labels = torch.zeros(num_clips, dtype=torch.long)

        # Raw audio waveform for audio alignment tests (4 sec @ 16kHz)
        self.audio_waveforms = torch.randn(
            num_clips, 1, self.config.audio_samples, generator=rng
        )

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> dict[str, Tensor | int]:
        return {
            "visual_feat": self.visual_feats[idx],
            "audio_feat": self.audio_feats[idx],
            "valence": self.valences[idx],
            "arousal": self.arousals[idx],
            "mood": self.moods[idx],
            "cong_label": self.cong_labels[idx],
            "movie_id": self.movie_ids[idx],
            "audio_waveform": self.audio_waveforms[idx],
        }


def create_dataloaders(
    dataset: Dataset,
    train_ids: set[int],
    val_ids: set[int],
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders from a dataset using film-level split."""
    train_indices = [
        i for i, mid in enumerate(dataset.movie_ids) if mid in train_ids
    ]
    val_indices = [
        i for i, mid in enumerate(dataset.movie_ids) if mid in val_ids
    ]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


class PrecomputedFeatureDataset(Dataset):
    """Loads pre-extracted visual/audio features from .pt files produced by
    `precompute.py`. Serves items in the same dict schema as
    SyntheticAutoEQDataset (minus `audio_waveform`, which only synthetic data
    needs for alignment tests).
    """

    def __init__(self, feature_dir: str, split_name: str):
        super().__init__()
        from pathlib import Path as _P

        d = _P(feature_dir)
        self.visual: dict[str, Tensor] = torch.load(d / f"{split_name}_visual.pt", weights_only=False)
        self.audio: dict[str, Tensor] = torch.load(d / f"{split_name}_audio.pt", weights_only=False)
        self.metadata: dict[str, dict] = torch.load(d / f"{split_name}_metadata.pt", weights_only=False)
        self.window_ids: list[str] = sorted(self.metadata.keys())
        self.movie_ids: list[int] = [int(self.metadata[w]["movie_id"]) for w in self.window_ids]

    def __len__(self) -> int:
        return len(self.window_ids)

    def __getitem__(self, idx: int) -> dict:
        wid = self.window_ids[idx]
        m = self.metadata[wid]
        valence = float(m["valence"])
        arousal = float(m["arousal"])
        return {
            "visual_feat": self.visual[wid],
            "audio_feat": self.audio[wid],
            "valence": torch.tensor(valence, dtype=torch.float32),
            "arousal": torch.tensor(arousal, dtype=torch.float32),
            "mood": torch.tensor(va_to_mood(valence, arousal), dtype=torch.long),
            "cong_label": torch.tensor(0, dtype=torch.long),
            "movie_id": self.movie_ids[idx],
        }

    def compute_per_movie_va(self) -> dict[int, tuple[float, float]]:
        valences = [float(self.metadata[w]["valence"]) for w in self.window_ids]
        arousals = [float(self.metadata[w]["arousal"]) for w in self.window_ids]
        return compute_movie_va(self.movie_ids, valences, arousals)


def align_audio(waveform: Tensor, target_samples: int = 64000) -> Tensor:
    """Pad or truncate audio waveform to exact target length.

    Args:
        waveform: (1, T) or (T,) audio tensor
        target_samples: target number of samples (default: 4s @ 16kHz)

    Returns:
        (1, target_samples) aligned audio tensor
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    current = waveform.size(-1)
    if current == target_samples:
        return waveform
    elif current > target_samples:
        return waveform[:, :target_samples]
    else:
        padding = torch.zeros(1, target_samples - current, device=waveform.device)
        return torch.cat([waveform, padding], dim=-1)
