import math

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .config import TrainConfig


# --- V/A -> 7 Mood category mapping (Zentner GEMS + Eerola) ---

MOOD_CENTERS = torch.tensor(
    [
        [-0.6, +0.7],  # 0: Tension
        [-0.6, -0.4],  # 1: Sadness
        [+0.5, -0.5],  # 2: Peacefulness
        [+0.7, +0.6],  # 3: Joyful Activation
        [+0.4, -0.2],  # 4: Tenderness
        [+0.2, +0.8],  # 5: Power
        [+0.5, +0.3],  # 6: Wonder
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
