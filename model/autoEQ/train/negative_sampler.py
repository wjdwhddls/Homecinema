import torch
from torch import Tensor

from .config import TrainConfig


class NegativeSampler:
    """Self-supervised negative sampling for congruence learning.

    Modifies audio features in a batch to create incongruent pairs:
    - 50% congruent (original pair, cong_label=0)
    - 25% slight incongruent (swap with V/A-nearby clip, cong_label=1)
    - 25% strong incongruent (swap with V/A-distant clip, cong_label=2)

    Audio is swapped from *different films* to avoid same-film style similarity.
    """

    def __init__(self, config: TrainConfig | None = None, seed: int | None = None):
        self.config = config or TrainConfig()
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample(
        self,
        audio_feat: Tensor,
        va_target: Tensor,
        movie_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply negative sampling to a batch.

        Args:
            audio_feat: (B, D) audio features
            va_target: (B, 2) valence/arousal targets
            movie_ids: (B,) movie IDs per sample

        Returns:
            (modified_audio_feat, cong_labels) where cong_labels is (B,) long
        """
        B = audio_feat.size(0)
        audio_out = audio_feat.clone()
        cong_labels = torch.zeros(B, dtype=torch.long, device=audio_feat.device)

        # Determine number of samples per class
        n_cong = round(B * self.config.neg_congruent_ratio)
        n_slight = round(B * self.config.neg_slight_ratio)
        # strong gets the rest
        n_strong = B - n_cong - n_slight

        # Random permutation to assign roles
        perm = torch.randperm(B, generator=self.rng)
        congruent_idx = perm[:n_cong]
        slight_idx = perm[n_cong : n_cong + n_slight]
        strong_idx = perm[n_cong + n_slight :]

        # Compute pairwise V/A distances for swap candidate selection
        va_dists = torch.cdist(va_target.float(), va_target.float())  # (B, B)

        # For slight: swap with nearby V/A (25-50 percentile of distances)
        for i in slight_idx:
            i_val = i.item()
            candidates = self._get_diff_film_candidates(
                i_val, movie_ids, B
            )
            if len(candidates) == 0:
                # No cross-film candidate: keep original audio, label as congruent
                cong_labels[i_val] = 0
                continue
            dists = va_dists[i_val, candidates]
            # Select from 25th-50th percentile range
            sorted_dists, sorted_indices = dists.sort()
            n_cands = len(candidates)
            lo = max(0, n_cands // 4)
            hi = max(lo + 1, n_cands // 2)
            pick = torch.randint(lo, hi, (1,), generator=self.rng).item()
            swap_idx = candidates[sorted_indices[pick].item()]
            audio_out[i_val] = audio_feat[swap_idx]
            cong_labels[i_val] = 1

        # For strong: swap with distant V/A (>= 75 percentile)
        for i in strong_idx:
            i_val = i.item()
            candidates = self._get_diff_film_candidates(
                i_val, movie_ids, B
            )
            if len(candidates) == 0:
                # No cross-film candidate: keep original audio, label as congruent
                cong_labels[i_val] = 0
                continue
            dists = va_dists[i_val, candidates]
            sorted_dists, sorted_indices = dists.sort()
            n_cands = len(candidates)
            lo = max(0, (3 * n_cands) // 4)
            hi = n_cands
            pick = torch.randint(lo, hi, (1,), generator=self.rng).item()
            swap_idx = candidates[sorted_indices[pick].item()]
            audio_out[i_val] = audio_feat[swap_idx]
            cong_labels[i_val] = 2

        return audio_out, cong_labels

    @staticmethod
    def _get_diff_film_candidates(
        idx: int, movie_ids: Tensor, batch_size: int
    ) -> list[int]:
        """Get indices of samples from different films."""
        my_film = movie_ids[idx].item()
        return [j for j in range(batch_size) if j != idx and movie_ids[j].item() != my_film]
