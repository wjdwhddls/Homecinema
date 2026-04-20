"""Simple-concat fusion variant (no gate).

Paired ablation against ``model_base/`` (Gated Weighted Concat) that ablates
the gate entirely: the 512-dim projected visual and audio vectors are simply
concatenated to a 1024-dim fused representation and passed to the VA / Mood
heads. No learned per-sample modality weighting, no gate-entropy loss term.

Isolates the question: "Does the learned 2-way softmax gate over (v, a)
contribute anything above naïve concatenation?" Every other component — PANNs
CNN14 audio, X-CLIP visual, modality dropout, feature noise, VA/Mood heads,
dataset, trainer, LOMO protocol — is identical to the baseline.
"""

from .config import TrainCogConfigConcat
from .model import AutoEQModelConcat

__all__ = ["AutoEQModelConcat", "TrainCogConfigConcat"]
