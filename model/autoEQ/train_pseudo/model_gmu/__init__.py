"""GMU (Gated Multimodal Unit, Arevalo 2017) fusion variant.

Paired ablation against ``model_base/`` (Gated Weighted Concat). GMU differs
from the baseline in two structural ways:

  1. **Element-wise gating**: the gate ``z`` is a sigmoid vector of the same
     dimensionality as the fused representation, so each feature dimension
     independently mixes between the two modalities — far finer-grained than
     the baseline's scalar 2-way softmax.
  2. **Single-vector fusion**: output is ``z ⊙ h_v + (1−z) ⊙ h_a`` (a single
     512-dim vector), not a 1024-dim concatenation. VA/Mood heads therefore
     consume ``fused_dim = 512`` (overridden in the config).

Formulation (Arevalo et al., ICLR 2017 §3):
    h_v = tanh(W_v · v)
    h_a = tanh(W_a · a)
    z   = σ(W_z · [v; a])       # element-wise sigmoid over concat
    fused = z ⊙ h_v + (1 − z) ⊙ h_a

Isolates the question: "Does fine-grained dim-wise gating outperform scalar
2-way gating for movie V/A regression?"
"""

from .config import TrainCogConfigGMU
from .model import AutoEQModelGMU

__all__ = ["AutoEQModelGMU", "TrainCogConfigGMU"]
