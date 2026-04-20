"""GMU ablation config — subclasses TrainCogConfig, overrides fusion dims.

GMU produces a single fused vector (not a concatenation), so ``fused_dim``
shrinks from 1024 to ``gmu_hidden_dim``. The gate-entropy loss term is also
zeroed: GMU's ``z`` is an element-wise sigmoid, not a categorical distribution
over two modalities, so the entropy regulariser used by the baseline's scalar
softmax gate does not apply cleanly.
"""

from dataclasses import dataclass

from ..config import TrainCogConfig


@dataclass
class TrainCogConfigGMU(TrainCogConfig):
    # GMU emits a single fused vector at hidden_dim, replacing the concatenated
    # 1024-dim baseline. VA / Mood heads will therefore be built with
    # ``fused_dim`` as the input size (config → head layer stack consumes it).
    gmu_hidden_dim: int = 512
    fused_dim: int = 512   # = gmu_hidden_dim

    # Categorical-gate entropy term doesn't apply to dim-wise sigmoid gating.
    lambda_gate_entropy: float = 0.0

    # Wandb — distinguish the ablation tracks by project name if enabled.
    wandb_project: str = "moodeq_cog_gmu"
