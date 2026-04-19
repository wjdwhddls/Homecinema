"""Baseline model — X-CLIP (video) + PANNs CNN14 (audio) + gated concat fusion.

Spec V3.3 §4-3. This is the configuration that produced the LOMO 9-fold
checkpoint at ``runs/phase3_v2_gemini_target/best_model.pt`` (mean CCC 0.4736).
Paired for comparison with ``model_beats/`` (audio encoder ablation).
"""

from .model import AutoEQModelCog

__all__ = ["AutoEQModelCog"]
