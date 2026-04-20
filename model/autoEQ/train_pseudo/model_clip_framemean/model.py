"""CLIP frame-mean ablation model — thin wrapper around AutoEQModelCog.

Because the CLIP image encoder's frame-mean output is 512-dim (identical to
X-CLIP's video embedding dim), no architectural change is needed: the
baseline fused tower (AudioProjection, GateNetworkCog, VAHeadCog, MoodHeadCog,
modality dropout, feature noise) works verbatim. We subclass only for
dispatch / type clarity in ``run_train.py`` / ``run_lomo.py`` factories and
so future CLIP-specific tweaks (e.g. attention pool over per-frame tokens
instead of uniform mean) can be layered in without touching the baseline.
"""

from __future__ import annotations

from ..model_base.model import AutoEQModelCog


class AutoEQModelClipFrameMean(AutoEQModelCog):
    """Same architecture as AutoEQModelCog.

    Expected inputs (unchanged from baseline):
        visual_feat: (B, 512)  — mean of CLIP ViT-B/32 per-frame pooler_output
                                 over the 8 window frames
        audio_feat:  (B, 2048) — frozen PANNs CNN14 (unchanged)
    """


__all__ = ["AutoEQModelClipFrameMean"]
