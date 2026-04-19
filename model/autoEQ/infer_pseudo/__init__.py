"""MoodEQ inference pipeline for CCMovies-trained models.

Consumes a best_model.pt checkpoint (from model.autoEQ.train_pseudo) and
produces a timeline.json describing per-scene EQ presets.

Pipeline stages (parallel where possible per spec V3.3 §5-0):

    [video input]
         ├─ scene_detect → window_slider → model_inference → ema_smoother
         └─ vad (parallel)
              └───────────────────────────┐
                                          ▼
                                  dialogue_density
                                          │
                                  mood_mapper + eq_preset
                                          │
                                  timeline_writer → JSON
"""

from .types import (
    Scene,
    Window,
    WindowVA,
    SceneVA,
    SpeechSegment,
    SceneEQ,
)
