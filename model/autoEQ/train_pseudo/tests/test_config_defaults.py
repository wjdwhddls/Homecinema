"""Option C config defaults: K=4 mood head, patience=10, augmentation off."""

from model.autoEQ.train_pseudo.config import TrainCogConfig


def test_default_is_k4_for_ccmovies_phase0():
    cfg = TrainCogConfig()
    assert cfg.num_mood_classes == 4, (
        "Default num_mood_classes should be 4 (quadrant). "
        "Phase 0 gate fails at K=7 on CCMovies because Power GEMS region is empty. "
        "EQ lookup still uses 7-mood at inference via va_to_mood()."
    )


def test_default_patience_matches_spec_v32():
    cfg = TrainCogConfig()
    assert cfg.early_stop_patience == 10, (
        "spec V3.2 §4-5 requires patience=10 for val CCC noise tolerance."
    )


def test_augmentation_defaults_are_no_op():
    cfg = TrainCogConfig()
    # Backward compat: running without CLI overrides must match pre-patch behavior
    assert cfg.feature_noise_std == 0.0
    assert cfg.mixup_prob == 0.0
    assert cfg.label_smooth_eps == 0.0
    assert cfg.label_smooth_sigma_threshold == 0.0
