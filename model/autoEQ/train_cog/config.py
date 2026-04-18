from dataclasses import dataclass


@dataclass
class TrainCogConfig:
    """CogniMuse-only training config. Independent dataclass (no inheritance
    from train.TrainConfig) — cong-related fields never exist here.
    """

    # --- Feature dimensions ---
    visual_dim: int = 512
    audio_raw_dim: int = 2048
    audio_proj_dim: int = 512
    fused_dim: int = 1024

    # --- Task heads ---
    num_mood_classes: int = 7  # Phase 0 may reduce to 4 (quadrant)
    gate_hidden_dim: int = 256
    head_hidden_dim: int = 256

    # --- Optimizer ---
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 30
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    # --- Loss weights (cong term absent by construction) ---
    lambda_va: float = 1.0
    lambda_mood: float = 0.5
    lambda_gate_entropy: float = 0.05

    # --- CCC loss (hybrid V/A loss) ---
    use_ccc_loss: bool = True
    ccc_loss_weight: float = 0.3  # L_va = (1-w)*MSE + w*(1-CCC)

    # --- Modality dropout (applied to all samples; p=0.05 matches prior
    # effective rate since old code only applied to congruent 50% × 0.1) ---
    modality_dropout_p: float = 0.05

    # --- Early stopping ---
    early_stop_patience: int = 5

    # --- Input specs ---
    num_frames: int = 8
    frame_size: int = 224
    audio_sr: int = 16000
    audio_sec: int = 4

    # --- Encoder paths (compatible with train.encoders) ---
    xclip_model: str = "microsoft/xclip-base-patch32"
    panns_checkpoint: str = ""

    # --- CogniMuse-specific ---
    cognimuse_dir: str = ""
    cognimuse_annotation: str = "experienced"  # experienced | intended | mean
    cognimuse_window_sec: int = 4
    cognimuse_stride_sec: int = 2

    # --- LOMO ---
    lomo_fold: int = -1  # -1 = unset, 0..6 = fold index
    val_tail_ratio: float = 0.15
    val_gap_windows: int = 2  # drops gap_windows between train-tail and val-head

    # --- σ filter (train split only; negative = disabled) ---
    sigma_filter_threshold: float = -1.0

    # --- Feature dir (precomputed .pt path) ---
    feature_dir: str = ""

    # --- Wandb logging ---
    use_wandb: bool = False
    wandb_project: str = "moodeq_cog"
    wandb_run_name: str = ""

    @property
    def audio_samples(self) -> int:
        return self.audio_sr * self.audio_sec
