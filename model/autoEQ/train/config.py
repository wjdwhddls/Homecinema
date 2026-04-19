from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- Feature dimensions ---
    visual_dim: int = 512
    audio_raw_dim: int = 2048
    audio_proj_dim: int = 512
    fused_dim: int = 1024

    # --- Task heads ---
    num_mood_classes: int = 7
    num_cong_classes: int = 3
    gate_hidden_dim: int = 256
    head_hidden_dim: int = 256
    cong_head_input_dim: int = 514  # 512 (L2-norm |v-a|) + 2 (gate weights)
    cong_head_hidden_dim: int = 128

    # --- Optimizer ---
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 50
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    # --- Loss weights ---
    lambda_va: float = 1.0
    lambda_mood: float = 0.5
    lambda_cong: float = 0.5
    lambda_gate_entropy: float = 0.05

    # --- CCC loss (hybrid V/A loss) ---
    use_ccc_loss: bool = True
    ccc_loss_weight: float = 0.3  # L_va = (1-w)*MSE + w*(1-CCC)

    # --- Modality dropout ---
    modality_dropout_p: float = 0.1

    # --- Negative sampling ratios ---
    neg_congruent_ratio: float = 0.50
    neg_slight_ratio: float = 0.25
    neg_strong_ratio: float = 0.25

    # --- Early stopping ---
    early_stop_patience: int = 10

    # --- Input specs ---
    num_frames: int = 8
    frame_size: int = 224
    audio_sr: int = 16000
    audio_sec: int = 4

    # --- Congruence noise ---
    cong_noise_std: float = 0.05

    # --- Encoder paths ---
    xclip_model: str = "microsoft/xclip-base-patch32"
    panns_checkpoint: str = ""

    # --- Dataset paths ---
    data_dir: str = ""
    feature_dir: str = ""
    cognimuse_dir: str = ""

    # --- Wandb logging ---
    use_wandb: bool = False
    wandb_project: str = "moodeq"
    wandb_run_name: str = ""

    @property
    def audio_samples(self) -> int:
        return self.audio_sr * self.audio_sec
