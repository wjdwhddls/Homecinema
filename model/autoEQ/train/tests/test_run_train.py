"""run_train.py CLI 엔트리포인트 + PrecomputedFeatureDataset 테스트."""

import json
from pathlib import Path

import pytest
import torch

from ..config import TrainConfig
from ..dataset import PrecomputedFeatureDataset
from ..precompute import save_features
from ..run_train import (
    _build_parser,
    build_dataset,
    build_splits,
    main,
    save_run_artifacts,
    select_device,
    set_seed,
)


# ---------- PrecomputedFeatureDataset ----------


def _write_dummy_features(tmp_path: Path, split_name: str = "toy", n_clips: int = 6):
    visual, audio, metadata = {}, {}, {}
    for i in range(n_clips):
        wid = f"clip{i}_w0"
        visual[wid] = torch.randn(512)
        audio[wid] = torch.randn(2048)
        metadata[wid] = {
            "clip_id": f"clip{i}",
            "movie_id": i % 3,  # 3 movies
            "valence": 0.5 if i < n_clips // 2 else -0.5,
            "arousal": 0.5 if i % 2 == 0 else -0.5,
            "start": 0.0,
            "end": 4.0,
        }
    save_features(visual, audio, metadata, tmp_path, split_name=split_name)


def test_precomputed_dataset_loads(tmp_path):
    _write_dummy_features(tmp_path)
    ds = PrecomputedFeatureDataset(str(tmp_path), split_name="toy")
    assert len(ds) == 6
    item = ds[0]
    assert item["visual_feat"].shape == (512,)
    assert item["audio_feat"].shape == (2048,)
    assert item["mood"].dtype == torch.long
    assert item["valence"].dtype == torch.float32
    assert isinstance(item["movie_id"], int)


def test_precomputed_dataset_per_movie_va(tmp_path):
    _write_dummy_features(tmp_path)
    ds = PrecomputedFeatureDataset(str(tmp_path), split_name="toy")
    mv = ds.compute_per_movie_va()
    assert set(mv.keys()) == {0, 1, 2}
    for v, a in mv.values():
        assert -1.0 <= v <= 1.0
        assert -1.0 <= a <= 1.0


# ---------- run_train primitives ----------


def test_set_seed_deterministic():
    set_seed(7)
    a = torch.rand(5)
    set_seed(7)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_select_device_cpu():
    assert select_device("cpu") == torch.device("cpu")


def test_parser_help_smoke():
    parser = _build_parser()
    help_text = parser.format_help()
    assert "--use_synthetic" in help_text
    assert "--feature_dir" in help_text
    assert "--stratified" in help_text


def test_build_dataset_requires_source():
    parser = _build_parser()
    args = parser.parse_args([])
    cfg = TrainConfig()
    with pytest.raises(SystemExit):
        build_dataset(args, cfg)


def test_build_splits_synthetic_stratified():
    """Synthetic 경로에서 stratified split 이 작동하는지 (compute_movie_va 경로)."""
    parser = _build_parser()
    args = parser.parse_args(["--use_synthetic", "--stratified"])
    cfg = TrainConfig(batch_size=8)
    ds = build_dataset(args, cfg)
    train, val, test = build_splits(args, ds)
    assert train & val == set()
    assert train & test == set()


def test_save_run_artifacts_writes_files(tmp_path):
    cfg = TrainConfig(batch_size=8, epochs=1, warmup_steps=1)
    from ..dataset import SyntheticAutoEQDataset, create_dataloaders, film_level_split
    from ..model import AutoEQModel
    from ..trainer import Trainer

    ds = SyntheticAutoEQDataset(num_clips=40, num_films=5, config=cfg, seed=0)
    tr_ids, va_ids, _ = film_level_split(ds.movie_ids)
    tr_loader, va_loader = create_dataloaders(ds, tr_ids, va_ids, cfg)
    model = AutoEQModel(cfg)
    trainer = Trainer(model, tr_loader, va_loader, cfg)
    trainer.fit(max_epochs=1)

    save_run_artifacts(trainer, tmp_path)
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "history.json").is_file()
    with (tmp_path / "history.json").open() as f:
        hist = json.load(f)
    assert isinstance(hist, list) and len(hist) == 1


# ---------- Full CLI smoke ----------


def test_cli_synthetic_end_to_end(tmp_path):
    argv = [
        "--use_synthetic",
        "--synthetic_num_clips", "40",
        "--synthetic_num_films", "5",
        "--epochs", "2",
        "--batch_size", "8",
        "--seed", "0",
        "--output_dir", str(tmp_path),
    ]
    result = main(argv)
    assert result["history_len"] == 2
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "history.json").is_file()


def test_cli_precomputed_feature_dir_end_to_end(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _write_dummy_features(features_dir, split_name="toy", n_clips=20)
    out_dir = tmp_path / "run"
    argv = [
        "--feature_dir", str(features_dir),
        "--split_name", "toy",
        "--epochs", "2",
        "--batch_size", "4",
        "--seed", "0",
        "--output_dir", str(out_dir),
    ]
    result = main(argv)
    assert result["history_len"] == 2
    assert (out_dir / "best_model.pt").is_file()
