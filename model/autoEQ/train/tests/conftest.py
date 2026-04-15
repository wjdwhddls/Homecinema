import pytest
import torch

from ..config import TrainConfig
from ..model import AutoEQModel
from ..dataset import SyntheticAutoEQDataset


@pytest.fixture
def config():
    return TrainConfig(batch_size=8, epochs=3, warmup_steps=2)


@pytest.fixture
def model(config):
    m = AutoEQModel(config)
    m.eval()
    return m


@pytest.fixture
def synthetic_dataset(config):
    return SyntheticAutoEQDataset(num_clips=120, num_films=10, config=config, seed=42)


@pytest.fixture
def synthetic_batch(synthetic_dataset, config):
    """A single batch of 8 samples from the synthetic dataset."""
    loader = torch.utils.data.DataLoader(
        synthetic_dataset, batch_size=config.batch_size, shuffle=False
    )
    return next(iter(loader))


@pytest.fixture
def device():
    return torch.device("cpu")
