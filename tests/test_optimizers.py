import pytest
import torch

from lerobot.common.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.common.optim.optimizers import (
    AdamConfig,
    AdamW8bitConfig,
    AdamWConfig,
    AdamWNormConfig,
    SGDConfig,
    load_optimizer_state,
    save_optimizer_state,
)


@pytest.mark.parametrize(
    "config_cls, expected_class",
    [
        (AdamConfig, torch.optim.Adam),
        (AdamWNormConfig, torch.optim.AdamW),
        (SGDConfig, torch.optim.SGD),
    ],
)
def test_optimizer_build(config_cls, expected_class, model_params):
    config = config_cls()
    optimizer = config.build(model_params)
    assert isinstance(optimizer, expected_class)
    assert optimizer.defaults["lr"] == config.lr


@pytest.mark.parametrize("config_cls", [AdamW8bitConfig, AdamWConfig])
def test_8bit_adamw_build(config_cls, model_params):
    from bitsandbytes.optim import AdamW8bit

    config = config_cls()
    optimizer = config.build(model_params)

    assert isinstance(optimizer, AdamW8bit)
    assert optimizer.defaults["lr"] == config.lr
    assert optimizer.args.optim_bits == 8


def test_save_optimizer_state(optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    assert (tmp_path / OPTIMIZER_STATE).is_file()
    assert (tmp_path / OPTIMIZER_PARAM_GROUPS).is_file()


def test_save_and_load_optimizer_state(model_params, optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    loaded_optimizer = AdamConfig().build(model_params)
    loaded_optimizer = load_optimizer_state(loaded_optimizer, tmp_path)

    torch.testing.assert_close(optimizer.state_dict(), loaded_optimizer.state_dict())
