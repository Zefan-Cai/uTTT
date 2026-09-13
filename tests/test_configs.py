from copy import deepcopy
import json

import pytest
import yaml

from tools.check_configs import ROOT, inspect_configs, llm_model_types, validate_llm, validate_nvs


def test_entire_release_is_statically_valid():
    entries, errors = inspect_configs()
    assert not errors, "\n".join(errors)
    assert len(entries) == 124
    assert sum(path.suffix == ".yaml" for path, config in entries) == 64
    assert sum(path.suffix == ".json" for path, config in entries) == 57


@pytest.fixture
def nvs_config():
    return yaml.safe_load((ROOT / "uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml").read_text())


@pytest.mark.parametrize("key,value", [("num_experts", 0), ("num_active_experts", 65),
                                       ("pool_mode", "per_block"), ("fw_head_dim", 31)])
def test_nvs_invalid_routing_and_dimensions(nvs_config, key, value):
    config = deepcopy(nvs_config)
    config["model"]["block_config"]["params"][key] = value
    with pytest.raises(ValueError):
        validate_nvs(config)


@pytest.mark.parametrize("key,value", [("total_batch_size", 127), ("grad_accum_steps", 0),
                                       ("num_views", True), ("batch_size_per_gpu", -1)])
def test_nvs_invalid_training(nvs_config, key, value):
    nvs_config["training"][key] = value
    with pytest.raises(ValueError):
        validate_nvs(nvs_config)


def test_missing_source_class(nvs_config):
    nvs_config["model"]["class_name"] = "uttt_nvs.models.lvsm.DoesNotExist"
    with pytest.raises(ValueError, match="not defined"):
        validate_nvs(nvs_config)


@pytest.mark.parametrize("updates", [
    {"model_type": "made_up"}, {"num_heads": 11}, {"memory_num_active_experts": 49},
    {"memory_pool_mode": "per_block"}, {"chunk_size": 0},
])
def test_llm_invalid_config(updates):
    config = json.loads((ROOT / "uttt_llm/configs/main/124M/uttt_moe.json").read_text())
    config.update(updates)
    with pytest.raises(ValueError):
        validate_llm(config, llm_model_types(ROOT))
