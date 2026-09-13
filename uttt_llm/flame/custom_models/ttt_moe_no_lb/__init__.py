from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_lact import (
    RecurrentLactRefConfig,
    RecurrentLactRefMoeV3Config,
)
from .modeling_recurrent_lact import (
    RecurrentLactRefForCausalLM,
    RecurrentLactRefModel,
    RecurrentLactRefMoeV3ForCausalLM,
    RecurrentLactRefMoeV3Model,
)

AutoConfig.register(
    RecurrentLactRefMoeV3Config.model_type,
    RecurrentLactRefMoeV3Config,
    exist_ok=True,
)
AutoModel.register(RecurrentLactRefMoeV3Config, RecurrentLactRefMoeV3Model, exist_ok=True)
AutoModelForCausalLM.register(
    RecurrentLactRefMoeV3Config,
    RecurrentLactRefMoeV3ForCausalLM,
    exist_ok=True,
)

__all__ = [
    "RecurrentLactRefConfig",
    "RecurrentLactRefForCausalLM",
    "RecurrentLactRefModel",
    "RecurrentLactRefMoeV3Config",
    "RecurrentLactRefMoeV3ForCausalLM",
    "RecurrentLactRefMoeV3Model",
]
