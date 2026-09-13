from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_lact import (
    RecurrentLactRefConfig,
    RecurrentLactRefMoeGlobalV45Config,
)
from .modeling_recurrent_lact import (
    RecurrentLactRefForCausalLM,
    RecurrentLactRefModel,
    RecurrentLactRefMoeGlobalV45ForCausalLM,
    RecurrentLactRefMoeGlobalV45Model,
)

AutoConfig.register(
    RecurrentLactRefMoeGlobalV45Config.model_type,
    RecurrentLactRefMoeGlobalV45Config,
    exist_ok=True,
)
AutoModel.register(RecurrentLactRefMoeGlobalV45Config, RecurrentLactRefMoeGlobalV45Model, exist_ok=True)
AutoModelForCausalLM.register(
    RecurrentLactRefMoeGlobalV45Config,
    RecurrentLactRefMoeGlobalV45ForCausalLM,
    exist_ok=True,
)

__all__ = [
    "RecurrentLactRefConfig",
    "RecurrentLactRefForCausalLM",
    "RecurrentLactRefModel",
    "RecurrentLactRefMoeGlobalV45Config",
    "RecurrentLactRefMoeGlobalV45ForCausalLM",
    "RecurrentLactRefMoeGlobalV45Model",
]
