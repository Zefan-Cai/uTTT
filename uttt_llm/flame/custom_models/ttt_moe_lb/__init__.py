from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_lact import (
    RecurrentLactRefConfig,
    RecurrentLactRefMoeV5Config,
)
from .modeling_recurrent_lact import (
    RecurrentLactRefForCausalLM,
    RecurrentLactRefModel,
    RecurrentLactRefMoeV5ForCausalLM,
    RecurrentLactRefMoeV5Model,
)

AutoConfig.register(
    RecurrentLactRefMoeV5Config.model_type,
    RecurrentLactRefMoeV5Config,
    exist_ok=True,
)
AutoModel.register(RecurrentLactRefMoeV5Config, RecurrentLactRefMoeV5Model, exist_ok=True)
AutoModelForCausalLM.register(
    RecurrentLactRefMoeV5Config,
    RecurrentLactRefMoeV5ForCausalLM,
    exist_ok=True,
)

__all__ = [
    "RecurrentLactRefConfig",
    "RecurrentLactRefForCausalLM",
    "RecurrentLactRefModel",
    "RecurrentLactRefMoeV5Config",
    "RecurrentLactRefMoeV5ForCausalLM",
    "RecurrentLactRefMoeV5Model",
]
