from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_lact import RecurrentLactRefConfig
from .modeling_recurrent_lact import RecurrentLactRefForCausalLM, RecurrentLactRefModel

AutoConfig.register(RecurrentLactRefConfig.model_type, RecurrentLactRefConfig, exist_ok=True)
AutoModel.register(RecurrentLactRefConfig, RecurrentLactRefModel, exist_ok=True)
AutoModelForCausalLM.register(RecurrentLactRefConfig, RecurrentLactRefForCausalLM, exist_ok=True)

__all__ = [
    "RecurrentLactRefConfig",
    "RecurrentLactRefForCausalLM",
    "RecurrentLactRefModel",
]
