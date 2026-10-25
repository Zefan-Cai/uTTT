from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_lact import RecurrentLactRefUtttDenseV3Config
from .modeling_recurrent_lact import (
    FastWeightMLPSubNNUtttDenseV3,
    RecurrentLactRefUtttDenseV3Block,
    RecurrentLactRefUtttDenseV3ForCausalLM,
    RecurrentLactRefUtttDenseV3Model,
)

AutoConfig.register(
    RecurrentLactRefUtttDenseV3Config.model_type,
    RecurrentLactRefUtttDenseV3Config,
    exist_ok=True,
)
AutoModel.register(
    RecurrentLactRefUtttDenseV3Config,
    RecurrentLactRefUtttDenseV3Model,
    exist_ok=True,
)
AutoModelForCausalLM.register(
    RecurrentLactRefUtttDenseV3Config,
    RecurrentLactRefUtttDenseV3ForCausalLM,
    exist_ok=True,
)

__all__ = [
    "FastWeightMLPSubNNUtttDenseV3",
    "RecurrentLactRefUtttDenseV3Config",
    "RecurrentLactRefUtttDenseV3Block",
    "RecurrentLactRefUtttDenseV3ForCausalLM",
    "RecurrentLactRefUtttDenseV3Model",
]
