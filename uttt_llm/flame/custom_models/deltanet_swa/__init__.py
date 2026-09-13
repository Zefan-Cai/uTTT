from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_deltanet_swa import DeltaNetWindowMixConfig
from .modeling_deltanet_swa import DeltaNetWindowMixForCausalLM, DeltaNetWindowMixModel

AutoConfig.register(DeltaNetWindowMixConfig.model_type, DeltaNetWindowMixConfig, exist_ok=True)
AutoModel.register(DeltaNetWindowMixConfig, DeltaNetWindowMixModel, exist_ok=True)
AutoModelForCausalLM.register(DeltaNetWindowMixConfig, DeltaNetWindowMixForCausalLM, exist_ok=True)

__all__ = ["DeltaNetWindowMixConfig", "DeltaNetWindowMixForCausalLM", "DeltaNetWindowMixModel"]
