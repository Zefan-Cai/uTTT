from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_gated_deltanet_swa import GatedDeltaNetWindowMixConfig
from .modeling_gated_deltanet_swa import GatedDeltaNetWindowMixForCausalLM, GatedDeltaNetWindowMixModel

AutoConfig.register(GatedDeltaNetWindowMixConfig.model_type, GatedDeltaNetWindowMixConfig, exist_ok=True)
AutoModel.register(GatedDeltaNetWindowMixConfig, GatedDeltaNetWindowMixModel, exist_ok=True)
AutoModelForCausalLM.register(
    GatedDeltaNetWindowMixConfig,
    GatedDeltaNetWindowMixForCausalLM,
    exist_ok=True,
)

__all__ = ["GatedDeltaNetWindowMixConfig", "GatedDeltaNetWindowMixForCausalLM", "GatedDeltaNetWindowMixModel"]
