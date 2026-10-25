from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_kda_swa import KDAWindowMixConfig
from .modeling_kda_swa import KDAWindowMixForCausalLM, KDAWindowMixModel

AutoConfig.register(KDAWindowMixConfig.model_type, KDAWindowMixConfig, exist_ok=True)
AutoModel.register(KDAWindowMixConfig, KDAWindowMixModel, exist_ok=True)
AutoModelForCausalLM.register(
    KDAWindowMixConfig,
    KDAWindowMixForCausalLM,
    exist_ok=True,
)

__all__ = ["KDAWindowMixConfig", "KDAWindowMixForCausalLM", "KDAWindowMixModel"]
