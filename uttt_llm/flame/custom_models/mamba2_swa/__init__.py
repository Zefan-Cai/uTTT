from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mamba2_swa import Mamba2WindowMixConfig
from .modeling_mamba2_swa import Mamba2WindowMixForCausalLM, Mamba2WindowMixModel

AutoConfig.register(Mamba2WindowMixConfig.model_type, Mamba2WindowMixConfig, exist_ok=True)
AutoModel.register(Mamba2WindowMixConfig, Mamba2WindowMixModel, exist_ok=True)
AutoModelForCausalLM.register(Mamba2WindowMixConfig, Mamba2WindowMixForCausalLM, exist_ok=True)

__all__ = ["Mamba2WindowMixConfig", "Mamba2WindowMixForCausalLM", "Mamba2WindowMixModel"]
