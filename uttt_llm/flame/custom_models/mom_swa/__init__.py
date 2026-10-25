from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mom_swa import MomWindowMixConfig
from .modeling_mom_swa import MomWindowMixForCausalLM, MomWindowMixModel

AutoConfig.register(MomWindowMixConfig.model_type, MomWindowMixConfig, exist_ok=True)
AutoModel.register(MomWindowMixConfig, MomWindowMixModel, exist_ok=True)
AutoModelForCausalLM.register(MomWindowMixConfig, MomWindowMixForCausalLM, exist_ok=True)

__all__ = ["MomWindowMixConfig", "MomWindowMixForCausalLM", "MomWindowMixModel"]
