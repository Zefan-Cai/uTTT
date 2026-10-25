"""Model registry for the uTTT language-modeling release.

Each model package registers its own config/model classes with the
transformers Auto* registries on import; importing this package is what makes
the `model_type` strings in configs/ resolvable. The two Transformer rows use
fla's built-in implementation rather than a package here; the KDA / Mamba-2 /
MoM rows wrap fla's layers (fla >= 0.5.2) with the shared sliding-window branch. Imports fail loudly on
purpose: every model here is part of the paper's Table 1, and a silently
unregistered model surfaces later as a confusing "model type not registered"
error far from its cause.
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from . import ttt_dense            # TTT-Dense (ours)        model_type: ttt_dense
from . import deltanet_swa         # DeltaNet-SWA
from . import gated_deltanet_swa   # Gated DeltaNet-SWA
from . import kda_swa              # KDA-SWA                 model_type: kda_swa
from . import mamba2_swa           # Mamba-2-SWA             model_type: mamba2_swa
from . import mom_swa              # MoM-SWA                 model_type: mom_swa
# The Transformer / Transformer-SWA rows run on fla's built-in transformer
# (model_type "transformer"); importing fla.models registers it.
import fla.models  # noqa: F401
from . import ttt_moe_no_lb        # TTT-MoE w/o LB          model_type: ttt_moe_no_lb
from . import ttt_moe_lb           # TTT-MoE w/ LB           model_type: ttt_moe_lb
from . import uttt_moe             # uTTT-MoE                model_type: uttt_moe
from . import uttt_dense           # uTTT-Dense (appendix)   model_type: uttt_dense

# LaCT (published) does not self-register, so register it here.
from .lact import LaCTSWIGLUConfig, LaCTForCausalLM, LaCTModel

AutoConfig.register(LaCTSWIGLUConfig.model_type, LaCTSWIGLUConfig, exist_ok=True)
AutoModel.register(LaCTSWIGLUConfig, LaCTModel, exist_ok=True)
AutoModelForCausalLM.register(LaCTSWIGLUConfig, LaCTForCausalLM, exist_ok=True)
