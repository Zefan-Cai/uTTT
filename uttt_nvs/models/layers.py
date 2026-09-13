from einops import einsum, rearrange, repeat
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from .class_name import get_obj_by_name


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class SelfAttention(nn.Module):
    """
    Self-attention layer
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
        torch_impl=False,
    ):
        super().__init__()
        assert (
            dim % head_dim == 0
        ), f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.causal = causal
        self.torch_impl = torch_impl

        if self.torch_impl:
            assert not self.causal, "Causal attention is not supported for torch implementation."

    def forward(self, x, vis_dict=None, *args):
        """
        x: (b, l, D)
        """
        # token split, multi-head attention, token cat
        q, k, v = rearrange(
            self.to_qkv(x), 
            "b l (qkv h d) -> qkv b h l d", 
            qkv=3, d=self.head_dim
        )

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.torch_impl:
            alpha = einsum(q, k, "b h s d, b h t d -> b h s t")
            alpha = alpha / self.head_dim**0.5
            alpha = alpha.softmax(dim=-1)
            x = einsum(alpha, v, "b h s t, b h t d -> b h s d")
        else:
            x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.c_proj(x)

        return x, vis_dict

    def extra_repr(self) -> str:
        return f"causal={self.causal}"


class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, vis_dict=None, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x, vis_dict


class Block(nn.Module):
    def __init__(self, dim, bias, block_config):
        super().__init__()
        self.length_dim_list = []
        module_list = []


        for _, module_config in enumerate(block_config):
            CLASS = get_obj_by_name(module_config["type"])
            module = nn.ModuleDict(
                {
                    "ln": LayerNorm(dim, bias=bias),
                    "f": CLASS(dim=dim, bias=bias, **module_config["params"]),
                }
            )

            self.length_dim_list.append(module_config.get("length_dim", "vl"))
            module_list.append(module)

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x, shape_info):
        vis_dict = None
        for idx, (module, length_dim) in enumerate(zip(self.module_list, self.length_dim_list)):
            residual = x

            x = module["ln"](x)

            if length_dim == "l":
                b, vl, d = x.shape
                l = shape_info["num_img_tokens"]
                x = x.reshape(b * (vl // l), l, d)
                x, vis_dict = module["f"](x, vis_dict, shape_info)
                x = x.reshape(b, vl, d)
            else:
                x, vis_dict = module["f"](x, vis_dict, shape_info)

            x = residual + x
        return x, vis_dict
