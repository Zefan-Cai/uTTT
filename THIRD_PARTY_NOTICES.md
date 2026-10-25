# Third-party notices

The uTTT repository is released under the MIT license (see `LICENSE`). It
vendors or derives code from the projects below, each of which keeps its own
copyright and license. Where a file carries an upstream header, that header is
authoritative; this file summarizes them. Full texts of the MIT and
BSD-3-Clause licenses are reproduced once at the end.

| Component | Where in this repository | Upstream | License |
|---|---|---|---|
| LaCT (Large-Chunk Test-Time Training) | `uttt_llm/flame/custom_models/lact/` (verbatim upstream `lact_model/`, `configs/`, `README.md`, `requirement.txt`); the fast-weight operation and fused kernels of `uttt_llm/flame/custom_models/{ttt_dense,ttt_moe_no_lb,ttt_moe_lb,uttt_moe,uttt_dense}/` and `uttt_nvs/models/kernels/` (`ttt_operation*.py`, `lact_*`, `triton_grouped_*`, `triton_lact_*`, `triton_pointwise_kernels.py`) are derived from it | Tianyuan Zhang et al., https://tianyuanzhang.com/projects/ttt-done-right/ | MIT |
| flame training harness | `uttt_llm/flame/` (training loop, config manager, data pipeline, checkpointing, `utils/preprocess.py`, `utils/convert_dcp_to_hf.py`, `utils/convert_hf_to_dcp.py`); files headed "Copyright (c) 2023-2025, Songlin Yang, Yu Zhang" | Songlin Yang, Yu Zhang, https://github.com/fla-org/flame | MIT |
| flash-linear-attention (`fla`) | `rotary.py` in each of `uttt_llm/flame/custom_models/{ttt_dense,ttt_moe_no_lb,ttt_moe_lb,uttt_moe,uttt_dense}/` (headed "Copyright (c) 2023-2025, Songlin Yang, Yu Zhang"); `uttt_llm/flame/custom_models/lact/lact_model/lact_triton_kernels/l2norm_triton_kernels.py` ("Modified from fla/modules/l2norm.py"). `uttt_nvs/models/{deltanet,gated_deltanet}_block.py` re-implement fla's `ShortConvolution` and the (gated) delta-rule mixers in plain PyTorch ("PyTorch replacements for fla modules"). The `fla` package itself is an installed dependency for the LLM side, not vendored | Songlin Yang, Yu Zhang, https://github.com/fla-org/flash-linear-attention | MIT |
| torchtitan | Files headed "Copyright (c) Meta Platforms, Inc. and affiliates": `uttt_llm/flame/train.py`, `uttt_llm/flame/eval.py`, `uttt_llm/flame/eval_loss.py`, `uttt_llm/flame/config_manager.py`, `uttt_llm/flame/components/checkpoint.py`, `uttt_llm/flame/models/parallelize_fla.py`, `uttt_llm/flame/models/pipeline_fla.py`, `uttt_llm/flame/tools/utils.py`. The `torchtitan` package itself is an installed dependency | Meta Platforms, Inc., https://github.com/pytorch/torchtitan | BSD-3-Clause |
| torchtune | `uttt_llm/flame/models/activation_offloading.py` (headed "Copyright (c) Meta Platforms, Inc. and affiliates"; "Adapted from torchtune/training/_activation_offloading.py") | Meta Platforms, Inc., https://github.com/pytorch/torchtune | BSD-3-Clause |
| TransformerEngine permutation kernels | `uttt_nvs/models/kernels/triton_permute.py` and `uttt_llm/flame/custom_models/{ttt_moe_no_lb,ttt_moe_lb,uttt_moe}/lact_moe_triton/triton_permute.py` ("modified from transformer_engine/pytorch/triton/permutation.py"; three argsort kernels adapted from triton-lang/triton issue #3698) | NVIDIA Corporation, https://github.com/NVIDIA/TransformerEngine | Apache-2.0 |
| Moonlight (Muon update step) | The Newton–Schulz orthogonalization in `ttt_operation*.py` under `uttt_nvs/models/kernels/`, `uttt_nvs/models/ttt_minimal_v2.py`, `uttt_nvs/models/ttt_dense_block.py`, `uttt_nvs/models/{deltanet,gated_deltanet}_block.py` and `uttt_llm/flame/custom_models/lact/lact_model/` ("modified from MoonshotAI/Moonlight examples/toy_train.py"; coefficients adapted from suggestions by @jxbz, @leloykun and @YouJiacheng) | Moonshot AI, https://github.com/MoonshotAI/Moonlight | MIT |
| Crowdsampling the Plenoptic Function (VGG-19 perceptual loss) | `uttt_nvs/models/perceptual_loss.py` ("Adapted from zhengqili/Crowdsampling-the-Plenoptic-Function models/networks.py", via LaCT). The MatConvNet `imagenet-vgg-verydeep-19.mat` weights it loads are not vendored; download them from https://www.vlfeat.org/matconvnet/pretrained/ | Zhengqi Li et al., https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function | No license file in the upstream repository (checked 2026-10-25); the adapted VGG-19 feature extractor is used as LaCT (MIT) ships it |
| Mamba-2 Triton kernels (`mamba_ssm`) | `uttt_llm/flame/custom_models/mamba2_swa/mamba_ssm_triton/` (selective state-space scan kernels used by the Mamba-2-SWA baseline added in this release) | Tri Dao, Albert Gu, https://github.com/state-spaces/mamba | Apache-2.0 |

Other dependencies (PyTorch, Triton, flash-attn, Hugging Face `transformers`
and `datasets`, lm-evaluation-harness, LPIPS, and the rest of the two
`requirements.txt` files) are installed from their own distributions and are
not vendored here.

The novel-view-synthesis models follow the LVSM design (Jin et al., 2025) and
cite it; no file in `uttt_nvs/` carries an LVSM source header, and no LVSM code
is vendored.

The Apache-2.0 text is available at https://www.apache.org/licenses/LICENSE-2.0
and alongside the upstream files.

---

## MIT License

Applies to LaCT, flame, flash-linear-attention, Moonlight and Crowdsampling the Plenoptic Function as noted above.

```
MIT License

Copyright (c) <the respective copyright holders named above>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BSD 3-Clause License

Applies to the torchtitan- and torchtune-derived files as noted above.

```
BSD 3-Clause License

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
