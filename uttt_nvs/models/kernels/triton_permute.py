"""
This file is modified from the original file in TransformerEngine:
https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/triton/permutation.py

Most edits are about:
1. Adding a batch dimension to inputs and outputs.
2. A few tweaks to the shape. e.g. moving the expert dimension from the last dimension to the second last dimension.
"""

from typing import Union

import torch
import triton
import triton.language as tl

from triton.language import core
from triton.language.standard import _log2
from packaging import version

# The following three argsort related kernels are adapted from
# the issue https://github.com/triton-lang/triton/issues/3698


get_int_dtype = core.get_int_dtype
if version.parse(triton.__version__) >= version.parse("3.5.0"):
    get_int_dtype = triton.constexpr_function(get_int_dtype)


@triton.jit
def _compare_and_swap(x, indices, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    z = tl.reshape(indices, shape)

    mask = tl.arange(0, 2)[None, :, None]

    l_value = tl.reshape(
        tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape), x.shape
    ).to(x.dtype)
    r_value = tl.reshape(
        tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape), x.shape
    ).to(x.dtype)

    l_indice = tl.reshape(
        tl.broadcast_to(tl.sum(z * (1 - mask), 1)[:, None, :], shape), x.shape
    )
    r_indice = tl.reshape(
        tl.broadcast_to(tl.sum(z * mask, 1)[:, None, :], shape), x.shape
    )

    idtype = get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    il_value = l_value.to(idtype, bitcast=True)
    ir_value = r_value.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    flag1 = tl.where(
        ((l_value > r_value) ^ flip) != 0, il_value ^ ir_value, tl.zeros_like(ix)
    )
    ret = ix ^ flag1
    flag2 = tl.where(
        ((l_value > r_value) ^ flip) != 0, l_indice ^ r_indice, tl.zeros_like(ix)
    )
    ind = indices ^ flag2

    return ret.to(x.dtype, bitcast=True), ind


@triton.jit
def _bitonic_merge(
    x, indices, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    if order == 2:
        shape: tl.constexpr = [n_outer * (2 ** (n_dims - 1 - stage)), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = tl.full(x.shape, value=order, dtype=tl.int32)
    for i in tl.static_range(stage):
        x, indices = _compare_and_swap(x, indices, flip, i + (n_dims - stage), n_dims)
    return x, indices


@triton.jit
def _argsort(x, indices, n_dims: tl.constexpr):
    for i in tl.static_range(1, n_dims + 1):
        x, indices = _bitonic_merge(x, indices, i, 2 if i < n_dims else 1, n_dims)
    return x, indices


@triton.jit
def _row_id_map_batched_pass_1_kernel(
    routing_map_ptr,  # [B, E, num_tokens] # 0/1 tensor
    row_id_map_ptr,  # [B, num_tokens, E + E + 1],   # int32
    workspace_ptr,  # [B, E,  ceil(num_tokens, BLOCK_SIZE)] # int32
    # sizes
    num_tokens,
    # strides
    stride_routing_map_batch,
    stride_routing_map_expert,
    stride_routing_map_token,
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_workspace_batch,
    stride_workspace_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,  # in num_token dimension.
):
    pid_b = tl.program_id(axis=0)
    pid_e = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    routing_map_ptr = (
        routing_map_ptr
        + pid_b * stride_routing_map_batch
        + pid_e * stride_routing_map_expert
    )

    routing_map = tl.load(
        routing_map_ptr + offs_n * stride_routing_map_token,
        mask=offs_n < num_tokens,
        other=0,
    ).to(
        tl.int32
    )  # [BLOCK_SIZE]

    # compute cumsum

    row_id_within_block = tl.cumsum(routing_map) * routing_map

    n_tokens_within_block = tl.sum(routing_map)

    # store the row_id_within_block to the row_id_map, and store the n_tokens_within_block to the workspace
    row_id_map_ptr = (
        row_id_map_ptr
        + pid_b * stride_row_id_map_batch
        + pid_e * stride_row_id_map_expert
    )
    tl.store(
        row_id_map_ptr + offs_n * stride_row_id_map_token,
        row_id_within_block,
        mask=offs_n < num_tokens,
    )

    workspace_ptr = (
        workspace_ptr + pid_b * stride_workspace_batch + pid_e * stride_workspace_expert
    )
    tl.store(workspace_ptr + pid_n, n_tokens_within_block)


@triton.jit
def _row_id_map_batched_pass_2_kernel(
    # pointers
    row_id_map_ptr,  # [B, num_tokens, E + E + 1],   # int32
    workspace_ptr,  # [B, E,  ceil(num_tokens, BLOCK_SIZE)] # int32
    # sizes
    num_tokens,
    # strides
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_workspace_batch,
    stride_workspace_expert,
    # metas
    WORKSPACE_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_e = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_id_map_ptr = (
        row_id_map_ptr
        + pid_b * stride_row_id_map_batch
        + pid_e * stride_row_id_map_expert
    )

    row_id_map_within_block = tl.load(
        row_id_map_ptr + offs_n * stride_row_id_map_token,
        mask=offs_n < num_tokens,
        other=0,
    ).to(
        tl.int32
    )  # [BLOCK_SIZE]

    # load all the worksapce values up untill current block.

    workspace_ptr = workspace_ptr + pid_b * stride_workspace_batch
    block_end_index = pid_e * stride_workspace_expert + pid_n

    workspace_off = tl.arange(0, WORKSPACE_LOAD_WIDTH)
    workspace_values_all_experts_all_blocks = tl.load(
        workspace_ptr + workspace_off,
        mask=workspace_off < block_end_index,
        other=0,
    ).to(
        tl.int32
    )  # [E, WORKSPACE_LOAD_WIDTH]

    num_tokens_before_block = tl.sum(workspace_values_all_experts_all_blocks)

    row_id = tl.where(
        row_id_map_within_block == 0,
        -1,
        row_id_map_within_block + num_tokens_before_block - 1,
    )

    # store the updated row_id to the row_id_map

    tl.store(
        row_id_map_ptr + offs_n * stride_row_id_map_token,
        row_id,
        mask=offs_n < num_tokens,
    )


@triton.jit
def _row_id_map_batched_pass_3_kernel(
    # pointers
    row_id_map_ptr,  # [B, num_tokens, E + E + 1],   # int32
    # sizes
    num_experts: tl.constexpr,
    # strides
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # metas
    LOAD_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid = tl.program_id(1)
    n_dims: tl.constexpr = _log2(LOAD_SIZE)
    off = tl.arange(0, LOAD_SIZE)
    row_id_map_ptr = row_id_map_ptr + pid_b * stride_row_id_map_batch

    row_id_map = tl.load(
        row_id_map_ptr + pid * stride_row_id_map_token + stride_row_id_map_expert * off,
        mask=off < num_experts,
        other=-1,
    )
    n_routed = tl.sum(tl.where(row_id_map != -1, 1, 0))
    indices = off
    # descending orders
    sorted_map, indices = _argsort(row_id_map, indices, n_dims=n_dims)
    # step-1: store the sorted row indices first.
    # store in the first E positions.
    tl.store(
        row_id_map_ptr + pid * stride_row_id_map_token + off * stride_row_id_map_expert,
        sorted_map,
        mask=off < n_routed,
    )
    # step-2: store the actived expert indices, with descending orders. [E, E + num_activated_experts]
    tl.store(
        row_id_map_ptr
        + pid * stride_row_id_map_token
        + (num_experts + off) * stride_row_id_map_expert,
        indices,
        mask=off < n_routed,
    )
    # step-3: store the number of routed tokens.
    tl.store(
        row_id_map_ptr
        + pid * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert,
        n_routed,
    )


def make_row_id_map(
    routing_map: torch.Tensor,
    batch_size: int,
    num_tokens: int,
    num_experts: int,
):
    """
    Prepare the row_id_map for the permutation.

    Parameters
    ----------
    routing_map: torch.Tensor
        Input tensor of shape `[B, num_experts, num_tokens]`. It is a mask tensor that indicates
        which experts are routed to which tokens. The values in it: 1 means the token is routed to
        this expert and 0 means not.
    batch_size: int
        Number of batches in the input tensor.
    num_tokens: int
        Number of tokens in the input tensor.
    num_experts: int
        Number of experts in the input tensor.

    Returns
    -------
    row_id_map: torch.Tensor
        The row_id_map for the permutation of shape `[batch_size, num_tokens, num_experts * 2 + 1]`.
        For each token, the last item is the number of experts that are routed (n_routed).
        The first n_routed items are the destination row indices in the permuted tokens.
        The [num_experts, num_experts + n_routed) items are the indices of the experts corresponding
        to the first n_routed row indices above.
    """
    row_id_map = routing_map.new_zeros(
        (batch_size, num_tokens, num_experts * 2 + 1), dtype=torch.int32
    )
    # min_block_size = min(triton.next_power_of_2(num_tokens), 1024)
    # block_size = min(min_block_size, 1024)
    block_size = 512
    grid = (batch_size, num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = torch.empty(grid, dtype=torch.int32, device="cuda")

    # supposing num_tokens == 5, num_experts == 3, block_size == 3
    # and we have a routing_map like this:
    # [[1, 1, 0],
    #  [1, 0, 1],
    #  [0, 0, 1],
    #  [1, 1, 0],
    #  [0, 0, 0]]

    # pass 1: block cumsum
    # for each expert, compute the cumsum of every block_size tokens
    # the row_id_map will be like this after pass 1 (r means useless values):
    # [[1, 1, 0, r, r, r, r],
    #  [2, 0, 1, r, r, r, r],
    #  [0, 0, 2, r, r, r, r],
    #  [1, 1, 0, r, r, r, r],
    #  [0, 0, 0, r, r, r, r]]
    _row_id_map_batched_pass_1_kernel[grid](
        routing_map,
        row_id_map,
        workspace_tensor,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        routing_map.stride(2),
        row_id_map.stride(0),
        row_id_map.stride(1),
        row_id_map.stride(2),
        workspace_tensor.stride(0),
        workspace_tensor.stride(1),
        block_size,
    )

    # pass 2: cumsum all and process the mask
    # process the block cumsum into the global cumsum and then into the dst row indices
    # the row_id_map will be like this after pass 2 (r means useless value):
    # [[ 0,  3, -1, r, r, r, r],
    #  [ 1, -1,  5, r, r, r, r],
    #  [-1, -1,  6, r, r, r, r],
    #  [ 2,  4, -1, r, r, r, r],
    #  [-1, -1, -1, r, r, r, r]]
    _row_id_map_batched_pass_2_kernel[grid](
        row_id_map,
        workspace_tensor,
        num_tokens,
        row_id_map.stride(0),
        row_id_map.stride(1),
        row_id_map.stride(2),
        workspace_tensor.stride(0),
        workspace_tensor.stride(1),
        triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),
        block_size,
    )

    # pass 3: make the row_id_map from the sparse structure to the dense structure
    # the row_id_map will be like this after pass 3 (r means useless value):
    # [[3, 0, r, 1, 0, r, 2],
    #  [5, 1, r, 2, 0, r, 2],
    #  [6, r, r, 2, r, r, 1],
    #  [4, 2, r, 1, 0, r, 2],
    #  [r, r, r, r, r, r, 0]]
    grid = (batch_size, num_tokens)
    _row_id_map_batched_pass_3_kernel[grid](
        row_id_map,
        num_experts,
        row_id_map.stride(0),
        row_id_map.stride(1),
        row_id_map.stride(2),
        triton.next_power_of_2(num_experts),
    )

    return row_id_map


@triton.jit
def _permute_kernel(
    # pointers
    input_ptr,  # [B, num_tokens, hidden_size]
    output_ptr,  # [B, num_tokens_total, hidden_size]
    row_id_map_ptr,  # [B, num_tokens, num_experts * 2 + 1]
    probs_ptr,  # [B, num_experts, num_tokens_total]
    permuted_probs_ptr,  # [B, num_tokens_total]
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_batch,
    stride_input_token,
    stride_input_hidden,
    stride_output_batch,
    stride_output_token,
    stride_output_hidden,
    stride_probs_batch,
    stride_probs_expert,
    stride_probs_token,
    stride_permuted_probs_batch,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    input_ptr = input_ptr + pid_b * stride_input_batch
    output_ptr = output_ptr + pid_b * stride_output_batch
    row_id_map_ptr = row_id_map_ptr + pid_b * stride_row_id_map_batch
    if PERMUTE_PROBS:
        probs_ptr = probs_ptr + pid_b * stride_probs_batch
        permuted_probs_ptr = permuted_probs_ptr + pid_b * stride_permuted_probs_batch

    cur_off = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cur_off < hidden_size
    src_row = pid_t.to(tl.int64)
    input_off = src_row * stride_input_token + cur_off * stride_input_hidden
    inp = tl.load(input_ptr + input_off, mask=mask)
    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + num_experts * 2 * stride_row_id_map_expert
    )
    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + idx * stride_row_id_map_expert
        ).to(tl.int64)
        output_off = dst_row * stride_output_token + cur_off * stride_output_hidden

        if PERMUTE_PROBS:
            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )
            prob_off = pid_t * stride_probs_token + expert_idx * stride_probs_expert
            prob = tl.load(probs_ptr + prob_off)
            if pid_h == 0:
                permuted_prob_off = dst_row * stride_permuted_probs_token
                tl.store(permuted_probs_ptr + permuted_prob_off, prob)
            if prob == 0.0:
                # for routing_map padding
                # dst_row != -1 and prob == 0.0 means that this slot is padded
                tl.store(output_ptr + output_off, 0.0, mask=mask)
            else:
                tl.store(output_ptr + output_off, inp, mask=mask)
        else:
            tl.store(output_ptr + output_off, inp, mask=mask)


try:
    _permute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_permute_kernel)
except RuntimeError:
    pass


def triton_permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    batch_size: int,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    """
    Permute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[B, num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map: torch.Tensor
        The token to expert mapping tensor of shape `[B, num_tokens, num_experts * 2 + 1]`.
    probs: torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
        Shape: `[B, num_experts, num_tokens_total]`.
    num_tokens: int
        Number of tokens in the input tensor.
    num_experts: int
        Number of experts in the input tensor.
    num_out_tokens: int
        Number of tokens in the permuted tensor.
    hidden_size: int
        Hidden size of the input tensor.
    scale_hidden_dim: int
        Hidden size of the scale tensor.
    """
    output = torch.empty(
        (batch_size, num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda"
    )
    if probs is not None:
        permuted_probs = torch.empty(
            (
                batch_size,
                num_out_tokens,
            ),
            dtype=probs.dtype,
            device="cuda",
        )
    else:
        permuted_probs = None

    # pylint: disable=unnecessary-lambda-assignment
    grid = lambda META: (
        batch_size,
        num_tokens,
        triton.cdiv(hidden_size, META["BLOCK_SIZE"]),
    )
    _permute_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        num_experts,
        hidden_size,
        row_id_map.stride(0),
        row_id_map.stride(1),
        row_id_map.stride(2),
        inp.stride(0),
        inp.stride(1),
        inp.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        probs.stride(0) if probs is not None else None,
        probs.stride(1) if probs is not None else None,
        probs.stride(2) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        permuted_probs.stride(1) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
    )
    return output, permuted_probs


########################################################
# Backward pass of permute
########################################################


@triton.jit
def _permute_bwd_kernel_batched(
    # pointers
    fwd_output_grad_ptr,  # [B, num_out_tokens, hidden_size]
    fwd_input_grad_ptr,  # [B, num_tokens, hidden_size]
    row_id_map_ptr,  # [B, num_tokens, 2*num_experts + 1]
    permuted_probs_grad_ptr,  # [B, num_out_tokens]
    probs_grad_ptr,  # [B, num_experts, num_tokens], same shape as probs_ptr
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_fwd_output_grad_batch,
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_batch,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_permuted_probs_grad_batch,
    stride_permuted_probs_grad_token,
    stride_probs_grad_batch,
    stride_probs_grad_expert,
    stride_probs_grad_token,
    # metas
    PROBS_LOAD_WIDTH: tl.constexpr,
    COMPUTE_PROB_GRAD: tl.constexpr,  # weather to backpropagate the probs grad
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # batch offsets
    fwd_output_grad_ptr += pid_b * stride_fwd_output_grad_batch
    fwd_input_grad_ptr += pid_b * stride_fwd_input_grad_batch
    row_id_map_ptr += pid_b * stride_row_id_map_batch
    if COMPUTE_PROB_GRAD:
        permuted_probs_grad_ptr += pid_b * stride_permuted_probs_grad_batch
        probs_grad_ptr += pid_b * stride_probs_grad_batch

    cur_off = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    h_mask = cur_off < hidden_size

    # zero grad row for probs once per (b, token)
    if COMPUTE_PROB_GRAD and (pid_h == 0):
        map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
        probs_grad_row_off = (
            pid_t * stride_probs_grad_token + map_load_off * stride_probs_grad_expert
        )
        tl.store(
            probs_grad_ptr + probs_grad_row_off, 0.0, mask=map_load_off < num_experts
        )

    act_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)

    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + (num_experts * 2) * stride_row_id_map_expert
    )  # last item is the number of routed tokens

    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + idx * stride_row_id_map_expert
        ).to(tl.int64)

        g_off = (
            dst_row * stride_fwd_output_grad_token
            + cur_off * stride_fwd_output_grad_hidden
        )
        gout = tl.load(fwd_output_grad_ptr + g_off, mask=h_mask).to(compute_type)

        act_accum += gout

        if COMPUTE_PROB_GRAD and (pid_h == 0):
            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )
            gpp_off = dst_row * stride_permuted_probs_grad_token
            gpp = tl.load(permuted_probs_grad_ptr + gpp_off)
            probs_grad_off = (
                pid_t * stride_probs_grad_token + expert_idx * stride_probs_grad_expert
            )
            tl.store(probs_grad_ptr + probs_grad_off, gpp)

    # convert to the original data type, then store back.
    act_accum = act_accum.to(data_type)
    gin_off = (
        pid_t.to(tl.int64) * stride_fwd_input_grad_token
        + cur_off * stride_fwd_input_grad_hidden
    )
    tl.store(fwd_input_grad_ptr + gin_off, act_accum, mask=h_mask)


try:
    _permute_bwd_kernel_batched = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_permute_bwd_kernel_batched)
except RuntimeError:
    pass


def triton_permute_with_mask_map_bwd_batched(
    fwd_output_grad: torch.Tensor,  # [B, num_out_tokens, hidden_size]
    row_id_map: torch.Tensor,  # [B, num_tokens, 2*num_experts+1]
    permuted_probs_grad: Union[torch.Tensor, None],  # [B, num_out_tokens]
    batch_size: int,
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
):
    """
    Batched backward for the simple permute (no scale).
    Returns (grad_inp, grad_probs)
        grad_inp: [B, num_tokens, hidden_size]
        grad_probs: [B, num_experts, num_tokens]
    """
    device = fwd_output_grad.device
    dtype = fwd_output_grad.dtype

    grad_inp = torch.empty(
        (batch_size, num_tokens, hidden_size), dtype=dtype, device=device
    )
    if permuted_probs_grad is not None:
        grad_probs = torch.empty(
            (batch_size, num_experts, num_tokens),
            dtype=permuted_probs_grad.dtype,
            device=device,
        )
    else:
        grad_probs = None

    grid = lambda META: (
        batch_size,
        num_tokens,
        triton.cdiv(hidden_size, META["BLOCK_SIZE"]),
    )

    _permute_bwd_kernel_batched[grid](
        fwd_output_grad,
        grad_inp,
        row_id_map,
        (permuted_probs_grad if permuted_probs_grad is not None else grad_inp),
        (grad_probs if grad_probs is not None else grad_inp),
        num_experts,
        hidden_size,
        # strides
        row_id_map.stride(0),
        row_id_map.stride(1),
        row_id_map.stride(2),
        fwd_output_grad.stride(0),
        fwd_output_grad.stride(1),
        fwd_output_grad.stride(2),
        grad_inp.stride(0),
        grad_inp.stride(1),
        grad_inp.stride(2),
        (permuted_probs_grad.stride(0) if permuted_probs_grad is not None else 0),
        (permuted_probs_grad.stride(1) if permuted_probs_grad is not None else 0),
        (grad_probs.stride(0) if grad_probs is not None else 0),
        (grad_probs.stride(1) if grad_probs is not None else 0),
        (grad_probs.stride(2) if grad_probs is not None else 0),
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
        COMPUTE_PROB_GRAD=(permuted_probs_grad is not None),
    )
    return grad_inp, grad_probs


@triton.jit
def _unpermute_kernel(
    # pointers
    input_ptr,  # [B, num_out_tokens, hidden_size]
    output_ptr,  # [B, num_tokens, hidden_size]
    row_id_map_ptr,  # [B, num_tokens, num_experts * 2 + 1]
    merging_probs_ptr,  # [B, num_experts, num_tokens]
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_input_batch,
    stride_input_token,
    stride_input_hidden,
    stride_output_batch,
    stride_output_token,
    stride_output_hidden,
    stride_merging_probs_batch,
    stride_merging_probs_expert,
    stride_merging_probs_token,
    WITH_MERGING_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = input_ptr.dtype.element_ty
    compute_type = tl.float32

    # program ids: (batch, token, hidden-tile)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # batch offsets
    input_ptr += pid_b * stride_input_batch
    output_ptr += pid_b * stride_output_batch
    row_id_map_ptr += pid_b * stride_row_id_map_batch
    if WITH_MERGING_PROBS:
        merging_probs_ptr += pid_b * stride_merging_probs_batch

    current_offset = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = current_offset < hidden_size

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)

    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + (num_experts * 2) * stride_row_id_map_expert
    )

    for idx in tl.range(n_routed):
        # which row in the permuted buffer contributes to token pid_t
        src_row = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + idx * stride_row_id_map_expert
        ).to(tl.int64)

        # load from permuted input
        input_off = src_row * stride_input_token + current_offset * stride_input_hidden
        inp = tl.load(input_ptr + input_off, mask=mask).to(compute_type)

        if WITH_MERGING_PROBS:
            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )
            # works for either [B, T, E] or [B, E, T]; just pass correct strides
            merging_prob_off = (
                pid_t * stride_merging_probs_token
                + expert_idx * stride_merging_probs_expert
            )
            merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(
                compute_type
            )
            inp *= merging_prob

        accumulator += inp

    accumulator = accumulator.to(data_type)
    dst_row = pid_t.to(tl.int64)
    output_off = dst_row * stride_output_token + current_offset * stride_output_hidden
    tl.store(output_ptr + output_off, accumulator, mask=mask)


try:
    _unpermute_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_unpermute_kernel)
except RuntimeError:
    pass


def triton_unpermute_with_mask_map(
    inp: torch.Tensor,  # [B, num_out_tokens, hidden_size]
    row_id_map: torch.Tensor,  # [B, num_tokens, 2*num_experts+1]
    merging_probs: Union[
        torch.Tensor, None
    ],  # typically [B, num_experts, num_tokens] or [B, num_tokens, num_experts]
    batch_size: int,
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
):
    """
    Unpermute the input tensor (batched) based on row_id_map.
    Returns: (output, unpermuted_probs)
      output: [B, num_tokens, hidden_size]
      unpermuted_probs: [B, num_tokens, num_experts] if permuted_probs is not None else None
    """
    device = inp.device
    dtype = inp.dtype

    output = torch.empty(
        (batch_size, num_tokens, hidden_size), dtype=dtype, device=device
    )

    # grid: (B, T, H-tiles)
    grid = lambda META: (
        batch_size,
        num_tokens,
        triton.cdiv(hidden_size, META["BLOCK_SIZE"]),
    )

    _unpermute_kernel[grid](
        # ptrs
        inp,
        output,
        row_id_map,
        merging_probs,
        # sizes
        num_experts,
        hidden_size,
        # strides (row_id_map)
        row_id_map.stride(0),  # batch
        row_id_map.stride(1),  # token
        row_id_map.stride(2),  # expert(+meta)
        # strides (input / output)
        inp.stride(0),
        inp.stride(1),
        inp.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # strides (merging_probs)
        (merging_probs.stride(0) if merging_probs is not None else 0),
        (merging_probs.stride(1) if merging_probs is not None else 0),
        (merging_probs.stride(2) if merging_probs is not None else 0),
        # metas
        WITH_MERGING_PROBS=(merging_probs is not None),
    )
    return output


@triton.jit
def _unpermute_bwd_with_merging_probs_kernel(
    # pointers
    fwd_output_grad_ptr,  # [B, num_tokens, hidden_size]
    fwd_input_grad_ptr,  # [B, num_out_tokens, hidden_size]
    fwd_input_ptr,  # [B, num_out_tokens, hidden_size]
    merging_probs_ptr,  # [B, T, E] or [B, E, T] -> use strides
    merging_probs_grad_ptr,  # same shape/strides as merging_probs
    row_id_map_ptr,  # [B, num_tokens, num_experts * 2 + 1]
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    stride_fwd_output_grad_batch,
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_batch,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_batch,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_batch,
    stride_merging_probs_expert,
    stride_merging_probs_token,
    stride_merging_probs_grad_batch,
    stride_merging_probs_grad_expert,
    stride_merging_probs_grad_token,
    # metas
    PROBS_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    # program ids: (batch, token)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # add batch offsets
    fwd_output_grad_ptr += pid_b * stride_fwd_output_grad_batch
    fwd_input_grad_ptr += pid_b * stride_fwd_input_grad_batch
    fwd_input_ptr += pid_b * stride_fwd_input_batch
    row_id_map_ptr += pid_b * stride_row_id_map_batch
    merging_probs_ptr += pid_b * stride_merging_probs_batch
    merging_probs_grad_ptr += pid_b * stride_merging_probs_grad_batch

    # zero grad row for this token
    map_load_off = tl.arange(0, PROBS_LOAD_WIDTH)
    token_probs_grad_off = (
        pid_t * stride_merging_probs_grad_token
        + map_load_off * stride_merging_probs_grad_expert
    )
    tl.store(
        merging_probs_grad_ptr + token_probs_grad_off,
        0.0,
        mask=map_load_off < num_experts,
    )

    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + (num_experts * 2) * stride_row_id_map_expert
    )

    for idx in tl.range(n_routed):
        dst_row = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + idx * stride_row_id_map_expert
        ).to(tl.int64)

        expert_idx = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + (num_experts + idx) * stride_row_id_map_expert
        )

        prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)

        current_start = 0
        while current_start < hidden_size:
            current_offset = current_start + tl.arange(0, BLOCK_SIZE)
            mask = current_offset < hidden_size

            # grad wrt output token pid_t comes from fwd_output_grad[pid_t, :]
            src_row = pid_t.to(tl.int64)
            in_off = (
                src_row * stride_fwd_output_grad_token
                + current_offset * stride_fwd_output_grad_hidden
            )
            inp = tl.load(fwd_output_grad_ptr + in_off, mask=mask).to(compute_type)

            # apply merging prob and scatter to fwd_input_grad[dst_row, :]
            merging_prob_off = (
                pid_t * stride_merging_probs_token
                + expert_idx * stride_merging_probs_expert
            )
            merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(
                compute_type
            )

            out_vals = (inp * merging_prob).to(data_type)
            out_off = (
                dst_row * stride_fwd_input_grad_token
                + current_offset * stride_fwd_input_grad_hidden
            )
            tl.store(fwd_input_grad_ptr + out_off, out_vals, mask=mask)

            # accumulate dL/d(merging_prob) = sum_h fwd_input[dst_row, h] * dL/dy[token, h]
            fwd_input_off = (
                dst_row * stride_fwd_input_token
                + current_offset * stride_fwd_input_hidden
            )
            fwd_input_vals = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
            prob_grad_accum += fwd_input_vals.to(compute_type) * inp

            current_start += BLOCK_SIZE

        probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
        probs_grad_off = (
            pid_t * stride_merging_probs_grad_token
            + expert_idx * stride_merging_probs_grad_expert
        )
        tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)


try:
    _unpermute_bwd_with_merging_probs_kernel = triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
            triton.Config({"BLOCK_SIZE": 4096}),
        ],
        key=["hidden_size"],
    )(_unpermute_bwd_with_merging_probs_kernel)
except RuntimeError:
    pass


def triton_unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,  # [B, num_tokens, hidden_size]
    row_id_map: torch.Tensor,  # [B, num_tokens, 2*num_experts+1]
    fwd_input: torch.Tensor,  # [B, num_out_tokens, hidden_size]
    merging_probs: torch.Tensor,  # [B, E, T]
    batch_size: int,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    """
    Batched backward for unpermute with merging probs.
    Returns: (act_grad, merging_probs_grad)
      act_grad: [B, num_out_tokens, hidden_size]
      merging_probs_grad: same shape as merging_probs
    """
    device = fwd_output_grad.device
    dtype = fwd_output_grad.dtype

    act_grad = torch.empty(
        (batch_size, num_out_tokens, hidden_size), dtype=dtype, device=device
    )
    merging_probs_grad = torch.empty_like(merging_probs)

    # (B, T)
    grid = (batch_size, num_tokens)

    _unpermute_bwd_with_merging_probs_kernel[grid](
        fwd_output_grad,
        act_grad,
        fwd_input,
        merging_probs,
        merging_probs_grad,
        row_id_map,
        num_experts,
        hidden_size,
        # row_id_map strides
        row_id_map.stride(0),
        row_id_map.stride(1),
        row_id_map.stride(2),
        # fwd_output_grad strides
        fwd_output_grad.stride(0),
        fwd_output_grad.stride(1),
        fwd_output_grad.stride(2),
        # fwd_input_grad (act_grad) strides
        act_grad.stride(0),
        act_grad.stride(1),
        act_grad.stride(2),
        # fwd_input strides
        fwd_input.stride(0),
        fwd_input.stride(1),
        fwd_input.stride(2),
        # merging_probs strides
        merging_probs.stride(0),
        merging_probs.stride(1),
        merging_probs.stride(2),
        # merging_probs_grad strides
        merging_probs_grad.stride(0),
        merging_probs_grad.stride(1),
        merging_probs_grad.stride(2),
        PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
    )
    return act_grad, merging_probs_grad


def _test_row_id_map_correctness():
    batch_size = 2
    num_tokens = 5
    num_experts = 3
    routing_map = torch.tensor(
        [
            [[1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]],
            [[0, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    print("Input routing_map:")
    print(routing_map)
    routing_map = routing_map.permute(0, 2, 1).contiguous()
    row_id_map = make_row_id_map(routing_map, batch_size, num_tokens, num_experts)
    print(row_id_map)


########################################################
# warper with torch.autograd.Function
########################################################


class PermuteWithMaskMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, expert_mask, probs, topk: int, row_id_map: torch.Tensor):
        """
        Args:
            inp: [B, num_tokens, hidden_size]
            expert_mask: [B, num_experts, num_tokens] 0 for not selected, 1 for selected
            probs: [B, num_experts, num_tokens] or None
        Outs:
            out: [B, num_output_tokens, hidden_size]
            permuted_probs: [B, num_output_tokens]

        """
        B, num_tokens, hidden_size = inp.shape
        num_experts = expert_mask.shape[1]
        num_output_tokens = topk * num_tokens

        if row_id_map is None:
            row_id_map = make_row_id_map(expert_mask, B, num_tokens, num_experts)
        out, permuted_probs = triton_permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            B,
            num_tokens,
            num_experts,
            num_output_tokens,
            hidden_size,
        )
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.hidden_size = hidden_size
        ctx.batch_size = B
        ctx.save_for_backward(probs, row_id_map)
        return out, permuted_probs, row_id_map

    @staticmethod
    def backward(ctx, grad_permuted_output, grad_permuted_probs, grad_row_id_map):
        """
        Args:
            grad_permuted_output: [B, num_output_tokens, hidden_size]
            grad_permuted_probs: [B, num_output_tokens] or None
            grad_row_id_map: None
        Outs:
            grad_inp: [B, num_tokens, hidden_size]
            grad_expert_mask: None
            grad_probs: [B, num_experts, num_tokens]
        """

        B, num_tokens, hidden_size, num_experts = (
            ctx.batch_size,
            ctx.num_tokens,
            ctx.hidden_size,
            ctx.num_experts,
        )
        if grad_permuted_probs is not None:
            grad_permuted_probs = grad_permuted_probs.contiguous()

        probs, row_id_map = ctx.saved_tensors

        grad_inp, grad_probs = triton_permute_with_mask_map_bwd_batched(
            grad_permuted_output.contiguous(),
            row_id_map,
            grad_permuted_probs,
            B,
            num_tokens,
            num_experts,
            hidden_size,
        )
        return grad_inp, None, grad_probs, None, None


def permute_with_expert_mask(
    inp: torch.Tensor,
    expert_mask: torch.Tensor,
    probs: torch.Tensor,
    topk: int,
    row_id_map: torch.Tensor = None,
):
    return PermuteWithMaskMap.apply(inp, expert_mask, probs, topk, row_id_map)


class UnPermuteWithMaskMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_inps, row_id_map, original_probs):
        """
        Args:
            permuted_inp: [B, num_output_tokens, hidden_size]
            row_id_map: [B, num_tokens, 2*num_experts+1]
            original_probs: [B, num_experts, num_tokens] or None
        Outs:
            merged_unpermuted_inps: [B, num_tokens, hidden_size]
        """
        B, num_output_tokens, hidden_size = permuted_inps.shape
        _, num_experts, num_tokens = original_probs.shape

        merged_unpermuted_inps = triton_unpermute_with_mask_map(
            permuted_inps,
            row_id_map,
            original_probs,
            B,
            num_tokens,
            num_experts,
            hidden_size,
        )
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.hidden_size = hidden_size
        ctx.batch_size = B
        ctx.num_output_tokens = num_output_tokens
        ctx.save_for_backward(permuted_inps, row_id_map, original_probs)
        return merged_unpermuted_inps

    @staticmethod
    def backward(ctx, grad_merged_output):
        """
        Args:
            grad_merged_output: [B, num_tokens, hidden_size]
        Outs:
            grad_permuted_inps: [B, num_output_tokens, hidden_size]
            grad_original_probs: [B, num_experts, num_tokens]
        """

        fwd_input, row_id_map, merging_probs = ctx.saved_tensors
        num_experts = ctx.num_experts
        num_tokens = ctx.num_tokens
        hidden_size = ctx.hidden_size
        batch_size = ctx.batch_size
        num_output_tokens = ctx.num_output_tokens

        grad_permuted_inps, grad_original_probs = (
            triton_unpermute_with_mask_map_bwd_with_merging_probs(
                grad_merged_output.contiguous(),
                row_id_map,
                fwd_input,
                merging_probs,
                batch_size,
                num_tokens,
                num_experts,
                num_output_tokens,
                hidden_size,
            )
        )

        return grad_permuted_inps, None, grad_original_probs


def unpermute_and_merge_with_probs(permuted_x, row_id_map, original_probs):
    """
    Args:
        permuted_x: [B, num_output_tokens, hidden_size]
        row_id_map: [B, num_tokens, 2*num_experts+1]
        original_probs: [B, num_experts, num_tokens] or None
    Outs:
        merged_unpermuted_inps: [B, num_tokens, hidden_size]
    """
    return UnPermuteWithMaskMap.apply(permuted_x, row_id_map, original_probs)


########################################################
# merged triton kernels for TTT MoE where we want to permute key, value and lr0, lr1, lr2 together
########################################################


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _permute_kv_with_lrs_kernel(
    # inputs
    k_ptr,  # [B, T, H]
    v_ptr,  # [B, T, H]
    lr0_ptr,  # [B, T]
    lr1_ptr,  # [B, T]
    lr2_ptr,  # [B, T]
    row_id_map_ptr,  # [B, T, 2*E+1], int32
    probs_ptr,  # [B, E, T], fp32
    # outputs
    out_k_ptr,  # [B, N, H]
    out_v_scaled_ptr,  # [B, N, H]
    out_lr0_ptr,  # [B, N]
    out_lr1_ptr,  # [B, N]
    out_lr2_ptr,  # [B, N]
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides: row_id_map
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # strides: k
    stride_k_batch,
    stride_k_token,
    stride_k_hidden,
    # strides: v
    stride_v_batch,
    stride_v_token,
    stride_v_hidden,
    # strides: lr0
    stride_lr0_batch,
    stride_lr0_token,
    # strides: lr1
    stride_lr1_batch,
    stride_lr1_token,
    # strides: lr2
    stride_lr2_batch,
    stride_lr2_token,
    # strides: out_k
    stride_out_k_batch,
    stride_out_k_token,
    stride_out_k_hidden,
    # strides: out_v_scaled
    stride_out_v_batch,
    stride_out_v_token,
    stride_out_v_hidden,
    # strides: out_lr0
    stride_out_lr0_batch,
    stride_out_lr0_token,
    # strides: out_lr1
    stride_out_lr1_batch,
    stride_out_lr1_token,
    # strides: out_lr2
    stride_out_lr2_batch,
    stride_out_lr2_token,
    # strides: probs
    stride_probs_batch,
    stride_probs_expert,
    stride_probs_token,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    data_type_k = k_ptr.dtype.element_ty
    data_type_v = out_v_scaled_ptr.dtype.element_ty
    compute_type = tl.float32

    # program ids: (batch, token, hidden-tile)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # batch offsets
    k_ptr += pid_b * stride_k_batch
    v_ptr += pid_b * stride_v_batch
    lr0_ptr += pid_b * stride_lr0_batch
    lr1_ptr += pid_b * stride_lr1_batch
    lr2_ptr += pid_b * stride_lr2_batch
    row_id_map_ptr += pid_b * stride_row_id_map_batch
    probs_ptr += pid_b * stride_probs_batch
    out_k_ptr += pid_b * stride_out_k_batch
    out_v_scaled_ptr += pid_b * stride_out_v_batch
    out_lr0_ptr += pid_b * stride_out_lr0_batch
    out_lr1_ptr += pid_b * stride_out_lr1_batch
    out_lr2_ptr += pid_b * stride_out_lr2_batch

    # hidden offsets for this tile
    offs_h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    h_mask = offs_h < hidden_size

    src_row = pid_t.to(tl.int64)

    # load k and v for this token and hidden tile
    k_off = src_row * stride_k_token + offs_h * stride_k_hidden
    v_off = src_row * stride_v_token + offs_h * stride_v_hidden
    k_vec = tl.load(k_ptr + k_off, mask=h_mask)
    v_vec = tl.load(v_ptr + v_off, mask=h_mask)

    # load scalar lrs once per (B, T)
    # if pid_h == 0:
    #     lr0_val = tl.load(lr0_ptr + src_row * stride_lr0_token)
    #     lr1_val = tl.load(lr1_ptr + src_row * stride_lr1_token)
    #     lr2_val = tl.load(lr2_ptr + src_row * stride_lr2_token)

    # number of routed experts for this token
    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + (num_experts * 2) * stride_row_id_map_expert
    )

    for idx in tl.range(0, n_routed):
        # destination row index
        dst_row = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + idx * stride_row_id_map_expert
        ).to(tl.int64)

        # expert index for this slot
        expert_idx = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + (num_experts + idx) * stride_row_id_map_expert
        )

        # load router prob p(b, e, t)
        prob_off = pid_t * stride_probs_token + expert_idx * stride_probs_expert
        prob = tl.load(probs_ptr + prob_off).to(compute_type)

        out_k_off = dst_row * stride_out_k_token + offs_h * stride_out_k_hidden
        out_v_off = dst_row * stride_out_v_token + offs_h * stride_out_v_hidden

        tl.store(out_k_ptr + out_k_off, k_vec, mask=h_mask)
        v_vals = v_vec.to(compute_type) * prob
        tl.store(out_v_scaled_ptr + out_v_off, v_vals.to(data_type_v), mask=h_mask)

        # lr0/1/2: permute according to row_id_map, independent of prob
        if pid_h == 0:
            lr0_val = tl.load(lr0_ptr + src_row * stride_lr0_token)
            lr1_val = tl.load(lr1_ptr + src_row * stride_lr1_token)
            lr2_val = tl.load(lr2_ptr + src_row * stride_lr2_token)

            lr0_row_off = dst_row * stride_out_lr0_token
            lr1_row_off = dst_row * stride_out_lr1_token
            lr2_row_off = dst_row * stride_out_lr2_token
            tl.store(out_lr0_ptr + lr0_row_off, lr0_val)
            tl.store(out_lr1_ptr + lr1_row_off, lr1_val)
            tl.store(out_lr2_ptr + lr2_row_off, lr2_val)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _permute_kv_with_lrs_bwd_kernel(
    # grads wrt outputs
    grad_k_perm_ptr,  # [B, N, H]
    grad_v_scaled_ptr,  # [B, N, H]
    grad_lr0_perm_ptr,  # [B, N]
    grad_lr1_perm_ptr,  # [B, N]
    grad_lr2_perm_ptr,  # [B, N]
    # grads wrt inputs (to write)
    grad_k_ptr,  # [B, T, H]
    grad_v_ptr,  # [B, T, H]
    grad_lr0_ptr,  # [B, T]
    grad_lr1_ptr,  # [B, T]
    grad_lr2_ptr,  # [B, T]
    probs_ptr,  # [B, E, T]
    probs_grad_ptr,  # [B, E, T]
    v_ptr,  # [B, T, H] (fwd input v)
    row_id_map_ptr,  # [B, T, 2*E+1]
    # sizes
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    # strides: row_id_map
    stride_row_id_map_batch,
    stride_row_id_map_token,
    stride_row_id_map_expert,
    # strides: grad_k_perm
    stride_grad_k_perm_batch,
    stride_grad_k_perm_token,
    stride_grad_k_perm_hidden,
    # strides: grad_v_scaled
    stride_grad_v_scaled_batch,
    stride_grad_v_scaled_token,
    stride_grad_v_scaled_hidden,
    # strides: grad_k
    stride_grad_k_batch,
    stride_grad_k_token,
    stride_grad_k_hidden,
    # strides: grad_v
    stride_grad_v_batch,
    stride_grad_v_token,
    stride_grad_v_hidden,
    # strides: grad_lr0_perm
    stride_grad_lr0_perm_batch,
    stride_grad_lr0_perm_token,
    # strides: grad_lr1_perm
    stride_grad_lr1_perm_batch,
    stride_grad_lr1_perm_token,
    # strides: grad_lr2_perm
    stride_grad_lr2_perm_batch,
    stride_grad_lr2_perm_token,
    # strides: grad_lr0
    stride_grad_lr0_batch,
    stride_grad_lr0_token,
    # strides: grad_lr1
    stride_grad_lr1_batch,
    stride_grad_lr1_token,
    # strides: grad_lr2
    stride_grad_lr2_batch,
    stride_grad_lr2_token,
    # strides: probs
    stride_probs_batch,
    stride_probs_expert,
    stride_probs_token,
    # strides: probs_grad
    stride_probs_grad_batch,
    stride_probs_grad_expert,
    stride_probs_grad_token,
    # strides: v
    stride_v_batch,
    stride_v_token,
    stride_v_hidden,
    # metas
    PROBS_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type_k = grad_k_ptr.dtype.element_ty
    data_type_v = grad_v_ptr.dtype.element_ty
    compute_type = tl.float32

    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    src_row = pid_t.to(tl.int64)

    # batch offsets
    grad_k_perm_ptr += pid_b * stride_grad_k_perm_batch
    grad_v_scaled_ptr += pid_b * stride_grad_v_scaled_batch
    grad_lr0_perm_ptr += pid_b * stride_grad_lr0_perm_batch
    grad_lr1_perm_ptr += pid_b * stride_grad_lr1_perm_batch
    grad_lr2_perm_ptr += pid_b * stride_grad_lr2_perm_batch
    grad_k_ptr += pid_b * stride_grad_k_batch
    grad_v_ptr += pid_b * stride_grad_v_batch
    grad_lr0_ptr += pid_b * stride_grad_lr0_batch
    grad_lr1_ptr += pid_b * stride_grad_lr1_batch
    grad_lr2_ptr += pid_b * stride_grad_lr2_batch
    probs_ptr += pid_b * stride_probs_batch
    probs_grad_ptr += pid_b * stride_probs_grad_batch
    v_ptr += pid_b * stride_v_batch
    row_id_map_ptr += pid_b * stride_row_id_map_batch

    # number of routed experts for this token
    n_routed = tl.load(
        row_id_map_ptr
        + pid_t * stride_row_id_map_token
        + (2 * num_experts) * stride_row_id_map_expert
    )

    # ---- lr grads (scalar, no H loop) ----
    lr0_acc = tl.zeros((), dtype=compute_type)
    lr1_acc = tl.zeros((), dtype=compute_type)
    lr2_acc = tl.zeros((), dtype=compute_type)

    for idx in tl.range(0, n_routed):
        dst_row = tl.load(
            row_id_map_ptr
            + pid_t * stride_row_id_map_token
            + idx * stride_row_id_map_expert
        ).to(tl.int64)

        g_lr0 = tl.load(grad_lr0_perm_ptr + dst_row * stride_grad_lr0_perm_token)
        g_lr1 = tl.load(grad_lr1_perm_ptr + dst_row * stride_grad_lr1_perm_token)
        g_lr2 = tl.load(grad_lr2_perm_ptr + dst_row * stride_grad_lr2_perm_token)

        lr0_acc += g_lr0.to(compute_type)
        lr1_acc += g_lr1.to(compute_type)
        lr2_acc += g_lr2.to(compute_type)

    tl.store(
        grad_lr0_ptr + src_row * stride_grad_lr0_token,
        lr0_acc.to(grad_lr0_ptr.dtype.element_ty),
    )
    tl.store(
        grad_lr1_ptr + src_row * stride_grad_lr1_token,
        lr1_acc.to(grad_lr1_ptr.dtype.element_ty),
    )
    tl.store(
        grad_lr2_ptr + src_row * stride_grad_lr2_token,
        lr2_acc.to(grad_lr2_ptr.dtype.element_ty),
    )

    # ---- probs grads ----
    off_e = tl.arange(0, PROBS_LOAD_WIDTH)
    probs_grad_accum = tl.zeros((PROBS_LOAD_WIDTH,), dtype=compute_type)

    # ---- k/v grads over hidden dimension ----
    current_start = 0
    while current_start < hidden_size:
        offs_h = current_start + tl.arange(0, BLOCK_SIZE)
        h_mask = offs_h < hidden_size

        grad_k_block = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        grad_v_block = tl.zeros((BLOCK_SIZE,), dtype=compute_type)

        # v_pre for this token & hidden block
        v_off = src_row * stride_v_token + offs_h * stride_v_hidden
        v_pre = tl.load(v_ptr + v_off, mask=h_mask).to(compute_type)

        for idx in tl.range(0, n_routed):
            dst_row = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + idx * stride_row_id_map_expert
            ).to(tl.int64)

            expert_idx = tl.load(
                row_id_map_ptr
                + pid_t * stride_row_id_map_token
                + (num_experts + idx) * stride_row_id_map_expert
            )

            # prob p(b,e,t)
            prob_off = pid_t * stride_probs_token + expert_idx * stride_probs_expert
            p = tl.load(probs_ptr + prob_off).to(compute_type)

            # grads from permuted outputs
            gk_off = (
                dst_row * stride_grad_k_perm_token + offs_h * stride_grad_k_perm_hidden
            )
            gv_off = (
                dst_row * stride_grad_v_scaled_token
                + offs_h * stride_grad_v_scaled_hidden
            )
            gk = tl.load(grad_k_perm_ptr + gk_off, mask=h_mask).to(compute_type)
            gv = tl.load(grad_v_scaled_ptr + gv_off, mask=h_mask).to(compute_type)

            # grad wrt k: just sum over all routed dst_rows
            grad_k_block += gk

            # grad wrt v: sum over gv * p
            grad_v_block += gv * p

            # grad wrt prob: sum_h gv * v_pre, but zero if p == 0 (padding)
            v_eff = tl.where(p == 0.0, 0.0, v_pre)
            local_contrib = tl.sum(gv * v_eff, axis=0)
            probs_grad_accum += tl.where(off_e == expert_idx, local_contrib, 0.0)

        # store k & v grads for this block
        grad_k_off = src_row * stride_grad_k_token + offs_h * stride_grad_k_hidden
        grad_v_off = src_row * stride_grad_v_token + offs_h * stride_grad_v_hidden

        tl.store(
            grad_k_ptr + grad_k_off,
            grad_k_block.to(data_type_k),
            mask=h_mask,
        )
        tl.store(
            grad_v_ptr + grad_v_off,
            grad_v_block.to(data_type_v),
            mask=h_mask,
        )

        current_start += BLOCK_SIZE

    # store probs grads for this token
    probs_grad_off = (
        src_row * stride_probs_grad_token + off_e * stride_probs_grad_expert
    )
    tl.store(
        probs_grad_ptr + probs_grad_off,
        probs_grad_accum.to(probs_grad_ptr.dtype.element_ty),
        mask=off_e < num_experts,
    )


class PermuteKVAndLRs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        k: torch.Tensor,  # [B, T, H]
        v: torch.Tensor,  # [B, T, H]
        lr0: torch.Tensor,  # [B, T] or [B, T, 1]
        lr1: torch.Tensor,
        lr2: torch.Tensor,
        expert_mask: torch.Tensor,  # [B, E, T], 0/1 or bool
        probs: torch.Tensor,  # [B, E, T], fp32
        topk: int,
    ):

        B, T, H = k.shape
        E = expert_mask.shape[1]
        assert expert_mask.shape == (B, E, T)
        assert probs.shape == (B, E, T)

        # normalize lr shapes to [B, T]
        lr_was_3d = lr0.ndim == 3
        if lr0.ndim == 3:
            lr0_2d = lr0.squeeze(-1)
            lr1_2d = lr1.squeeze(-1)
            lr2_2d = lr2.squeeze(-1)
        else:
            lr0_2d, lr1_2d, lr2_2d = lr0, lr1, lr2

        assert lr0_2d.shape == (B, T)
        assert lr1_2d.shape == (B, T)
        assert lr2_2d.shape == (B, T)

        # build row_id_map from mask once
        routing_map = expert_mask.to(torch.int32)
        row_id_map = make_row_id_map(routing_map, B, T, E)

        # assuming exact top-k routing per token
        num_out_tokens = topk * T

        # outputs
        out_k = torch.empty((B, num_out_tokens, H), dtype=k.dtype, device=k.device)
        out_v_scaled = torch.empty_like(out_k)  # same dtype as k
        out_lr0 = torch.empty(
            (B, num_out_tokens), dtype=lr0_2d.dtype, device=lr0_2d.device
        )
        out_lr1 = torch.empty_like(out_lr0)
        out_lr2 = torch.empty_like(out_lr0)

        # launch grid: (B, T, H-tiles)
        grid = lambda META: (
            B,
            T,
            triton.cdiv(H, META["BLOCK_SIZE"]),
        )

        _permute_kv_with_lrs_kernel[grid](
            # inputs
            k,
            v,
            lr0_2d,
            lr1_2d,
            lr2_2d,
            row_id_map,
            probs,
            # outputs
            out_k,
            out_v_scaled,
            out_lr0,
            out_lr1,
            out_lr2,
            # sizes
            E,
            H,
            # strides: row_id_map
            row_id_map.stride(0),
            row_id_map.stride(1),
            row_id_map.stride(2),
            # strides: k
            k.stride(0),
            k.stride(1),
            k.stride(2),
            # strides: v
            v.stride(0),
            v.stride(1),
            v.stride(2),
            # strides: lr0
            lr0_2d.stride(0),
            lr0_2d.stride(1),
            # strides: lr1
            lr1_2d.stride(0),
            lr1_2d.stride(1),
            # strides: lr2
            lr2_2d.stride(0),
            lr2_2d.stride(1),
            # strides: out_k
            out_k.stride(0),
            out_k.stride(1),
            out_k.stride(2),
            # strides: out_v_scaled
            out_v_scaled.stride(0),
            out_v_scaled.stride(1),
            out_v_scaled.stride(2),
            # strides: out_lr0
            out_lr0.stride(0),
            out_lr0.stride(1),
            # strides: out_lr1
            out_lr1.stride(0),
            out_lr1.stride(1),
            # strides: out_lr2
            out_lr2.stride(0),
            out_lr2.stride(1),
            # strides: probs
            probs.stride(0),
            probs.stride(1),
            probs.stride(2),
        )

        # save context
        ctx.batch_size = B
        ctx.num_tokens = T
        ctx.num_experts = E
        ctx.hidden_size = H
        ctx.lr_was_3d = lr_was_3d
        ctx.topk = topk

        # we need v (for prob grad), probs, and row_id_map
        ctx.save_for_backward(v, probs, row_id_map)

        return out_k, out_v_scaled, out_lr0, out_lr1, out_lr2

    @staticmethod
    def backward(
        ctx, grad_out_k, grad_out_v_scaled, grad_out_lr0, grad_out_lr1, grad_out_lr2
    ):
        B = ctx.batch_size
        T = ctx.num_tokens
        E = ctx.num_experts
        H = ctx.hidden_size
        lr_was_3d = ctx.lr_was_3d

        v, probs, row_id_map = ctx.saved_tensors
        device = grad_out_k.device

        # allocate grads wrt inputs
        grad_k = torch.empty((B, T, H), dtype=grad_out_k.dtype, device=device)
        grad_v = torch.empty((B, T, H), dtype=v.dtype, device=v.device)
        grad_lr0 = torch.empty(
            (B, T), dtype=grad_out_lr0.dtype, device=grad_out_lr0.device
        )
        grad_lr1 = torch.empty_like(grad_lr0)
        grad_lr2 = torch.empty_like(grad_lr0)
        grad_probs = torch.empty_like(probs)

        grid = lambda META: (
            B,
            T,
        )

        _permute_kv_with_lrs_bwd_kernel[grid](
            # grads wrt outputs
            grad_out_k,
            grad_out_v_scaled,
            grad_out_lr0,
            grad_out_lr1,
            grad_out_lr2,
            # grads wrt inputs (to write)
            grad_k,
            grad_v,
            grad_lr0,
            grad_lr1,
            grad_lr2,
            probs,
            grad_probs,
            v,
            row_id_map,
            # sizes
            ctx.num_experts,
            ctx.hidden_size,
            # strides: row_id_map
            row_id_map.stride(0),
            row_id_map.stride(1),
            row_id_map.stride(2),
            # strides: grad_k_perm
            grad_out_k.stride(0),
            grad_out_k.stride(1),
            grad_out_k.stride(2),
            # strides: grad_v_scaled
            grad_out_v_scaled.stride(0),
            grad_out_v_scaled.stride(1),
            grad_out_v_scaled.stride(2),
            # strides: grad_k
            grad_k.stride(0),
            grad_k.stride(1),
            grad_k.stride(2),
            # strides: grad_v
            grad_v.stride(0),
            grad_v.stride(1),
            grad_v.stride(2),
            # strides: grad_lr0_perm
            grad_out_lr0.stride(0),
            grad_out_lr0.stride(1),
            # strides: grad_lr1_perm
            grad_out_lr1.stride(0),
            grad_out_lr1.stride(1),
            # strides: grad_lr2_perm
            grad_out_lr2.stride(0),
            grad_out_lr2.stride(1),
            # strides: grad_lr0
            grad_lr0.stride(0),
            grad_lr0.stride(1),
            # strides: grad_lr1
            grad_lr1.stride(0),
            grad_lr1.stride(1),
            # strides: grad_lr2
            grad_lr2.stride(0),
            grad_lr2.stride(1),
            # strides: probs
            probs.stride(0),
            probs.stride(1),
            probs.stride(2),
            # strides: probs_grad
            grad_probs.stride(0),
            grad_probs.stride(1),
            grad_probs.stride(2),
            # strides: v
            v.stride(0),
            v.stride(1),
            v.stride(2),
            PROBS_LOAD_WIDTH=triton.next_power_of_2(E),
        )

        # restore lr grad shape if lr were [B, T, 1]
        if lr_was_3d:
            grad_lr0 = grad_lr0.unsqueeze(-1)
            grad_lr1 = grad_lr1.unsqueeze(-1)
            grad_lr2 = grad_lr2.unsqueeze(-1)

        # inputs: k, v, lr0, lr1, lr2, expert_mask, probs, topk
        grad_expert_mask = None  # mask is discrete
        grad_topk = None

        return (
            grad_k,
            grad_v,
            grad_lr0,
            grad_lr1,
            grad_lr2,
            grad_expert_mask,
            grad_probs,
            grad_topk,
        )


def permute_kv_and_lrs(
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    expert_mask: torch.Tensor,
    probs: torch.Tensor,
    topk: int,
):
    return PermuteKVAndLRs.apply(k, v, lr0, lr1, lr2, expert_mask, probs, topk)


########################################################
# Reference implementation and correctness test
########################################################


def _reference_permute(inp, row_id_map, probs=None):
    # inp: [B, T, H], row_id_map: [B, T, 2E+1], probs: [B, E, T] or None
    B, T, H = inp.shape
    E2p1 = row_id_map.shape[-1]
    assert E2p1 % 2 == 1
    E = (E2p1 - 1) // 2

    n_routed_per_token = row_id_map[..., 2 * E]  # [B, T]
    n_out_tokens_per_batch = torch.sum(n_routed_per_token, dim=1)  # [B]
    num_out_tokens = int(torch.max(n_out_tokens_per_batch).item())

    out = torch.empty((B, num_out_tokens, H), dtype=inp.dtype, device=inp.device)
    out[:] = 0
    perm_probs = None
    if probs is not None:
        perm_probs = torch.empty(
            (B, num_out_tokens), dtype=probs.dtype, device=probs.device
        )
        perm_probs[:] = 0

    for b in range(B):
        for t in range(T):
            k = int(row_id_map[b, t, 2 * E].item())
            for j in range(k):
                dst = int(row_id_map[b, t, j].item())
                e = int(row_id_map[b, t, E + j].item())
                if probs is None:
                    out[b, dst, :] = inp[b, t, :]
                else:
                    p = probs[b, e, t]
                    perm_probs[b, dst] = p
                    if float(p.item()) == 0.0:
                        out[b, dst, :] = 0
                    else:
                        out[b, dst, :] = inp[b, t, :]
    return out, perm_probs, num_out_tokens


def _reference_permute_kv_and_lrs(
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    expert_mask: torch.Tensor,
    probs: torch.Tensor,
    topk: int,
):
    k_permuted, k_permuted_router_probs, k_row_id_map = permute_with_expert_mask(
        k, expert_mask, probs, topk, None
    )

    v_permuted, v_permuted_router_probs, _ = permute_with_expert_mask(
        v, expert_mask, probs, topk, k_row_id_map
    )

    v_permuted = (v_permuted * k_permuted_router_probs.unsqueeze(dim=-1)).to(
        k_permuted.dtype
    )

    lr0i_permuted, _, _ = permute_with_expert_mask(
        lr0.unsqueeze(-1), expert_mask, None, topk, k_row_id_map
    )
    lr1i_permuted, _, _ = permute_with_expert_mask(
        lr1.unsqueeze(-1), expert_mask, None, topk, k_row_id_map
    )
    lr2i_permuted, _, _ = permute_with_expert_mask(
        lr2.unsqueeze(-1), expert_mask, None, topk, k_row_id_map
    )
    lr0i_permuted = lr0i_permuted.squeeze(dim=-1)
    lr1i_permuted = lr1i_permuted.squeeze(dim=-1)
    lr2i_permuted = lr2i_permuted.squeeze(dim=-1)

    return (
        k_permuted,
        v_permuted,
        lr0i_permuted,
        lr1i_permuted,
        lr2i_permuted,
    )


def _make_inputs_for_permute_kv_and_lrs(B, T, E, H, topK, require_grad=True):
    """
    k: torch.Tensor,  # [B, T, H]
    v: torch.Tensor,  # [B, T, H]
    lr0: torch.Tensor,  # [B, T] or [B, T, 1]
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    expert_mask: torch.Tensor,  # [B, E, T], 0/1 or bool
    probs: torch.Tensor,  # [B, E, T], fp32
    topk: int,
    """

    device = "cuda"
    k = torch.randn(
        B, T, H, device=device, dtype=torch.bfloat16, requires_grad=require_grad
    )
    v = torch.randn(
        B, T, H, device=device, dtype=torch.bfloat16, requires_grad=require_grad
    )
    lr0 = torch.randn(
        B, T, device=device, dtype=torch.float32, requires_grad=require_grad
    )
    lr1 = torch.randn(
        B, T, device=device, dtype=torch.float32, requires_grad=require_grad
    )
    lr2 = torch.randn(
        B, T, device=device, dtype=torch.float32, requires_grad=require_grad
    )
    probs = torch.rand(
        B, E, T, device=device, dtype=torch.float32, requires_grad=require_grad
    )

    with torch.no_grad():
        values, _ = torch.topk(probs, k=topK, dim=1)
        value_threshold = values[
            :,
            [-1],
        ]
        expert_mask = (probs >= value_threshold).to(torch.bool)
    return k, v, lr0, lr1, lr2, expert_mask, probs, topK


def report_error(ref, ours, name=""):
    print(
        f"{name} - Shape: {ref.shape} Ours shape: {ours.shape} Ref dtype: {ref.dtype} Ours dtype: {ours.dtype}"
    )

    abs_error = torch.abs(ref - ours)
    abs_ref = torch.abs(ref)
    abs_ours = torch.abs(ours)

    error_rel = abs_error / (1e-4 + torch.maximum(abs_ref, abs_ours))

    mean_error_rel = error_rel.mean()
    max_error_rel = error_rel.max()

    top_k_error_rel = torch.topk(error_rel, k=10).values
    top_k_error_mean = top_k_error_rel.mean()

    mean_abs_error = abs_error.mean()
    max_abs_error = abs_error.max()
    mean_abs_ref = abs_ref.mean()
    max_abs_ref = abs_ref.max()
    mean_abs_ours = abs_ours.mean()
    max_abs_ours = abs_ours.max()

    print(
        f"{name} - Mean Relative Error: {mean_error_rel}, Max Relative Error: {max_error_rel} Top 10 Relative Error: {top_k_error_mean}"
    )
    print(
        f"{name} - Mean Absolute Error: {mean_abs_error}, Max Absolute Error: {max_abs_error}"
    )
    print(
        f"{name} - Mean Reference Signal Magnitude: {mean_abs_ref}, Max Reference Signal Magnitude: {max_abs_ref}"
    )
    print(
        f"{name} - Mean Ours Signal Magnitude: {mean_abs_ours}, Max Ours Signal Magnitude: {max_abs_ours}"
    )
    print("--------------------------------")


def test_permute_kv_and_lrs_forward():
    B, T, E, H, topK = 4, 3000, 4, 900, 2
    k, v, lr0, lr1, lr2, expert_mask, probs, topK = _make_inputs_for_permute_kv_and_lrs(
        B, T, E, H, topK, require_grad=True
    )
    ref_k = k.clone().detach().requires_grad_(True)
    ref_v = v.clone().detach().requires_grad_(True)
    ref_lr0 = lr0.clone().detach().requires_grad_(True)
    ref_lr1 = lr1.clone().detach().requires_grad_(True)
    ref_lr2 = lr2.clone().detach().requires_grad_(True)
    ref_probs = probs.clone().detach().requires_grad_(True)
    out_k, out_v, out_lr0, out_lr1, out_lr2 = permute_kv_and_lrs(
        k, v, lr0, lr1, lr2, expert_mask, probs, topK
    )
    ref_out_k, ref_out_v, ref_out_lr0, ref_out_lr1, ref_out_lr2 = (
        _reference_permute_kv_and_lrs(
            ref_k, ref_v, ref_lr0, ref_lr1, ref_lr2, expert_mask, ref_probs, topK
        )
    )
    report_error(ref_out_k, out_k, "out_k")
    report_error(ref_out_v, out_v, "out_v")
    report_error(ref_out_lr0, out_lr0, "out_lr0")
    report_error(ref_out_lr1, out_lr1, "out_lr1")
    report_error(ref_out_lr2, out_lr2, "out_lr2")

    ### bwd

    loss = (
        out_k.sum() * 1.0
        + out_v.sum() * 2.0
        + out_lr0.sum() * 3.0
        + out_lr1.sum() * 5.0
        + out_lr2.sum() * -2.0
    )
    loss.backward()

    loss_ref = (
        ref_out_k.sum() * 1.0
        + ref_out_v.sum() * 2.0
        + ref_out_lr0.sum() * 3.0
        + ref_out_lr1.sum() * 5.0
        + ref_out_lr2.sum() * -2.0
    )
    loss_ref.backward()

    report_error(k.grad, ref_k.grad, "k.grad")
    report_error(v.grad, ref_v.grad, "v.grad")
    report_error(lr0.grad, ref_lr0.grad, "lr0.grad")
    report_error(lr1.grad, ref_lr1.grad, "lr1.grad")
    report_error(lr2.grad, ref_lr2.grad, "lr2.grad")
    report_error(probs.grad, ref_probs.grad, "probs.grad")


def test_permute_batched_random(B, T, E, H, topK):
    torch.manual_seed(123)
    device = "cuda"

    max_experts = min(topK, E - 1)
    print(f"max_experts: {max_experts}")

    routing_map = torch.rand(B, E, T, device=device)
    # [B, max_experts, T]
    values, topk = torch.topk(routing_map, k=max_experts, dim=1)
    value_threshold = values[
        :,
        [-1],
    ]
    routing_map = (routing_map >= value_threshold).to(torch.int32)
    # make mark topk as zero and ones

    row_id_map = make_row_id_map(routing_map, B, T, E)

    inp = torch.randn(B, T, H, device=device, dtype=torch.float16)
    probs = torch.rand(B, E, T, device=device, dtype=torch.float32)

    routed_pos = torch.nonzero(routing_map == 1, as_tuple=False)
    if routed_pos.numel() > 0:
        # randomly set 10%  to 0
        k = max(1, routed_pos.shape[0] // 10)
        idx = torch.randperm(routed_pos.shape[0], device=device)[:k]
        subset = routed_pos[idx]
        for b, e, t in subset:
            probs[int(b), int(e), int(t)] = 0.0

    ref_out, ref_perm_probs, num_out_tokens = _reference_permute(
        inp, row_id_map, probs=probs
    )

    out, perm_probs, row_id_map2 = permute_with_expert_mask(
        inp=inp,
        expert_mask=routing_map,
        probs=probs,
        topk=max_experts,
    )

    report_error(ref_out, out, "out")
    report_error(ref_perm_probs, perm_probs, "perm_probs")

    # torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)
    # torch.testing.assert_close(perm_probs, ref_perm_probs, atol=0, rtol=0)
