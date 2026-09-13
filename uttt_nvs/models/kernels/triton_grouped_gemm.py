import torch
import triton
import triton.language as tl
import itertools


def get_autotune_configs(
    block_M_list=(64, 128),
    block_N_list=(64, 128, 256),
    block_K_list=(32, 64),
    num_stages_list=(2, 3),
    threads_list=(128, 256),
):
    configs = []
    for BM, BN, BK, stages, threads in itertools.product(
        block_M_list, block_N_list, block_K_list, num_stages_list, threads_list
    ):
        configs.append(
            triton.Config(
                {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                num_warps=threads // 32,
                num_stages=stages,
            )
        )
    return configs


def get_autotune_configs_small(
    block_M_list=(64, 128),
    block_N_list=(128, 256),
    block_K_list=(32, 64),
    num_stages_list=(2, 3),
    threads_list=(256,),
):
    configs = []

    for BM, BN, BK, stages, threads in itertools.product(
        block_M_list, block_N_list, block_K_list, num_stages_list, threads_list
    ):
        configs.append(
            triton.Config(
                {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                num_warps=threads // 32,
                num_stages=stages,
            )
        )
    return configs


@triton.autotune(
    configs=get_autotune_configs_small(),
    key=["M", "E", "K"],
)
@triton.jit
def _grouped_gemm_transpose_fwd_kernel(
    w1_ptr,  # [B, E, M, K]
    x_ptr,  # [B, K, N]
    group_sizes_ptr,  # [B, E]
    y_ptr,  # [B, N, M]  <-- transposed output layout
    B: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    N,  # runtime N
    K: tl.constexpr,
    # strides (in elements)
    stride_w_b,
    stride_w_e,
    stride_w_m,
    stride_w_k,
    stride_x_b,
    stride_x_k,
    stride_x_n,
    stride_y_b,
    stride_y_n,
    stride_y_m,
    # tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_bn = tl.program_id(axis=0)  # tiles over N (grouped across all (b,e))
    pid_m = tl.program_id(axis=1)  # tiles over M

    # Decode pid_bn -> (batch_id, expert_id, pid_n_in_group)
    pid_n = pid_bn
    pid_b = 0
    pid_e = 0

    # We'll compute the global N start index for this tile; also need the group's end for masking
    n_tile_start_index = 0
    n_tile_group_end_idx = N

    hit = False
    group_id_total = 0
    while (not hit) and (group_id_total < B * E):
        b = group_id_total // E
        e = group_id_total % E

        gsize = tl.load(group_sizes_ptr + b * E + e)
        tiles_in_group = tl.cdiv(gsize, BLOCK_N)

        # Reset per-batch running offset at the start of each new batch
        if e == 0:
            n_tile_start_index = 0

        if pid_n < tiles_in_group:
            pid_b = b
            pid_e = e
            hit = True
            n_tile_group_end_idx = n_tile_start_index + gsize
        else:
            pid_n -= tiles_in_group
            n_tile_start_index += gsize

        group_id_total += 1

    # Tile's starting column in the (permuted) N dimension (within this (b,e) group)
    n_tile_start_index += pid_n * BLOCK_N

    # Early exit if we somehow launched extra tiles (shouldn't happen if grid is exact)
    if n_tile_start_index >= N:
        return

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = n_tile_start_index + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)  # [BK]

    # N mask (respect the group's true size and batch-wide N)
    n_mask_1d = (offs_n < n_tile_group_end_idx) & (offs_n < N)

    # Base pointers for this (b, e)
    w1_be_ptr = w1_ptr + pid_b * stride_w_b + pid_e * stride_w_e
    x_b_ptr = x_ptr + pid_b * stride_x_b
    y_b_ptr = y_ptr + pid_b * stride_y_b

    # Accumulator (fp32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k
        k_mask = k_ids < K

        # Load W1 tile: (BM, BK)
        w_ptrs = (
            w1_be_ptr + (offs_m[:, None] * stride_w_m) + (k_ids[None, :] * stride_w_k)
        )
        w_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        w = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.bfloat16)

        # Load X tile: (BK, BN)
        x_ptrs = (
            x_b_ptr + (k_ids[:, None] * stride_x_k) + (offs_n[None, :] * stride_x_n)
        )
        x_mask = (k_mask[:, None]) & (n_mask_1d[None, :])
        x = tl.load(x_ptrs, mask=x_mask, other=0).to(tl.bfloat16)

        # (BM,BK) x (BK,BN) -> (BM,BN)
        acc += tl.dot(w, x, out_dtype=tl.float32)

    # Store directly into Y[b] of shape [N, M] using (m,n) addressing:
    # address = base + n * stride_y_n + m * stride_y_m
    y_ptrs = y_b_ptr + (offs_m[:, None] * stride_y_m) + (offs_n[None, :] * stride_y_n)
    o_mask = (offs_m[:, None] < M) & (n_mask_1d[None, :])
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=o_mask)


def grouped_gemm_transpose_triton(
    W1: torch.Tensor,  # [B, E, M, K], bf16, contiguous
    X: torch.Tensor,  # [B, K, N],   bf16, contiguous
    group_sizes: torch.Tensor,  # [B, E], int
) -> torch.Tensor:
    """
    Computes Y = (W1 @ X).transpose(1, 2) in a grouped fashion, returning Y as [B, N, M].
    Assumes that within each batch, columns (tokens) of X are permuted such that each expert's
    tokens are contiguous and their counts are given by group_sizes[b, e].
    """
    assert W1.dtype == torch.bfloat16 and X.dtype == torch.bfloat16
    assert W1.is_contiguous() and X.is_contiguous(), "W1 and X must be contiguous"
    assert W1.device == X.device == group_sizes.device

    B, E, M, K = W1.shape
    Bx, Kx, N = X.shape
    assert Bx == B and Kx == K, "X must be [B, K, N] with matching B,K."

    # Optional: sanity check (can be commented out for performance)
    # Ensure each batch's group sizes sum to N
    # if not torch.all(group_sizes.sum(dim=1) == N):
    #     raise ValueError("For each batch b, sum_e group_sizes[b, e] must equal N.")

    # Output is the transposed layout directly: [B, N, M]
    Y = torch.empty((B, N, M), device=X.device, dtype=torch.bfloat16)

    # Strides (in elements)
    stride_w_b, stride_w_e, stride_w_m, stride_w_k = W1.stride()
    stride_x_b, stride_x_k, stride_x_n = X.stride()
    stride_y_b, stride_y_n, stride_y_m = Y.stride()

    # Exact grid size along N-tiles: sum over all (b,e) of ceil(group_sizes[b,e] / BLOCK_N)
    def grid(meta):
        BN = meta["BLOCK_N"]
        # Integer ceil-div per group, summed across all batches/experts.
        # Works for group_sizes on either CPU or GPU; .item() syncs once.
        # total_n_tiles = int(((group_sizes + BN - 1) // BN).sum().item())
        # total_n_tiles = int(((group_sizes + BN - 1) // BN).sum().item())
        total_n_tiles = (triton.cdiv(N, BN) + E - 1) * B
        return (total_n_tiles, triton.cdiv(M, meta["BLOCK_M"]))

    _grouped_gemm_transpose_fwd_kernel[grid](
        W1,
        X,
        group_sizes,
        Y,
        B,
        E,
        M,
        N,
        K,
        stride_w_b,
        stride_w_e,
        stride_w_m,
        stride_w_k,
        stride_x_b,
        stride_x_k,
        stride_x_n,
        stride_y_b,
        stride_y_n,
        stride_y_m,
    )
    return Y


################################################################################
# Backward Kernels
################################################################################


@triton.autotune(
    configs=get_autotune_configs_small(),
    key=["H", "E", "D"],
)
@triton.jit
def _grouped_gemm_bwd_kernel(
    dy_ptr,  # [B, H, N]          (bf16)
    x_ptr,  # [B, N, D]          (bf16)
    group_sizes_ptr,  # [B, E]             (int)
    wgrad_ptr,  # [B, E, H, D]       (bf16)  <-- output
    B: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,
    N,  # runtime N (sum of groups per batch)
    D: tl.constexpr,
    # strides (elements)
    stride_dy_b,
    stride_dy_h,
    stride_dy_n,
    stride_x_b,
    stride_x_n,
    stride_x_d,
    stride_wg_b,
    stride_wg_e,
    stride_wg_h,
    stride_wg_d,
    # tiling
    BLOCK_M: tl.constexpr,  # along H (rows of W_grad)
    BLOCK_N: tl.constexpr,  # along D (cols of W_grad)
    BLOCK_K: tl.constexpr,  # along N (reduction over tokens)
):
    pid_h = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    pid_be = tl.program_id(axis=2)

    pid_b = pid_be // E
    pid_e = pid_be % E

    # Compute the (group) token segment [n_start, n_start + g_size)
    # by scanning group_sizes[b, :]
    n_start = tl.zeros((), dtype=tl.int32)
    g_size = tl.zeros((), dtype=tl.int32)
    for e0 in range(E):
        sz = tl.load(group_sizes_ptr + pid_b * E + e0).to(tl.int32)
        n_start += tl.where(e0 < pid_e, sz, 0)
        g_size = tl.where(e0 == pid_e, sz, g_size)

    # Offsets along output H and D tiles
    offs_h = pid_h * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_d = pid_d * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)  # [BK] over N

    h_mask = offs_h < H
    d_mask = offs_d < D

    # Base pointers per-batch
    dy_b_ptr = dy_ptr + pid_b * stride_dy_b  # [H, N]
    x_b_ptr = x_ptr + pid_b * stride_x_b  # [N, D]
    wg_be_ptr = wgrad_ptr + pid_b * stride_wg_b + pid_e * stride_wg_e  # [H, D]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over the token segment for this (b,e)
    k_end = n_start + g_size
    # Iterate k0 from n_start .. k_end in steps of BLOCK_K
    k0 = n_start
    while k0 < k_end:
        k_ids = k0 + offs_k
        k_mask = (k_ids < k_end) & (k_ids < N)

        # Load dy tile: (BM, BK) from [H, N]
        dy_ptrs = (
            dy_b_ptr + (offs_h[:, None] * stride_dy_h) + (k_ids[None, :] * stride_dy_n)
        )
        dy_mask = (h_mask[:, None]) & (k_mask[None, :])
        dy = tl.load(dy_ptrs, mask=dy_mask, other=0).to(tl.bfloat16)

        # Load x tile: (BK, BN) from [N, D]
        x_ptrs = (
            x_b_ptr + (k_ids[:, None] * stride_x_n) + (offs_d[None, :] * stride_x_d)
        )
        x_mask2 = (k_mask[:, None]) & (d_mask[None, :])
        x = tl.load(x_ptrs, mask=x_mask2, other=0).to(tl.bfloat16)

        # (H x N) @ (N x D) -> (H x D)
        acc += tl.dot(dy, x, out_dtype=tl.float32)

        k0 += BLOCK_K

    # Store (H, D) tile into wgrad[b,e]
    out_ptrs = (
        wg_be_ptr + (offs_h[:, None] * stride_wg_h) + (offs_d[None, :] * stride_wg_d)
    )
    o_mask = (h_mask[:, None]) & (d_mask[None, :])
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=o_mask)


def grouped_gemm_bwd_triton(
    output_grad: torch.Tensor,  # [B, H, N], bf16, contiguous
    X: torch.Tensor,  # [B, N, D], bf16, contiguous
    group_sizes: torch.Tensor,  # [B, E],    int
) -> torch.Tensor:
    """
    Computes W_grad[b,e] = output_grad[b, :, seg(b,e)] @ X[b, seg(b,e), :]
    where seg(b,e) is the contiguous token block defined by group_sizes[b, e].
    Returns W_grad in shape [B, E, H, D] (bf16).
    """
    assert output_grad.dtype == torch.bfloat16 and X.dtype == torch.bfloat16

    B, H, N = output_grad.shape
    _, _, D = X.shape
    B, E = group_sizes.shape

    W_grad = torch.empty((B, E, H, D), device=X.device, dtype=torch.bfloat16)

    stride_dy_b, stride_dy_h, stride_dy_n = H * N, N, 1
    stride_x_b, stride_x_n, stride_x_d = N * D, D, 1
    stride_wg_b, stride_wg_e, stride_wg_h, stride_wg_d = (
        E * H * D,
        H * D,
        D,
        1,
    )

    def grid(meta):
        return (
            triton.cdiv(H, meta["BLOCK_M"]),
            triton.cdiv(D, meta["BLOCK_N"]),
            B * E,
        )

    _grouped_gemm_bwd_kernel[grid](
        output_grad,
        X,
        group_sizes,
        W_grad,
        B,
        E,
        H,
        N,
        D,
        stride_dy_b,
        stride_dy_h,
        stride_dy_n,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_wg_b,
        stride_wg_e,
        stride_wg_h,
        stride_wg_d,
    )
    return W_grad


def reference_grouped_gemm_transpose(W1, X, group_sizes):
    B, E, M, K = W1.shape
    _, Kx, N = X.shape
    assert Kx == K
    Y = torch.empty((B, N, M), device=W1.device, dtype=W1.dtype)
    for b in range(B):
        n_start = 0
        for e in range(E):
            g = int(group_sizes[b, e].item())
            if g == 0:
                continue
            # W1[b,e]: (M,K), X[b,:, n_start:n_start+g]: (K,g)
            block = (W1[b, e] @ X[b, :, n_start : n_start + g]).to(W1.dtype)
            # Transposed storage: [N, M]
            Y[b, n_start : n_start + g, :] = block.transpose(0, 1).contiguous()
            n_start += g
    return Y


def reference_grouped_gemm_bwd(output_grad, X, group_sizes):
    B, H, N = output_grad.shape
    _, _, D = X.shape
    B, E = group_sizes.shape
    W_grad = torch.empty((B, E, H, D), device=X.device, dtype=X.dtype)
    for b in range(B):
        s_index = 0
        for e in range(E):
            g = int(group_sizes[b, e].item())

            # output_grad[b, :, s_index:s_index+g]: (H, g)
            # X[b, s_index:s_index+g, :]: (g, D)
            block = (
                output_grad[b, :, s_index : s_index + g]
                @ X[b, s_index : s_index + g, :]
            ).to(X.dtype)
            W_grad[b, e] = block.contiguous()
            s_index += g
    return W_grad


def test_correctness():

    from benchmark import report_error

    def make_inputs(B, E, D, H, N):
        W1 = torch.randn(B, E, D, H, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(B, H, N, device="cuda", dtype=torch.bfloat16)

        group_sizes = torch.randint(1, N // E, (B, E), device="cuda", dtype=torch.int32)

        sum_group_sizes = group_sizes.sum(dim=1)

        extra_size = N - sum_group_sizes
        group_sizes[:, -1] += extra_size

        return W1, x, group_sizes

    B, E, D, H, N = 4, 4, 1024, 1024, 4096
    _inputs = make_inputs(B, E, D, H, N)
    fp32_inputs = [
        _inputs[0].to(torch.float32),
        _inputs[1].to(torch.float32),
        _inputs[2],
    ]
    ref_y = reference_grouped_gemm_transpose(*fp32_inputs)
    triton_y = grouped_gemm_transpose_triton(*_inputs)
    report_error(ref_y, triton_y, "output")


def test_correctness_bwd():
    from benchmark import report_error

    def make_inputs(B, E, D, H, N):
        output_grad = torch.randn(B, H, N, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
        group_sizes = torch.randint(1, N // E, (B, E), device="cuda", dtype=torch.int32)

        sum_group_sizes = group_sizes.sum(dim=1)

        extra_size = N - sum_group_sizes
        group_sizes[:, -1] += extra_size

        return output_grad, x, group_sizes

    B, E, D, H, N = 4, 4, 1024, 1024, 4096
    _inputs = make_inputs(B, E, D, H, N)

    fp32_inputs = [
        _inputs[0].to(torch.float32),
        _inputs[1].to(torch.float32),
        _inputs[2],
    ]
    ref_w_grad = reference_grouped_gemm_bwd(*fp32_inputs)
    triton_w_grad = grouped_gemm_bwd_triton(*_inputs)
    report_error(ref_w_grad, triton_w_grad, "w_grad")


if __name__ == "__main__":
    # test_correctness()
    test_correctness_bwd()
