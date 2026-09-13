import torch
import triton
import triton.language as tl
import itertools


# -----------------------
# Autotune configurations
# -----------------------
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
    block_N_list=(64, 128, 256),
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


# --------------------------------------------
# Fused kernel: computes DW0_W2 and DW1 together
# --------------------------------------------
@triton.autotune(
    configs=get_autotune_configs_small(),
    key=["H2", "D"],  # tune on row extent (2H) and D
)
@triton.jit
def _two_grouped_gemm_bwd_kernel(
    # Task 0 (DW0_W2): dy02 @ K
    dy02_ptr,  # [B, 2H, N]  (bf16)
    k_ptr,  # [B, N, D]   (bf16)
    # Task 1 (DW1): hidden @ V  (then store transposed to [D,H])
    dy1_ptr,  # [B, H, N]   (bf16)  -- "Hidden"
    v_ptr,  # [B, N, D]   (bf16)
    # Group partition info (shared)
    group_sizes_ptr,  # [B, E]      (int32)
    # Outputs
    w02_ptr,  # [B, E, 2H, D]  (bf16)
    w1_ptr,  # [B, E, D, H]   (bf16)
    # Shapes
    B,  # runtime (not used in compile-time loops)
    E: tl.constexpr,  # number of experts (compile-time for scan)
    H: tl.constexpr,  # Hidden
    H2: tl.constexpr,  # 2*Hidden
    N,  # total tokens per batch (runtime)
    D: tl.constexpr,  # feature dim
    # Strides (in elements)
    stride_dy02_b,
    stride_dy02_h2,
    stride_dy02_n,
    stride_k_b,
    stride_k_n,
    stride_k_d,
    stride_dy1_b,
    stride_dy1_h,
    stride_dy1_n,
    stride_v_b,
    stride_v_n,
    stride_v_d,
    stride_w02_b,
    stride_w02_e,
    stride_w02_h2,
    stride_w02_d,
    stride_w1_b,
    stride_w1_e,
    stride_w1_d,
    stride_w1_h,
    # Tiling
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)  # rows (H-like)
    pid_d = tl.program_id(axis=1)  # cols (D)
    pid_t = tl.program_id(axis=2)  # packs (b,e,task)

    # Decode (b, e, task) from pid_t
    # We pack as: pid_t = (b * E + e) * 2 + task
    bid = pid_t // 2
    task = pid_t % 2  # 0 -> DW0_W2 (dy02 @ K), 1 -> DW1 (hidden @ V)
    pid_b = bid // E
    pid_e = bid % E

    # Compute the segment [n_start, n_start + g_size) for this (b,e)
    n_start = tl.zeros((), dtype=tl.int32)
    g_size = tl.zeros((), dtype=tl.int32)
    for e0 in range(E):
        sz = tl.load(group_sizes_ptr + pid_b * E + e0).to(tl.int32)
        n_start += tl.where(e0 < pid_e, sz, 0)
        g_size = tl.where(e0 == pid_e, sz, g_size)
    k_end = n_start + g_size

    # Common offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along H-like rows
    offs_d = pid_d * BLOCK_N + tl.arange(0, BLOCK_N)  # along D
    offs_k = tl.arange(0, BLOCK_K)

    d_mask = offs_d < D

    # Base pointers per-batch
    # Task 0 bases
    dy02_b_ptr = dy02_ptr + pid_b * stride_dy02_b  # [2H, N]
    k_b_ptr = k_ptr + pid_b * stride_k_b  # [N, D]
    w02_be_ptr = w02_ptr + pid_b * stride_w02_b + pid_e * stride_w02_e  # [2H, D]

    # Task 1 bases
    dy1_b_ptr = dy1_ptr + pid_b * stride_dy1_b  # [H, N]
    v_b_ptr = v_ptr + pid_b * stride_v_b  # [N, D]
    w1_be_ptr = w1_ptr + pid_b * stride_w1_b + pid_e * stride_w1_e  # [D, H]

    # ------------------------
    # Task 0: DW0_W2 (2H x D)
    # ------------------------
    if task == 0:
        m_mask = offs_m < H2
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k0 = n_start
        while k0 < k_end:
            k_ids = k0 + offs_k
            k_mask = (k_ids < k_end) & (k_ids < N)

            # dy02 tile: (BM, BK) from [2H, N]
            dy_ptrs = (
                dy02_b_ptr
                + (offs_m[:, None] * stride_dy02_h2)
                + (k_ids[None, :] * stride_dy02_n)
            )
            dy_mask = (m_mask[:, None]) & (k_mask[None, :])
            dy = tl.load(dy_ptrs, mask=dy_mask, other=0).to(tl.bfloat16)

            # K tile: (BK, BN) from [N, D]
            x_ptrs = (
                k_b_ptr + (k_ids[:, None] * stride_k_n) + (offs_d[None, :] * stride_k_d)
            )
            x_mask = (k_mask[:, None]) & (d_mask[None, :])
            x = tl.load(x_ptrs, mask=x_mask, other=0).to(tl.bfloat16)

            acc += tl.dot(dy, x, out_dtype=tl.float32)
            k0 += BLOCK_K

        # Store: [2H, D] tile (BM, BN)
        out_ptrs = (
            w02_be_ptr
            + (offs_m[:, None] * stride_w02_h2)
            + (offs_d[None, :] * stride_w02_d)
        )
        o_mask = (m_mask[:, None]) & (d_mask[None, :])
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=o_mask)

    # ------------------------
    # Task 1: DW1 (D x H)
    #   We compute (H x N) @ (N x D) -> (H x D) and
    #   store into [D, H] by swapping strides on store.
    # ------------------------
    else:
        m_mask = offs_m < H
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k0 = n_start
        while k0 < k_end:
            k_ids = k0 + offs_k
            k_mask = (k_ids < k_end) & (k_ids < N)

            # hidden (dy1) tile: (BM, BK) from [H, N]
            dy_ptrs = (
                dy1_b_ptr
                + (offs_m[:, None] * stride_dy1_h)
                + (k_ids[None, :] * stride_dy1_n)
            )
            dy_mask = (m_mask[:, None]) & (k_mask[None, :])
            dy = tl.load(dy_ptrs, mask=dy_mask, other=0).to(tl.bfloat16)

            # V tile: (BK, BN) from [N, D]
            x_ptrs = (
                v_b_ptr + (k_ids[:, None] * stride_v_n) + (offs_d[None, :] * stride_v_d)
            )
            x_mask = (k_mask[:, None]) & (d_mask[None, :])
            x = tl.load(x_ptrs, mask=x_mask, other=0).to(tl.bfloat16)

            acc += tl.dot(dy, x, out_dtype=tl.float32)
            k0 += BLOCK_K

        # Store into [D, H]: index = base + D*stride_w1_d + H*stride_w1_h
        # Our acc is (BM,H_tile x BN,D_tile) -> store as (H,D), so we use (offs_m, offs_d) with swapped strides.
        out_ptrs = (
            w1_be_ptr
            + (offs_m[:, None] * stride_w1_h)
            + (offs_d[None, :] * stride_w1_d)
        )
        o_mask = (m_mask[:, None]) & (d_mask[None, :])
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=o_mask)


# -------------------
# Python-side wrapper
# -------------------
def grouped_two_gemm_fw_bwd_triton(
    DY0_DY2: torch.Tensor,  # [B, 2H, N] (bf16, contiguous)
    K: torch.Tensor,  # [B, N, D]   (bf16, contiguous)
    Hidden: torch.Tensor,  # [B, H, N]   (bf16, contiguous)
    V: torch.Tensor,  # [B, N, D]   (bf16, contiguous)
    group_sizes: torch.Tensor,  # [B, E]   (int32)
):
    """
    Outputs:
        DW0_W2: [B, E, 2H, D] (bf16)
        DW1: [B, E, D, H] (bf16)
    """
    assert DY0_DY2.dtype == torch.bfloat16
    assert K.dtype == torch.bfloat16
    assert Hidden.dtype == torch.bfloat16
    assert V.dtype == torch.bfloat16
    assert group_sizes.dtype in (torch.int32, torch.int64)
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)

    B, H2, N = DY0_DY2.shape
    H = H2 // 2
    Bk, Nk, D = K.shape

    Bg, E = group_sizes.shape

    device = DY0_DY2.device

    DW0_W2 = torch.empty((B, E, H2, D), device=device, dtype=torch.bfloat16)
    DW1 = torch.empty((B, E, D, H), device=device, dtype=torch.bfloat16)

    # Strides (in elements)
    stride_dy02_b, stride_dy02_h2, stride_dy02_n = H2 * N, N, 1
    stride_k_b, stride_k_n, stride_k_d = N * D, D, 1
    stride_dy1_b, stride_dy1_h, stride_dy1_n = H * N, N, 1
    stride_v_b, stride_v_n, stride_v_d = N * D, D, 1
    stride_w02_b, stride_w02_e, stride_w02_h2, stride_w02_d = E * H2 * D, H2 * D, D, 1
    stride_w1_b, stride_w1_e, stride_w1_d, stride_w1_h = E * D * H, D * H, H, 1

    def grid(meta):
        return (
            triton.cdiv(H2, meta["BLOCK_M"]),  # rows cover the larger (2H) extent
            triton.cdiv(D, meta["BLOCK_N"]),
            B * E * 2,  # two tasks per (b,e)
        )

    _two_grouped_gemm_bwd_kernel[grid](
        DY0_DY2,
        K,
        Hidden,
        V,
        group_sizes,
        DW0_W2,
        DW1,
        B,
        E,
        H,
        H2,
        N,
        D,
        stride_dy02_b,
        stride_dy02_h2,
        stride_dy02_n,
        stride_k_b,
        stride_k_n,
        stride_k_d,
        stride_dy1_b,
        stride_dy1_h,
        stride_dy1_n,
        stride_v_b,
        stride_v_n,
        stride_v_d,
        stride_w02_b,
        stride_w02_e,
        stride_w02_h2,
        stride_w02_d,
        stride_w1_b,
        stride_w1_e,
        stride_w1_d,
        stride_w1_h,
    )
    return DW0_W2, DW1


# -------------------------
# Reference (PyTorch/CPU/GPU)
# -------------------------
@torch.no_grad()
def reference_two_grouped_bwd(
    DY0_DY2: torch.Tensor,  # [B, 2H, N]
    K: torch.Tensor,  # [B, N, D]
    Hidden: torch.Tensor,  # [B, H, N]
    V: torch.Tensor,  # [B, N, D]
    group_sizes: torch.Tensor,  # [B, E]
):
    B, H2, N = DY0_DY2.shape
    H = H2 // 2
    _, _, D = K.shape
    _, E = group_sizes.shape

    device = DY0_DY2.device
    DW0_W2 = torch.empty((B, E, H2, D), device=device, dtype=torch.bfloat16)
    DW1 = torch.empty((B, E, D, H), device=device, dtype=torch.bfloat16)

    for b in range(B):
        n_start = 0
        for e in range(E):
            g = int(group_sizes[b, e].item())
            n_end = n_start + g

            # (2H x g) @ (g x D) -> (2H x D)
            w02 = DY0_DY2[b, :, n_start:n_end].float() @ K[b, n_start:n_end, :].float()

            # (H x g) @ (g x D) -> (H x D), store as (D x H)
            w1 = Hidden[b, :, n_start:n_end].float() @ V[b, n_start:n_end, :].float()

            DW0_W2[b, e] = w02.to(torch.bfloat16)
            DW1[b, e] = w1.transpose(0, 1).to(torch.bfloat16)

            n_start = n_end

    return DW0_W2, DW1


# -------------------------
# Helpers + correctness test
# -------------------------
def make_inputs(B, E, D, H, N):
    device = "cuda"
    dtype = torch.bfloat16
    # Gradients w.r.t. outputs
    DY0_DY2 = torch.randn(B, 2 * H, N, device=device, dtype=dtype)
    Hidden = torch.randn(B, H, N, device=device, dtype=dtype)
    # Inputs (K, V)
    K = torch.randn(B, N, D, device=device, dtype=dtype)
    V = torch.randn(B, N, D, device=device, dtype=dtype)

    mean_size = N // E
    group_sizes = torch.randint(
        1, max(2, mean_size), (B, E), device=device, dtype=torch.int32
    )
    # ensure sum of group sizes == N per batch
    sum_group_sizes = group_sizes.sum(dim=1, keepdim=True)  # [B,1]
    extra = N - sum_group_sizes.squeeze(1)
    group_sizes[:, -1] += extra  # last expert takes the slack
    return DY0_DY2, Hidden, K, V, group_sizes


def test_correctness():
    from benchmark import report_error

    torch.manual_seed(0)
    B, E, D, H, N = 4, 4, 1024, 1024, 4096

    DY0_DY2, Hidden, K, V, group_sizes = make_inputs(B, E, D, H, N)

    ref_dw02, ref_dw1 = reference_two_grouped_bwd(DY0_DY2, K, Hidden, V, group_sizes)
    tri_dw02, tri_dw1 = grouped_two_gemm_fw_bwd_triton(
        DY0_DY2, K, Hidden, V, group_sizes
    )

    report_error(ref_dw02, tri_dw02, "DW0_W2")
    report_error(ref_dw1, tri_dw1, "DW1")
    print("=> Done testing correctness")


if __name__ == "__main__":
    test_correctness()
