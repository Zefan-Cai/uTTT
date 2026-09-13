import torch
import triton
import triton.language as tl


# Smaller autotune space for faster testing.
def get_autotune_configs_small(
    block_M_list=(64, 128),
    block_N_list=(64, 128, 256),
    block_K_list=(32, 64),
    num_stages_list=(2, 3),
    threads_list=(256,),
):
    import itertools

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
    key=["H", "E", "D"],
)
@triton.jit
def _grouping_grouped_gemm_kernel(
    w0w2_ptr,  # [B, E, 2H, D]   (bf16)
    gw0w2_ptr,  # [B, E, 2H, D]   (bf16)
    w1_ptr,  # [B, E, 2D, H]   (bf16)  -- we will use the V-half: [D, H]
    gw1_ptr,  # [B, E, 2D, H]   (bf16)
    k_ptr,  # [B, N, D]       (bf16)
    v_ptr,  # [B, N, D]       (bf16)
    group_sizes_ptr,  # [B, E]       (int32/int64)
    y02_ptr,  # [B, 2H, N]      (bf16)  -> Y0_Y2
    gy02_ptr,  # [B, 2H, N]      (bf16)  -> G_Y0_Y2
    dh_ptr,  # [B, H,  N]      (bf16)  -> DHidden
    gdh_ptr,  # [B, H,  N]      (bf16)  -> G_DHidden
    # shape params
    B: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,  # Hidden
    N,  # total tokens (runtime)
    D: tl.constexpr,  # head_dim / K,V dim
    # strides (elements)
    # W0_W2 / G_W0_W2
    stride_w02_b,
    stride_w02_e,
    stride_w02_m,
    stride_w02_k,
    stride_gw02_b,
    stride_gw02_e,
    stride_gw02_m,
    stride_gw02_k,
    # W1 / G_W1  (layout [B,E,2D,H])
    stride_w1_b,
    stride_w1_e,
    stride_w1_2d,
    stride_w1_h,
    stride_gw1_b,
    stride_gw1_e,
    stride_gw1_2d,
    stride_gw1_h,
    # K / V
    stride_k_b,
    stride_k_n,
    stride_k_d,
    stride_v_b,
    stride_v_n,
    stride_v_d,
    # outputs
    stride_y02_b,
    stride_y02_m,
    stride_y02_n,
    stride_gy02_b,
    stride_gy02_m,
    stride_gy02_n,
    stride_dh_b,
    stride_dh_m,
    stride_dh_n,
    stride_gdh_b,
    stride_gdh_m,
    stride_gdh_n,
    # tiling
    BLOCK_M: tl.constexpr,  # along H (we reuse for 2H by offsetting)
    BLOCK_N: tl.constexpr,  # along N (grouped)
    BLOCK_K: tl.constexpr,  # along D (reduction)
):

    # ---------------------------------------
    # Decode program IDs into (b,e) and N-tile
    # ---------------------------------------
    pid_bn = tl.program_id(axis=0)  # over N-tiles of all groups
    pid_m = tl.program_id(axis=1)  # over H tiles

    # Map pid_bn -> (pid_b, pid_e, pid_n) + starting N index for this group's tile
    pid_n = pid_bn
    pid_b = 0
    pid_e = 0
    n_tile_start_index = 0
    n_tile_group_end_idx = N

    # Scan groups in batch-major order
    hit = False
    ge_linear = 0
    while (not hit) and (ge_linear < B * E):
        b = ge_linear // E
        e = ge_linear % E
        gsize = tl.load(group_sizes_ptr + b * E + e)
        tiles_in_group = tl.cdiv(gsize, BLOCK_N)

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
        ge_linear += 1

    # Compute intra-group N start for this tile
    n_tile_start_index += pid_n * BLOCK_N
    if n_tile_start_index >= N:
        return

    # ---------------------------------------
    # Offsets and masks
    # ---------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]  (along H)
    offs_n = n_tile_start_index + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)  # [BK]

    m_mask = offs_m < H
    n_mask_1d = (offs_n < n_tile_group_end_idx) & (offs_n < N)

    # ---------------------------------------
    # Base pointers for this (b,e)
    # ---------------------------------------
    # W0/W2
    w02_be = w0w2_ptr + pid_b * stride_w02_b + pid_e * stride_w02_e
    gw02_be = gw0w2_ptr + pid_b * stride_gw02_b + pid_e * stride_gw02_e
    # W1 halves (we use the V-half: offset 0 or D along the 2D axis)
    # half_offset = 0 if W1_V_HALF == 0 else D
    w1_be = w1_ptr + pid_b * stride_w1_b + pid_e * stride_w1_e
    gw1_be = gw1_ptr + pid_b * stride_gw1_b + pid_e * stride_gw1_e
    # K/V for this batch
    k_b = k_ptr + pid_b * stride_k_b
    v_b = v_ptr + pid_b * stride_v_b
    # Outputs for this batch
    y02_b = y02_ptr + pid_b * stride_y02_b
    gy02_b = gy02_ptr + pid_b * stride_gy02_b
    dh_b = dh_ptr + pid_b * stride_dh_b
    gdh_b = gdh_ptr + pid_b * stride_gdh_b

    # =====================================================================
    # Phase A: four accumulators for K-path (Y0, Y2) and (GY0, GY2)
    # =====================================================================
    acc_y0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_y2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # acc_gy0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # acc_gy2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, D, BLOCK_K):
        k_ids = k0 + offs_k  # [BK]
        k_mask = k_ids < D

        # K tile: (BN, BK)
        k_ptrs = k_b + (offs_n[:, None] * stride_k_n) + (k_ids[None, :] * stride_k_d)
        kmask2 = n_mask_1d[:, None] & k_mask[None, :]
        K_tile = tl.load(k_ptrs, mask=kmask2, other=0).to(tl.bfloat16)

        # W0 (BM, BK) and W2 (BM, BK)
        w0_ptrs = (
            w02_be + (offs_m[:, None] * stride_w02_m) + (k_ids[None, :] * stride_w02_k)
        )
        w2_ptrs = (
            w02_be
            + ((H + offs_m)[:, None] * stride_w02_m)
            + (k_ids[None, :] * stride_w02_k)
        )
        wmask = m_mask[:, None] & k_mask[None, :]
        W0 = tl.load(w0_ptrs, mask=wmask, other=0).to(tl.bfloat16)
        W2 = tl.load(w2_ptrs, mask=wmask, other=0).to(tl.bfloat16)

        # Accumulate: (BM,BK) x (BK,BN)
        acc_y0 += tl.dot(W0, tl.trans(K_tile), out_dtype=tl.float32)
        acc_y2 += tl.dot(W2, tl.trans(K_tile), out_dtype=tl.float32)

    # Store Y0 / Y2
    y0_ptrs = (
        y02_b + (offs_m[:, None] * stride_y02_m) + (offs_n[None, :] * stride_y02_n)
    )
    y2_ptrs = (
        y02_b
        + ((H + offs_m)[:, None] * stride_y02_m)
        + (offs_n[None, :] * stride_y02_n)
    )
    ymask = m_mask[:, None] & n_mask_1d[None, :]
    tl.store(y0_ptrs, acc_y0.to(tl.bfloat16), mask=ymask)
    tl.store(y2_ptrs, acc_y2.to(tl.bfloat16), mask=ymask)

    # compute gy0 and gy2
    # set accumulators to 0
    acc_y0 *= 0.0
    acc_y2 *= 0.0
    for k0 in range(0, D, BLOCK_K):
        k_ids = k0 + offs_k  # [BK]
        k_mask = k_ids < D

        # K tile: (BN, BK)
        k_ptrs = k_b + (offs_n[:, None] * stride_k_n) + (k_ids[None, :] * stride_k_d)
        kmask2 = n_mask_1d[:, None] & k_mask[None, :]
        K_tile = tl.load(k_ptrs, mask=kmask2, other=0).to(tl.bfloat16)

        # G_W0 / G_W2
        g0_ptrs = (
            gw02_be
            + (offs_m[:, None] * stride_gw02_m)
            + (k_ids[None, :] * stride_gw02_k)
        )
        g2_ptrs = (
            gw02_be
            + ((H + offs_m)[:, None] * stride_gw02_m)
            + (k_ids[None, :] * stride_gw02_k)
        )
        # GW0, GW2
        wmask = m_mask[:, None] & k_mask[None, :]
        G0 = tl.load(g0_ptrs, mask=wmask, other=0).to(tl.bfloat16)
        G2 = tl.load(g2_ptrs, mask=wmask, other=0).to(tl.bfloat16)

        # Accumulate: (BM,BK) x (BK,BN)
        acc_y0 += tl.dot(G0, tl.trans(K_tile), out_dtype=tl.float32)
        acc_y2 += tl.dot(G2, tl.trans(K_tile), out_dtype=tl.float32)

    # Store G_Y0 / G_Y2
    gy0_ptrs = (
        gy02_b + (offs_m[:, None] * stride_gy02_m) + (offs_n[None, :] * stride_gy02_n)
    )
    gy2_ptrs = (
        gy02_b
        + ((H + offs_m)[:, None] * stride_gy02_m)
        + (offs_n[None, :] * stride_gy02_n)
    )
    tl.store(gy0_ptrs, acc_y0.to(tl.bfloat16), mask=ymask)
    tl.store(gy2_ptrs, acc_y2.to(tl.bfloat16), mask=ymask)

    # =====================================================================
    # Phase B: two accumulators for V-path (DH, GDH) using the V-half of W1
    # =====================================================================
    # acc_dh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # acc_gdh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    acc_y0 *= 0.0  # for dh
    acc_y2 *= 0.0  # for gdh
    for k0 in range(0, D, BLOCK_K):
        k_ids = k0 + offs_k
        k_mask = k_ids < D

        # V tile: (BN, BK)
        v_ptrs = v_b + (offs_n[:, None] * stride_v_n) + (k_ids[None, :] * stride_v_d)
        vmask2 = n_mask_1d[:, None] & k_mask[None, :]
        V_tile = tl.load(v_ptrs, mask=vmask2, other=0).to(tl.bfloat16)

        # W1_V half as (BM, BK): treat original layout [2D, H]; we index rows by H, cols by D-half
        w1_ptrs = (
            w1_be + (offs_m[:, None] * stride_w1_h) + (k_ids[None, :] * stride_w1_2d)
        )
        gw1_ptrs = (
            gw1_be + (offs_m[:, None] * stride_gw1_h) + (k_ids[None, :] * stride_gw1_2d)
        )
        wmask = m_mask[:, None] & k_mask[None, :]
        W1V = tl.load(w1_ptrs, mask=wmask, other=0).to(tl.bfloat16)
        GW1V = tl.load(gw1_ptrs, mask=wmask, other=0).to(tl.bfloat16)

        acc_y0 += tl.dot(W1V, tl.trans(V_tile), out_dtype=tl.float32)  # (H x N)
        acc_y2 += tl.dot(GW1V, tl.trans(V_tile), out_dtype=tl.float32)

    # Store DH / GDH
    dh_ptrs = dh_b + (offs_m[:, None] * stride_dh_m) + (offs_n[None, :] * stride_dh_n)
    gdh_ptrs = (
        gdh_b + (offs_m[:, None] * stride_gdh_m) + (offs_n[None, :] * stride_gdh_n)
    )
    dmask = m_mask[:, None] & n_mask_1d[None, :]
    tl.store(dh_ptrs, acc_y0.to(tl.bfloat16), mask=dmask)
    tl.store(gdh_ptrs, acc_y2.to(tl.bfloat16), mask=dmask)


def grouping_four_grouped_gemm_triton(
    W0_W2: torch.Tensor,  # [B, E, 2H, D] bf16, contiguous
    G_W0_W2: torch.Tensor,  # [B, E, 2H, D] bf16, contiguous
    W1: torch.Tensor,  # [B, E, 2D, H] bf16, contiguous
    G_W1: torch.Tensor,  # [B, E, 2D, H] bf16, contiguous
    K: torch.Tensor,  # [B, N, D]     bf16, contiguous (tokens permuted by expert)
    V: torch.Tensor,  # [B, N, D]     bf16, contiguous (same permutation as K)
    group_sizes: torch.Tensor,  # [B, E]    int32/64
):
    """
    Computes, with shared grouping along N:
      Y0_Y2   = W0_W2   @ K^T         -> [B, 2H, N]
      G_Y0_Y2 = G_W0_W2 @ K^T         -> [B, 2H, N]
      DHidden = (W1_V)^T @ V^T        -> [B,  H, N]  where W1_V is the chosen D-half of W1
      G_DHidden = (G_W1_V)^T @ V^T    -> [B,  H, N]

    Assumes tokens for each expert are contiguous per batch and that
    group_sizes[b].sum() == N for all b.
    """
    # basic checks
    for t in (W0_W2, G_W0_W2, W1, G_W1, K, V):
        assert t.is_contiguous(), "All inputs must be contiguous"
        assert t.dtype == torch.bfloat16, "bf16 expected for all inputs"

    B, E, M2, D = W0_W2.shape
    assert M2 % 2 == 0
    H = M2 // 2

    Bx, Ex, D2, Hx = W1.shape
    assert (Bx, Ex, Hx) == (B, E, H), "W1 must be [B,E,2D,H] matching B,E,H"
    # assert D2 == 2 * D, "W1's first dim must be 2*D"

    Bk, N, Dk = K.shape
    assert (Bk, Dk) == (B, D), "K must be [B,N,D] with D matching"
    Bv, Nv, Dv = V.shape
    assert (Bv, Nv, Dv) == (B, N, D), "V must be [B,N,D] matching"

    assert group_sizes.shape == (B, E), "group_sizes must be [B,E]"
    assert group_sizes.device == K.device

    # outputs
    Y0_Y2 = torch.empty((B, 2 * H, N), device=K.device, dtype=torch.bfloat16)
    G_Y0_Y2 = torch.empty((B, 2 * H, N), device=K.device, dtype=torch.bfloat16)
    DHidden = torch.empty((B, H, N), device=K.device, dtype=torch.bfloat16)
    G_DHidden = torch.empty((B, H, N), device=K.device, dtype=torch.bfloat16)

    # strides (elements)
    # sw02_b, sw02_e, sw02_m, sw02_k = W0_W2.stride()
    # sgw02_b, sgw02_e, sgw02_m, sgw02_k = G_W0_W2.stride()

    # sw1_b, sw1_e, sw1_2d, sw1_h = W1.stride()
    # sgw1_b, sgw1_e, sgw1_2d, sgw1_h = G_W1.stride()

    # sk_b, sk_n, sk_d = K.stride()
    # sv_b, sv_n, sv_d = V.stride()

    # sy02_b, sy02_m, sy02_n = Y0_Y2.stride()
    # sgy02_b, sgy02_m, sgy02_n = G_Y0_Y2.stride()
    # sdh_b, sdh_m, sdh_n = DHidden.stride()
    # sgdh_b, sgdh_m, sgdh_n = G_DHidden.stride()

    # compute strides assuming contiguous tensors
    sw02_b, sw02_e, sw02_m, sw02_k = E * M2 * D, M2 * D, D, 1
    sgw02_b, sgw02_e, sgw02_m, sgw02_k = E * M2 * D, M2 * D, D, 1

    sw1_b, sw1_e, sw1_2d, sw1_h = E * D2 * H, D2 * H, H, 1
    sgw1_b, sgw1_e, sgw1_2d, sgw1_h = E * D2 * H, D2 * H, H, 1

    sk_b, sk_n, sk_d = N * D, D, 1
    sv_b, sv_n, sv_d = N * D, D, 1

    sy02_b, sy02_m, sy02_n = 2 * H * N, N, 1
    sgy02_b, sgy02_m, sgy02_n = 2 * H * N, N, 1
    sdh_b, sdh_m, sdh_n = H * N, N, 1
    sgdh_b, sgdh_m, sgdh_n = H * N, N, 1

    # total N-tiles across all groups
    def grid(meta):
        BN = meta["BLOCK_N"]
        total_tiles = int(((group_sizes + BN - 1) // BN).sum().item())
        return (total_tiles, triton.cdiv(H, meta["BLOCK_M"]))

    _grouping_grouped_gemm_kernel[grid](
        W0_W2,
        G_W0_W2,
        W1,
        G_W1,
        K,
        V,
        group_sizes,
        Y0_Y2,
        G_Y0_Y2,
        DHidden,
        G_DHidden,
        B,
        E,
        H,
        N,
        D,
        # strides
        sw02_b,
        sw02_e,
        sw02_m,
        sw02_k,
        sgw02_b,
        sgw02_e,
        sgw02_m,
        sgw02_k,
        sw1_b,
        sw1_e,
        sw1_2d,
        sw1_h,
        sgw1_b,
        sgw1_e,
        sgw1_2d,
        sgw1_h,
        sk_b,
        sk_n,
        sk_d,
        sv_b,
        sv_n,
        sv_d,
        sy02_b,
        sy02_m,
        sy02_n,
        sgy02_b,
        sgy02_m,
        sgy02_n,
        sdh_b,
        sdh_m,
        sdh_n,
        sgdh_b,
        sgdh_m,
        sgdh_n,
        # kernel constexpr
    )

    return Y0_Y2, G_Y0_Y2, DHidden, G_DHidden


@torch.no_grad()
def reference_four_grouped(W0_W2, G_W0_W2, W1, G_W1, K, V, group_sizes):
    B, E, M2, D = W0_W2.shape
    H = M2 // 2
    _, _, D2, Hx = W1.shape
    assert Hx == H and D2 == D
    Bk, N, Dk = K.shape
    assert (Bk, Dk) == (B, D)
    assert V.shape == K.shape

    Y0_Y2 = torch.zeros((B, 2 * H, N), device=K.device, dtype=torch.float32)
    G_Y0_Y2 = torch.zeros((B, 2 * H, N), device=K.device, dtype=torch.float32)
    DHidden = torch.zeros((B, H, N), device=K.device, dtype=torch.float32)
    G_DHidden = torch.zeros((B, H, N), device=K.device, dtype=torch.float32)

    for b in range(B):
        n0 = 0
        for e in range(E):
            g = int(group_sizes[b, e].item())
            if g == 0:
                continue

            Kseg = K[b, n0 : n0 + g, :].to(torch.float32)  # (g, D)
            Vseg = V[b, n0 : n0 + g, :].to(torch.float32)

            # W0/W2 paths
            W0 = W0_W2[b, e, 0:H, :].to(torch.float32)  # (H, D)
            W2 = W0_W2[b, e, H : 2 * H, :].to(torch.float32)  # (H, D)
            GW0 = G_W0_W2[b, e, 0:H, :].to(torch.float32)
            GW2 = G_W0_W2[b, e, H : 2 * H, :].to(torch.float32)

            Y0 = W0 @ Kseg.T  # (H, g)
            Y2 = W2 @ Kseg.T
            GY0 = GW0 @ Kseg.T
            GY2 = GW2 @ Kseg.T

            Y0_Y2[b, 0:H, n0 : n0 + g] = Y0
            Y0_Y2[b, H : 2 * H, n0 : n0 + g] = Y2
            G_Y0_Y2[b, 0:H, n0 : n0 + g] = GY0
            G_Y0_Y2[b, H : 2 * H, n0 : n0 + g] = GY2

            # W1 V-half path

            W1V = W1[b, e, :, :].to(torch.float32)  # (D, H)
            GW1V = G_W1[b, e, :, :].to(torch.float32)

            DH = W1V.T @ Vseg.T  # (H, g)
            GDH = GW1V.T @ Vseg.T

            DHidden[b, :, n0 : n0 + g] = DH
            G_DHidden[b, :, n0 : n0 + g] = GDH

            n0 += g

    return (
        Y0_Y2.to(torch.bfloat16),
        G_Y0_Y2.to(torch.bfloat16),
        DHidden.to(torch.bfloat16),
        G_DHidden.to(torch.bfloat16),
    )


def make_inputs(B, E, D, H, N):
    w0_w2 = torch.randn(B, E, 2 * H, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(B, E, D, H, device="cuda", dtype=torch.bfloat16)
    G_W1 = torch.randn(B, E, D, H, device="cuda", dtype=torch.bfloat16)
    G_w0_w2 = torch.randn(B, E, 2 * H, D, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
    mean_size = N // E
    group_sizes = torch.randint(1, mean_size, (B, E), device="cuda", dtype=torch.int32)
    sum_group_sizes = group_sizes.sum(dim=1)
    extra_size = N - sum_group_sizes
    group_sizes[:, -1] += extra_size

    return w0_w2, G_w0_w2, w1, G_W1, x, v, group_sizes


def test_correctness():

    from benchmark import report_error

    B, E, D, H, N = 4, 4, 1024, 1024, 4096
    inputs = make_inputs(B, E, D, H, N)

    ref_y0_y2, ref_g_y0_y2, ref_dhidden, ref_g_dhidden = reference_four_grouped(*inputs)
    triton_y0_y2, triton_g_y0_y2, triton_dhidden, triton_g_dhidden = (
        grouping_four_grouped_gemm_triton(*inputs)
    )
    report_error(ref_y0_y2, triton_y0_y2, "y0_y2")
    report_error(ref_g_y0_y2, triton_g_y0_y2, "g_y0_y2")
    report_error(ref_dhidden, triton_dhidden, "dhidden")
    report_error(ref_g_dhidden, triton_g_dhidden, "g_dhidden")
    print("=> Done testing correctness")


if __name__ == "__main__":
    test_correctness()
