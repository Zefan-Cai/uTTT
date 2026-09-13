import torch
import triton
import triton.language as tl


def get_autotune_configs_small(
    block_M_list=(64, 128),  # here BLOCK_M == tile along D
    block_N_list=(64, 128, 256),  # tile along N
    block_K_list=(32, 64),  # tile along reduction (H or 2H)
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


@triton.autotune(configs=get_autotune_configs_small(), key=["H", "E", "D"])
@triton.jit
def _grouped_grad_kv_fused_kernel(
    # --- K-path weights/activations ---
    gw02_ptr,  # [B, E, 2H, D]  (bf16)
    w02_ptr,  # [B, E, 2H, D]  (bf16)
    dylr_ptr,  # [B, 2H, N]     (bf16)  DY0_with_lr0_and_DY2_with_lr2
    dy02_ptr,  # [B, 2H, N]     (bf16)  grad_Y0_Y2
    # --- V-path weights/activations ---
    w1_ptr,  # [B, E, DW, H]  (bf16)  DW == D or 2D (see flag)
    gw1_ptr,  # [B, E, DW, H]  (bf16)
    dh_ptr,  # [B, H,  N]     (bf16)  grad_DHidden
    hlr_ptr,  # [B, H,  N]     (bf16)  Hidden_with_lr1
    # grouping
    group_sizes_ptr,  # [B, E]         (int32/int64)
    # outputs
    gk_ptr,  # [B, N, D]      (bf16)  grad_K
    gv_ptr,  # [B, N, D]      (bf16)  grad_V
    # shapes
    B: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,  # Hidden
    N,  # runtime tokens
    D: tl.constexpr,  # head dim
    # strides (elements)
    # gw02 / w02  [B,E,2H,D]
    s_gw02_b,
    s_gw02_e,
    s_gw02_m,
    s_gw02_k,
    s_w02_b,
    s_w02_e,
    s_w02_m,
    s_w02_k,
    # dylr / dy02 [B,2H,N]
    s_dylr_b,
    s_dylr_m,
    s_dylr_n,
    s_dy02_b,
    s_dy02_m,
    s_dy02_n,
    # w1 / gw1   [B,E,DW,H]
    s_w1_b,
    s_w1_e,
    s_w1_d,
    s_w1_h,
    s_gw1_b,
    s_gw1_e,
    s_gw1_d,
    s_gw1_h,
    # dh / hlr   [B,H,N]
    s_dh_b,
    s_dh_m,
    s_dh_n,
    s_hlr_b,
    s_hlr_m,
    s_hlr_n,
    # outputs    [B,N,D]
    s_gk_b,
    s_gk_n,
    s_gk_d,
    s_gv_b,
    s_gv_n,
    s_gv_d,
    # tiling
    BLOCK_M: tl.constexpr,  # along D
    BLOCK_N: tl.constexpr,  # along N
    BLOCK_K: tl.constexpr,  # along reduction (H or 2H)
):
    # ----------------------------
    # Program ids and group decode
    # ----------------------------
    pid_bn = tl.program_id(axis=0)  # tiles over N across all (b,e)
    pid_d = tl.program_id(axis=1)  # tiles over D

    # Map pid_bn -> (pid_b, pid_e, pid_n_in_group) and tile offset in N
    pid_n = pid_bn
    pid_b = 0
    pid_e = 0
    n_tile_start = 0
    n_tile_end = N

    hit = False
    ge = 0
    while (not hit) and (ge < B * E):
        b = ge // E
        e = ge % E
        g = tl.load(group_sizes_ptr + b * E + e)
        t = tl.cdiv(g, BLOCK_N)
        if e == 0:
            n_tile_start = 0
        if pid_n < t:
            pid_b = b
            pid_e = e
            hit = True
            n_tile_end = n_tile_start + g
        else:
            pid_n -= t
            n_tile_start += g
        ge += 1

    n_tile_start += pid_n * BLOCK_N
    if n_tile_start >= N:
        return

    # Offsets
    offs_d = pid_d * BLOCK_M + tl.arange(0, BLOCK_M)  # [BD] along D
    offs_n = n_tile_start + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)  # [BK]

    d_mask = offs_d < D
    n_mask = (offs_n < n_tile_end) & (offs_n < N)

    # Base pointers for current (b,e)
    gw02_be = gw02_ptr + pid_b * s_gw02_b + pid_e * s_gw02_e
    w02_be = w02_ptr + pid_b * s_w02_b + pid_e * s_w02_e

    dylr_b = dylr_ptr + pid_b * s_dylr_b
    dy02_b = dy02_ptr + pid_b * s_dy02_b

    w1_be = w1_ptr + pid_b * s_w1_b + pid_e * s_w1_e
    gw1_be = gw1_ptr + pid_b * s_gw1_b + pid_e * s_gw1_e

    dh_b = dh_ptr + pid_b * s_dh_b
    hlr_b = hlr_ptr + pid_b * s_hlr_b

    gk_b = gk_ptr + pid_b * s_gk_b
    gv_b = gv_ptr + pid_b * s_gv_b

    # ======================================================
    # Phase 1: grad_K  (reduce over 2H)
    # acc_k shape: (BD, BN) == (Dtile, Ntile)
    # ======================================================
    acc_k = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    R2 = 2 * H
    for r0 in range(0, R2, BLOCK_K):
        r_ids = r0 + offs_k
        r_mask = r_ids < R2

        # Load W^T tiles: [D, 2H] -> (BD x BK)
        gwt_ptrs = gw02_be + (r_ids[None, :] * s_gw02_m) + (offs_d[:, None] * s_gw02_k)
        wt_ptrs = w02_be + (r_ids[None, :] * s_w02_m) + (offs_d[:, None] * s_w02_k)
        wmask = d_mask[:, None] & r_mask[None, :]
        GWt = tl.load(gwt_ptrs, mask=wmask, other=0).to(tl.bfloat16)  # (BD, BK)
        Wt = tl.load(wt_ptrs, mask=wmask, other=0).to(tl.bfloat16)

        # Load DY tiles: [2H, N] -> (BK x BN)
        dylr_ptrs = dylr_b + (r_ids[:, None] * s_dylr_m) + (offs_n[None, :] * s_dylr_n)
        dy02_ptrs = dy02_b + (r_ids[:, None] * s_dy02_m) + (offs_n[None, :] * s_dy02_n)
        dymask = r_mask[:, None] & n_mask[None, :]
        DYlr = tl.load(dylr_ptrs, mask=dymask, other=0).to(tl.bfloat16)  # (BK, BN)
        DY02 = tl.load(dy02_ptrs, mask=dymask, other=0).to(tl.bfloat16)

        # acc_k += GWt @ DYlr + Wt @ DY02
        acc_k += tl.dot(GWt, DYlr, out_dtype=tl.float32)
        acc_k += tl.dot(Wt, DY02, out_dtype=tl.float32)

    # store grad_K as [B, N, D] -> use transposed accumulator
    gk_ptrs = gk_b + (offs_n[:, None] * s_gk_n) + (offs_d[None, :] * s_gk_d)
    gk_mask = n_mask[:, None] & d_mask[None, :]
    tl.store(gk_ptrs, tl.trans(acc_k).to(tl.bfloat16), mask=gk_mask)

    # ======================================================
    # Phase 2: grad_V  (reduce over H)
    # acc_v shape: (BD, BN) == (Dtile, Ntile)
    # ======================================================
    acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for h0 in range(0, H, BLOCK_K):
        h_ids = h0 + offs_k
        h_mask = h_ids < H

        # Load W1 tiles: [D, H] -> (BD x BK)
        w1_ptrs = w1_be + (offs_d[:, None] * s_w1_d) + (h_ids[None, :] * s_w1_h)
        gw1_ptrs = gw1_be + (offs_d[:, None] * s_gw1_d) + (h_ids[None, :] * s_gw1_h)
        wmask = d_mask[:, None] & h_mask[None, :]
        W1 = tl.load(w1_ptrs, mask=wmask, other=0).to(tl.bfloat16)  # (BD, BK)
        GW1 = tl.load(gw1_ptrs, mask=wmask, other=0).to(tl.bfloat16)

        # Load activations: [H, N] -> (BK x BN)
        dh_ptrs = dh_b + (h_ids[:, None] * s_dh_m) + (offs_n[None, :] * s_dh_n)
        hlr_ptrs = hlr_b + (h_ids[:, None] * s_hlr_m) + (offs_n[None, :] * s_hlr_n)
        amask = h_mask[:, None] & n_mask[None, :]
        dH = tl.load(dh_ptrs, mask=amask, other=0).to(tl.bfloat16)  # (BK, BN)
        Hlr = tl.load(hlr_ptrs, mask=amask, other=0).to(tl.bfloat16)

        # acc_v += W1 @ dH + GW1 @ Hlr
        acc_v += tl.dot(W1, dH, out_dtype=tl.float32)
        acc_v += tl.dot(GW1, Hlr, out_dtype=tl.float32)

    # store grad_V as [B, N, D]
    gv_ptrs = gv_b + (offs_n[:, None] * s_gv_n) + (offs_d[None, :] * s_gv_d)
    gv_mask = n_mask[:, None] & d_mask[None, :]
    tl.store(gv_ptrs, tl.trans(acc_v).to(tl.bfloat16), mask=gv_mask)


@torch.no_grad()
def fused_grouped_grad_kv(
    G_W0_W2: torch.Tensor,  # [B,E,2H,D] bf16 contiguous
    W0_W2: torch.Tensor,  # [B,E,2H,D] bf16 contiguous
    DY_lr: torch.Tensor,  # [B,2H,N]   bf16 contiguous
    dY_02: torch.Tensor,  # [B,2H,N]   bf16 contiguous
    W1: torch.Tensor,  # [B,E,D,H] bf16 contiguous
    G_W1: torch.Tensor,  # [B,E,D,H] bf16 contiguous
    dH: torch.Tensor,  # [B,H,N]    bf16 contiguous
    H_lr: torch.Tensor,  # [B,H,N]    bf16 contiguous
    group_sizes: torch.Tensor,  # [B,E]      int32/int64 on same device
):
    # Basic checks (mirror your style)
    tensors = (G_W0_W2, W0_W2, DY_lr, dY_02, W1, G_W1, dH, H_lr)
    for t in tensors:
        assert t.is_contiguous(), "All inputs must be contiguous"
        assert t.dtype == torch.bfloat16, "bf16 expected for all inputs"

    B, E, M2, D = W0_W2.shape
    assert M2 % 2 == 0
    H = M2 // 2

    # DY and dY_02 sanity
    B, M2, N = DY_lr.shape

    # W1 / GW1
    B, E, D, H = W1.shape

    # outputs
    grad_K = torch.empty((B, N, D), device=W0_W2.device, dtype=torch.bfloat16)
    grad_V = torch.empty((B, N, D), device=W0_W2.device, dtype=torch.bfloat16)

    # compute contiguous strides (elements)
    s_gw02_b, s_gw02_e, s_gw02_m, s_gw02_k = E * (2 * H) * D, (2 * H) * D, D, 1
    s_w02_b, s_w02_e, s_w02_m, s_w02_k = s_gw02_b, s_gw02_e, s_gw02_m, s_gw02_k

    s_dylr_b, s_dylr_m, s_dylr_n = (2 * H) * N, N, 1
    s_dy02_b, s_dy02_m, s_dy02_n = (2 * H) * N, N, 1

    s_w1_b, s_w1_e, s_w1_d, s_w1_h = E * D * H, D * H, H, 1
    s_gw1_b, s_gw1_e, s_gw1_d, s_gw1_h = s_w1_b, s_w1_e, s_w1_d, s_w1_h

    s_dh_b, s_dh_m, s_dh_n = H * N, N, 1
    s_hlr_b, s_hlr_m, s_hlr_n = H * N, N, 1

    s_gk_b, s_gk_n, s_gk_d = N * D, D, 1
    s_gv_b, s_gv_n, s_gv_d = N * D, D, 1

    # grid: total N-tiles across all groups, and tiles across D
    def grid(meta):
        BN = meta["BLOCK_N"]
        total_n_tiles = int(((group_sizes + BN - 1) // BN).sum().item())
        return (total_n_tiles, triton.cdiv(D, meta["BLOCK_M"]))

    _grouped_grad_kv_fused_kernel[grid](
        # K path
        G_W0_W2,
        W0_W2,
        DY_lr,
        dY_02,
        # V path
        W1,
        G_W1,
        dH,
        H_lr,
        # grouping
        group_sizes,
        # outputs
        grad_K,
        grad_V,
        # shapes
        B,
        E,
        H,
        N,
        D,
        # strides
        s_gw02_b,
        s_gw02_e,
        s_gw02_m,
        s_gw02_k,
        s_w02_b,
        s_w02_e,
        s_w02_m,
        s_w02_k,
        s_dylr_b,
        s_dylr_m,
        s_dylr_n,
        s_dy02_b,
        s_dy02_m,
        s_dy02_n,
        s_w1_b,
        s_w1_e,
        s_w1_d,
        s_w1_h,
        s_gw1_b,
        s_gw1_e,
        s_gw1_d,
        s_gw1_h,
        s_dh_b,
        s_dh_m,
        s_dh_n,
        s_hlr_b,
        s_hlr_m,
        s_hlr_n,
        s_gk_b,
        s_gk_n,
        s_gk_d,
        s_gv_b,
        s_gv_n,
        s_gv_d,
    )
    return grad_K, grad_V


@torch.no_grad()
def reference_grad_kv(
    G_W0_W2,
    W0_W2,
    DY_lr,
    dY_02,
    W1,
    G_W1,
    dH,
    H_lr,
    group_sizes,
):
    B, E, M2, D = W0_W2.shape
    H = M2 // 2
    _, _, DW, Hx = W1.shape
    assert Hx == H
    Bk, N, Dk = (
        dH.transpose(1, 2).shape[0],
        dH.shape[2],
        None,
    )  # just to avoid unused var
    assert DY_lr.shape == (B, 2 * H, N) and dY_02.shape == (B, 2 * H, N)
    assert dH.shape == (B, H, N) and H_lr.shape == (B, H, N)

    grad_K = torch.zeros((B, N, D), device=W0_W2.device, dtype=torch.float32)
    grad_V = torch.zeros((B, N, D), device=W0_W2.device, dtype=torch.float32)

    for b in range(B):
        n0 = 0
        for e in range(E):
            g = int(group_sizes[b, e].item())
            if g == 0:
                continue

            # --- grad_K segment ---
            GWt = G_W0_W2[b, e].to(torch.float32).transpose(0, 1)  # [D,2H]
            Wt = W0_W2[b, e].to(torch.float32).transpose(0, 1)  # [D,2H]
            DYlr_seg = DY_lr[b, :, n0 : n0 + g].to(torch.float32)  # [2H,g]
            DY02_seg = dY_02[b, :, n0 : n0 + g].to(torch.float32)  # [2H,g]
            GK_seg = (GWt @ DYlr_seg + Wt @ DY02_seg).transpose(0, 1)  # [g,D]
            grad_K[b, n0 : n0 + g, :] = GK_seg

            # --- grad_V segment ---

            W1_slice = W1[b, e, :, :].to(torch.float32)  # [D,H]
            GW1_slice = G_W1[b, e, :, :].to(torch.float32)  # [D,H]
            dH_seg = dH[b, :, n0 : n0 + g].to(torch.float32)  # [H,g]
            Hlr_seg = H_lr[b, :, n0 : n0 + g].to(torch.float32)  # [H,g]
            GV_seg = (W1_slice @ dH_seg + GW1_slice @ Hlr_seg).transpose(0, 1)  # [g,D]
            grad_V[b, n0 : n0 + g, :] = GV_seg

            n0 += g

    return grad_K.to(torch.bfloat16), grad_V.to(torch.bfloat16)


def make_inputs(B, E, D, H, N):
    w0_w2 = torch.randn(B, E, 2 * H, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(B, E, D, H, device="cuda", dtype=torch.bfloat16)
    G_W1 = torch.randn(B, E, D, H, device="cuda", dtype=torch.bfloat16)
    G_w0_w2 = torch.randn(B, E, 2 * H, D, device="cuda", dtype=torch.bfloat16)
    dy0_y2 = torch.randn(B, 2 * H, N, device="cuda", dtype=torch.bfloat16)
    dh = torch.randn(B, H, N, device="cuda", dtype=torch.bfloat16)

    dy_lr = torch.randn(B, 2 * H, N, device="cuda", dtype=torch.bfloat16)
    H_lr = torch.randn(B, H, N, device="cuda", dtype=torch.bfloat16)
    mean_size = N // E
    group_sizes = torch.randint(1, mean_size, (B, E), device="cuda", dtype=torch.int32)
    sum_group_sizes = group_sizes.sum(dim=1)
    extra_size = N - sum_group_sizes
    group_sizes[:, -1] += extra_size

    return w0_w2, G_w0_w2, dy_lr, dy0_y2, w1, G_W1, dh, H_lr, group_sizes


def test_correctness():
    from benchmark import report_error

    B, E, D, H, N = 4, 4, 1024, 1024, 4096
    inputs = make_inputs(B, E, D, H, N)
    ref_grad_K, ref_grad_V = reference_grad_kv(*inputs)
    triton_grad_K, triton_grad_V = fused_grouped_grad_kv(*inputs)
    report_error(ref_grad_K, triton_grad_K, "grad_K")
    report_error(ref_grad_V, triton_grad_V, "grad_V")
    print("=> Done testing correctness")


if __name__ == "__main__":
    test_correctness()
