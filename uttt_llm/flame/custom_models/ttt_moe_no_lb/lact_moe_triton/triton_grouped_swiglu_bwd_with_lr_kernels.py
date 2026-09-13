import triton
import triton.language as tl
import torch
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
    configs=get_autotune_configs_small(),  # you already defined this above
    key=["M", "E", "K"],
)
@triton.jit
def _grouped_swiglu_three_bmm_with_lr_kernel(
    w0_w2_ptr,  # [B, E, 2M, K]
    w1_ptr,  # [B, E, K, M]
    x_ptr,  # [B, N_total, K]
    v_ptr,  # [B, N_total, K]
    lr0_ptr,  # [B, N_total]
    lr1_ptr,  # [B, N_total]
    lr2_ptr,  # [B, N_total]
    group_sizes_ptr,  # [B, E] (int32)
    dy0_dy2_ptr,  # [B, 2M, N_total]
    hidden_ptr,  # [B, M,  N_total]
    # problem sizes
    B: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    N_total,  # runtime
    K: tl.constexpr,
    # strides for W0_W2 [B, E, 2M, K]
    s_w0w2_b,
    s_w0w2_e,
    s_w0w2_m,
    s_w0w2_k,
    # strides for W1 [B, E, K, M]
    s_w1_b,
    s_w1_e,
    s_w1_k,
    s_w1_m,
    # strides for X [B, N_total, K]
    s_x_b,
    s_x_n,
    s_x_k,
    # strides for V [B, N_total, K]
    s_v_b,
    s_v_n,
    s_v_k,
    # strides for lr* [B, N]
    s_lr_b,
    s_lr_n,
    # strides for DY0_DY2 [B, 2M, N_total]
    s_dy_b,
    s_dy_m,
    s_dy_n,
    # strides for Hidden [B, M, N_total]
    s_h_b,
    s_h_m,
    s_h_n,
    # casting control
    out_dtype: tl.constexpr,  # "bf16" | "fp16" | "float32"
    # meta-tiling
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # --- program ids ---
    pid_bn = tl.program_id(axis=0)  # flattens (b, e, n-tiles-in-group)
    pid_m = tl.program_id(axis=1)  # m tiles

    # --- decode (batch, expert, n-tile-in-group) from pid_bn by scanning group_sizes ---
    pid_b = 0
    pid_e = 0
    pid_n = pid_bn
    n_tile_start = 0
    n_tile_group_end = N_total  # for boundary mask
    hit = False

    # Walk groups in (b, e) order; reset start when e == 0 (new batch)
    group_linear = 0
    while (not hit) and (group_linear < B * E):
        b = group_linear // E
        e = group_linear % E

        if e == 0:
            # new batch: reset tile start to 0 within the batch
            n_tile_start = 0

        gsz = tl.load(group_sizes_ptr + b * E + e)  # int32
        num_tiles = tl.cdiv(gsz, BLOCK_N)

        if pid_n < num_tiles:
            pid_b = b
            pid_e = e
            n_tile_group_end = n_tile_start + gsz
            hit = True
        else:
            pid_n -= num_tiles
            n_tile_start += gsz

        group_linear += 1

    # advance to the tile inside the chosen group
    n_tile_start += pid_n * BLOCK_N

    # if we oversubscribed grid, early-exit
    if n_tile_start >= N_total:
        return

    # --- tile offsets ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    # N mask is bounded by the chosen group's true end and by N_total
    mask_n_1d = (offs_n < n_tile_group_end) & (offs_n < N_total)

    # --- base pointers for this (b, e) ---
    w0_base = w0_w2_ptr + pid_b * s_w0w2_b + pid_e * s_w0w2_e
    w2_base = w0_base + M * s_w0w2_m
    w1_base = w1_ptr + pid_b * s_w1_b + pid_e * s_w1_e
    x_base = x_ptr + pid_b * s_x_b
    v_base = v_ptr + pid_b * s_v_b
    dy_base = dy0_dy2_ptr + pid_b * s_dy_b
    h_base = hidden_ptr + pid_b * s_h_b

    # --- accumulators (fp32) ---
    acc_y0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_y2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --- K loop ---
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k
        mask_k = k_ids < K

        # W0/W2: [M,K] tiles
        a0_ptrs = w0_base + (offs_m[:, None] * s_w0w2_m + k_ids[None, :] * s_w0w2_k)
        a2_ptrs = w2_base + (offs_m[:, None] * s_w0w2_m + k_ids[None, :] * s_w0w2_k)
        w_mask = mask_m[:, None] & mask_k[None, :]

        a0 = tl.load(a0_ptrs, mask=w_mask, other=0.0)
        a2 = tl.load(a2_ptrs, mask=w_mask, other=0.0)

        # W1^T as [M,K] (by addressing W1[k, m])
        a1_ptrs = w1_base + (k_ids[None, :] * s_w1_k + offs_m[:, None] * s_w1_m)
        a1 = tl.load(a1_ptrs, mask=w_mask, other=0.0)

        # X^T and V^T as [K,N]
        bx_ptrs = x_base + (offs_n[None, :] * s_x_n + k_ids[:, None] * s_x_k)
        bv_ptrs = v_base + (offs_n[None, :] * s_v_n + k_ids[:, None] * s_v_k)
        xn_mask = mask_k[:, None] & mask_n_1d[None, :]

        bx = tl.load(bx_ptrs, mask=xn_mask, other=0.0)
        bv = tl.load(bv_ptrs, mask=xn_mask, other=0.0)

        # Three GEMMs (fp32 accumulate)
        acc_y0 += tl.dot(a0, bx, out_dtype=tl.float32)
        acc_y2 += tl.dot(a2, bx, out_dtype=tl.float32)
        acc_dh += tl.dot(a1, bv, out_dtype=tl.float32)

    # --- epilogue in fp32 ---
    y0 = acc_y0
    y2 = acc_y2
    dh = acc_dh

    sigma = tl.sigmoid(y0)
    swish = sigma * y0

    hidden_tile = swish * y2
    dy0_tile = sigma * y2 * dh * (1.0 + y0 * (1.0 - sigma))
    dy2_tile = swish * dh

    # --- Load per-(B, N) scaling vectors (once per tile), cast to fp32 ---
    lr0_batch = lr0_ptr + pid_b * s_lr_b
    lr1_batch = lr1_ptr + pid_b * s_lr_b
    lr2_batch = lr2_ptr + pid_b * s_lr_b

    lr0_vec = tl.load(lr0_batch + offs_n * s_lr_n, mask=mask_n_1d, other=0.0).to(
        tl.float32
    )
    lr1_vec = tl.load(lr1_batch + offs_n * s_lr_n, mask=mask_n_1d, other=0.0).to(
        tl.float32
    )
    lr2_vec = tl.load(lr2_batch + offs_n * s_lr_n, mask=mask_n_1d, other=0.0).to(
        tl.float32
    )

    dy0_tile *= lr0_vec[None, :]
    dy2_tile *= lr2_vec[None, :]
    hidden_tile *= lr1_vec[None, :]

    # --- cast & store ---
    out_dtype_tl = (
        tl.float16
        if out_dtype == "fp16"
        else tl.bfloat16 if out_dtype == "bf16" else tl.float32
    )

    mask_out = mask_m[:, None] & mask_n_1d[None, :]

    # DY0 at [b, 0:M, n], DY2 at [b, M:2M, n]
    dy0_ptrs = dy_base + (offs_m[:, None] * s_dy_m + offs_n[None, :] * s_dy_n)
    dy2_ptrs = (
        dy_base + (M * s_dy_m) + (offs_m[:, None] * s_dy_m + offs_n[None, :] * s_dy_n)
    )
    h_ptrs = h_base + (offs_m[:, None] * s_h_m + offs_n[None, :] * s_h_n)

    tl.store(dy0_ptrs, dy0_tile.to(out_dtype_tl), mask=mask_out)
    tl.store(dy2_ptrs, dy2_tile.to(out_dtype_tl), mask=mask_out)
    tl.store(h_ptrs, hidden_tile.to(out_dtype_tl), mask=mask_out)


def swiglu_backward_three_bmm_grouped_with_lr_triton(
    W0_W2: torch.Tensor,  # [B, E, 2M, K], bf16/fp16/fp32
    W1: torch.Tensor,  # [B, E, K, M]
    X: torch.Tensor,  # [B, N_total, K]  (permuted & grouped)
    V: torch.Tensor,  # [B, N_total, K]  (permuted & grouped)
    lr0: torch.Tensor,  # [B, N_total]
    lr1: torch.Tensor,  # [B, N_total]
    lr2: torch.Tensor,  # [B, N_total]
    group_sizes: torch.Tensor,  # [B, E], int32
):
    """
    Returns:
      DY0_DY2: [B, 2M, N_total]
      Hidden : [B, M,  N_total]
    """
    assert (
        W0_W2.dtype == torch.bfloat16 and V.dtype == torch.bfloat16
    ), "W0_W2 and V must be bf16"
    assert (
        W0_W2.is_contiguous()
        and W1.is_contiguous()
        and X.is_contiguous()
        and V.is_contiguous()
    )
    assert W0_W2.ndim == 4 and W1.ndim == 4 and X.ndim == 3 and V.ndim == 3
    B, E, M2, K = W0_W2.shape
    assert M2 % 2 == 0, "2M must be even"
    M = M2 // 2
    Bx, N, K = X.shape

    # Allocate outputs
    Hidden = torch.empty((B, M, N), device=X.device, dtype=X.dtype)
    DY0_DY2 = torch.empty((B, 2 * M, N), device=X.device, dtype=X.dtype)

    # Strides (element strides)
    s_w0w2_b, s_w0w2_e, s_w0w2_m, s_w0w2_k = W0_W2.stride()
    s_w1_b, s_w1_e, s_w1_k, s_w1_m = W1.stride()
    s_x_b, s_x_n, s_x_k = X.stride()
    s_v_b, s_v_n, s_v_k = V.stride()
    s_dy_b, s_dy_m, s_dy_n = DY0_DY2.stride()
    s_h_b, s_h_m, s_h_n = Hidden.stride()
    s_lr_b, s_lr_n = lr0.stride(0), lr0.stride(1)

    out_dtype_str = (
        "float32"
        if X.dtype == torch.float32
        else "bf16" if X.dtype == torch.bfloat16 else "fp16"
    )

    # Grid: axis-0 covers (sum_g ceil(group_sizes[b,g]/BLOCK_N)) per batch.
    # Using an upper-bound like (ceil(N_total/BLOCK_N) + E) keeps things simple; kernel early-returns if over.
    def grid(meta):
        return (
            (triton.cdiv(N, meta["BLOCK_N"]) + E) * B,
            triton.cdiv(M, meta["BLOCK_M"]),
        )

    _grouped_swiglu_three_bmm_with_lr_kernel[grid](
        W0_W2,
        W1,
        X,
        V,
        lr0,
        lr1,
        lr2,
        group_sizes.to(dtype=torch.int32),
        DY0_DY2,
        Hidden,
        B,
        E,
        M,
        N,
        K,
        # strides
        s_w0w2_b,
        s_w0w2_e,
        s_w0w2_m,
        s_w0w2_k,
        s_w1_b,
        s_w1_e,
        s_w1_k,
        s_w1_m,
        s_x_b,
        s_x_n,
        s_x_k,
        s_v_b,
        s_v_n,
        s_v_k,
        s_lr_b,
        s_lr_n,
        s_dy_b,
        s_dy_m,
        s_dy_n,
        s_h_b,
        s_h_m,
        s_h_n,
        out_dtype=out_dtype_str,
    )
    return DY0_DY2, Hidden


def reference_grouped_swiglu_bwd_with_lr_example(
    w0_w2: torch.Tensor,
    w1: torch.Tensor,
    x: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    group_sizes: torch.Tensor,
):
    B, E, Hidden_times_2, D = w0_w2.shape
    Hidden = Hidden_times_2 // 2

    dw0_dw2_ret = torch.zeros_like(w0_w2)
    dw1_ret = torch.zeros_like(w1)
    for b_index in range(B):
        start_index = 0
        single_batch_y_list = []
        single_batch_hidden_list = []
        for e_index in range(E):
            group_size = group_sizes[b_index, e_index]

            x_i = x[
                b_index, start_index : start_index + group_size, :
            ]  # [group_size, D]
            v_i = v[
                b_index, start_index : start_index + group_size, :
            ]  # [group_size, D]

            lr0_i = lr0[b_index, start_index : start_index + group_size]
            lr1_i = lr1[b_index, start_index : start_index + group_size]
            lr2_i = lr2[b_index, start_index : start_index + group_size]

            start_index += group_size

            w0_w2_i = w0_w2[b_index, e_index, :, :]

            w0_i = w0_w2_i[:Hidden, :]
            w2_i = w0_w2_i[Hidden:, :]
            w1_i = w1[b_index, e_index, :, :]  # [D, Hidden]

            # [D, group_size]
            y1 = w0_i @ x_i.T
            y2 = w2_i @ x_i.T

            dh = w1_i.T @ v_i.T

            sigma = torch.sigmoid(y1)
            swish = sigma * y1

            hidden = lr1_i.unsqueeze(dim=0) * swish * y2

            DY1 = lr0_i.unsqueeze(dim=0) * sigma * y2 * dh * (1.0 + y1 * (1.0 - sigma))
            DY2 = lr2_i.unsqueeze(dim=0) * swish * dh

            dy0_dy2 = torch.cat([DY1, DY2], dim=0)  # [2hidden, group_size]

            d_w0_w2 = dy0_dy2 @ x_i
            d_w1 = (hidden @ v_i).T

            dw0_dw2_ret[b_index, e_index, :, :] = d_w0_w2
            dw1_ret[b_index, e_index, :, :] = d_w1

    return dw0_dw2_ret, dw1_ret


def test_correctness():
    from benchmark import report_error

    def make_inputs(B, E, D, H, N):
        w0_w2 = torch.randn(B, E, 2 * H, D, device="cuda", dtype=torch.bfloat16)
        w1 = torch.randn(B, E, D, H, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)
        lr0 = torch.randn(B, N, device="cuda", dtype=torch.float32)
        lr1 = torch.randn(B, N, device="cuda", dtype=torch.float32)
        lr2 = torch.randn(B, N, device="cuda", dtype=torch.float32)
        mean_size = N // E
        group_sizes = torch.randint(
            1, mean_size, (B, E), device="cuda", dtype=torch.int32
        )
        sum_group_sizes = group_sizes.sum(dim=1)
        extra_size = N - sum_group_sizes
        group_sizes[:, -1] += extra_size

        return w0_w2, w1, x, v, lr0, lr1, lr2, group_sizes

    B, E, D, H, N = 4, 4, 1024, 1024, 4096
    _inputs = make_inputs(B, E, D, H, N)
    fp32_inputs = [
        _inputs[0].to(torch.float32),
        _inputs[1].to(torch.float32),
        _inputs[2].to(torch.float32),
        _inputs[3].to(torch.float32),
        _inputs[4].to(torch.float32),
        _inputs[5].to(torch.float32),
        _inputs[6].to(torch.float32),
        _inputs[7],
    ]
    ref_y, ref_hidden = reference_grouped_swiglu_bwd_with_lr_example(*fp32_inputs)
    triton_y, triton_hidden = swiglu_backward_three_bmm_grouped_with_lr_triton(*_inputs)
    report_error(ref_y, triton_y, "output")
    report_error(ref_hidden, triton_hidden, "hidden")


if __name__ == "__main__":
    test_correctness()
