import torch
import triton
import triton.language as tl
import itertools

from triton.language.random import N_ROUNDS_DEFAULT


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


def reference_grouped_swiglu_fwd_example(
    w0_w2: torch.Tensor,
    x: torch.Tensor,
    group_sizes: torch.Tensor,
    block_M,
    block_N,
    block_K,
):
    """
    w0_w2: [B, E, 2 * Hidden, D]
    x: [B, num_tokens, D]
    group_sizes: [B, E]: number of tokens in each group
    """

    B, E, Hidden_times_2, D = w0_w2.shape

    num_N_tiles_per_group = [
        (g_size + block_N - 1) // block_N for g_size in group_sizes
    ]  # [B, E]

    num_M_tiles_in_hidden = (Hidden_times_2 + block_M - 1) // block_M

    total_N_blocks_per_batch = [sum(num_N_tiles_per_group[b]) for b in range(B)]

    # suppose given N_id, M_id, B_id, we can compute the group_id

    pid_bn = 0
    pid_m = 0

    # now find pid_n, and pid_b


@triton.autotune(
    configs=get_autotune_configs_small(),
    key=["M", "E", "K"],
)
@triton.jit
def _grouped_two_mm_swiglu_fwd_kernel(
    w0_w2_ptr,  # [B, E, 2 * Hidden, D]
    x_ptr,  # [B, num_permuted_tokens, D]
    group_sizes_ptr,  # [B, E]
    # total_N_blocks_per_batch: tl.constexpr,  # [B]
    output_ptr,  # [B, num_permuted_tokens, Hidden]
    B: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    N,  # num_tokens
    K: tl.constexpr,  # which is D, the reduce axis
    stride_w_b,  # = E * 2M * K
    stride_w_e,  # = 2M * K
    stride_w_m,  # = K
    stride_w_k,
    stride_x_b,
    stride_x_n,
    stride_x_k,
    stride_o_b,
    stride_o_m,
    stride_o_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    pid_bn = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # now, find pid_n, and pid_b
    pid_n = pid_bn
    pid_b = 0
    pid_E = 0

    remaining = pid_bn
    n_tile_start_index = 0

    # for batch_id in range(B):
    #     num_n_blocks_in_batch = tl.load(total_N_blocks_per_batch + batch_id)
    #     if pid_bn < num_n_blocks_in_batch:
    #         pid_b = batch_id
    #         break
    #     pid_bn -= num_n_blocks_in_batch

    # pid_n = pid_bn

    # for group_id in range(E):
    #     # num_n_tiles_in_group = tl.load(num_N_tiles_per_group + E * pid_b + group_id)
    #     group_size = tl.load(group_sizes_ptr + E * pid_b + group_id)
    #     num_n_tiles_in_group = tl.cdiv(group_size, block_N)
    #     if pid_n < num_n_tiles_in_group:
    #         pid_E = group_id
    #         break
    #     pid_n -= num_n_tiles_in_group
    #     n_tile_start_index += group_size

    # need to go over the group_sizes in a two-level loop!

    hit = False
    group_id_total = 0
    n_tile_group_endding_idx = N
    while (not hit) and group_id_total < E * B:
        batch_id = group_id_total // E
        group_id = group_id_total % E
        group_size = tl.load(group_sizes_ptr + E * batch_id + group_id)
        num_n_tiles_in_group = tl.cdiv(group_size, BLOCK_N)

        if group_id == 0:
            n_tile_start_index = 0

        if pid_n < num_n_tiles_in_group:
            pid_b = batch_id
            pid_E = group_id
            hit = True
            n_tile_group_endding_idx = n_tile_start_index + group_size
        else:
            pid_n -= num_n_tiles_in_group
            n_tile_start_index += group_size

        group_id_total += 1

    # found = False  # 0
    # n_tile_group_endding_idx = N  # for masking over the N dimension
    # for batch_id in range(B):
    #     # n_tile_start_index = 0
    #     n_tile_start_curr = 0
    #     for group_id in range(E):
    #         group_size = tl.load(group_sizes_ptr + E * batch_id + group_id)
    #         num_n_tiles_in_group = tl.cdiv(group_size, BLOCK_N)

    #         hit = (remaining < num_n_tiles_in_group) & (found == 0)

    #         n_tile_group_endding_idx = tl.where(
    #             hit, n_tile_start_curr + group_size, n_tile_group_endding_idx
    #         )

    #         pid_b = tl.where(hit, batch_id, pid_b)
    #         pid_E = tl.where(hit, group_id, pid_E)
    #         n_tile_start_index = tl.where(hit, n_tile_start_curr, n_tile_start_index)
    #         found = tl.where(hit, True, found)

    #         remaining = tl.where(found, remaining, remaining - num_n_tiles_in_group)
    #         n_tile_start_curr = tl.where(
    #             found, n_tile_start_curr, n_tile_start_curr + group_size
    #         )

    # if pid_n < num_n_tiles_in_group:
    #     pid_b = batch_id
    #     pid_E = group_id
    #     break
    # else:
    #     pid_n -= num_n_tiles_in_group
    #     n_tile_start_index += group_size

    # n_tile_start_index += remaining * BLOCK_N

    n_tile_start_index += pid_n * BLOCK_N

    if n_tile_start_index >= N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = n_tile_start_index + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    n_mask_1d = (offs_n < n_tile_group_endding_idx) & (offs_n < N)

    # Pointers base for this batch
    w0_batch_ptr = w0_w2_ptr + pid_b * stride_w_b + pid_E * stride_w_e
    w2_batch_ptr = w0_batch_ptr + M * stride_w_m
    x_batch_ptr = x_ptr + pid_b * stride_x_b
    o_batch_ptr = output_ptr + pid_b * stride_o_b

    # Accumulators in fp32
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k  # [BLOCK_K]
        k_mask = k_ids < K

        # Load tiles: W0/W2 shape -> (BLOCK_M, BLOCK_K), X shape -> (BLOCK_N, BLOCK_K)
        w0_ptrs = (
            w0_batch_ptr
            + (offs_m[:, None] * stride_w_m)
            + (k_ids[None, :] * stride_w_k)
        )
        w2_ptrs = (
            w2_batch_ptr
            + (offs_m[:, None] * stride_w_m)
            + (k_ids[None, :] * stride_w_k)
        )
        x_ptrs = (
            x_batch_ptr + (offs_n[:, None] * stride_x_n) + (k_ids[None, :] * stride_x_k)
        )

        w_mask = (offs_m[:, None] < M) & k_mask[None, :]
        # x_mask = (offs_n[:, None] < N) & k_mask[None, :]
        x_mask = n_mask_1d[:, None] & k_mask[None, :]

        # Loads as bf16; Triton will upcast in tl.dot to fp32 accumulators.
        w0 = tl.load(w0_ptrs, mask=w_mask, other=0).to(tl.bfloat16)
        w2 = tl.load(w2_ptrs, mask=w_mask, other=0).to(tl.bfloat16)
        x = tl.load(x_ptrs, mask=x_mask, other=0).to(tl.bfloat16)  # (BLOCK_N, BLOCK_K)

        # (M,K) x (K,N): we trans(x) to (BLOCK_K, BLOCK_N)
        acc0 += tl.dot(w0, tl.trans(x), out_dtype=tl.float32)
        acc2 += tl.dot(w2, tl.trans(x), out_dtype=tl.float32)

    # Apply SiLU in fp32 and fuse multiply
    y0 = acc0  # fp32
    y2 = acc2  # fp32
    # SiLU(x) = x * sigmoid(x)
    out = y2 * (y0 * tl.sigmoid(y0))

    # Store to bf16
    o_ptrs = (
        o_batch_ptr + (offs_m[:, None] * stride_o_m) + (offs_n[None, :] * stride_o_n)
    )
    # o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    o_mask = (offs_m[:, None] < M) & n_mask_1d[None, :]
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=o_mask)


def fused_two_mm_swiglu_triton(
    W0_W2: torch.Tensor,
    X: torch.Tensor,
    group_sizes: torch.Tensor,
):
    """
    Wraps the Triton kernel. Shapes:
      W0_W2: [B, E, 2M, K]  (bf16)
      X     : [B, num_permuted_tokens, K]  (bf16)
      group_sizes: [B, E]
      total_N_blocks_per_batch: [B]
    returns O: [B, num_permuted_tokens, Hidden] (bf16) where O = SiLU(W0 @ X^T) * (W2 @ X^T)
    """
    assert W0_W2.dtype == torch.bfloat16 and X.dtype == torch.bfloat16
    assert W0_W2.is_contiguous() and X.is_contiguous(), "W0_W2 and X must be contiguous"

    B, num_experts, M_times_2, K = W0_W2.shape
    Bx, N, Kx = X.shape
    assert Bx == B and Kx == K, "X must be [B, N, K] with matching B,K."

    M = M_times_2 // 2

    O = torch.empty((B, M, N), device=X.device, dtype=torch.bfloat16)

    # Strides (PyTorch: element strides)
    stride_w_b, stride_w_e, stride_w_m, stride_w_k = W0_W2.stride()
    stride_x_b, stride_x_n, stride_x_k = X.stride()
    stride_o_b, stride_o_m, stride_o_n = O.stride()

    # total_tiles = total_N_blocks_per_batch.sum().item()

    # grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)

    def grid(meta):
        return (
            (triton.cdiv(N, meta["BLOCK_N"]) + num_experts) * B,
            triton.cdiv(M, meta["BLOCK_M"]),
        )

    _grouped_two_mm_swiglu_fwd_kernel[grid](
        W0_W2,
        X,
        group_sizes,
        O,
        B,
        num_experts,
        M,
        N,
        K,
        stride_w_b,
        stride_w_e,
        stride_w_m,
        stride_w_k,
        stride_x_b,
        stride_x_n,
        stride_x_k,
        stride_o_b,
        stride_o_m,
        stride_o_n,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # BLOCK_K=BLOCK_K,
        # num_warps=num_warps,
        # num_stages=num_stages,
    )
    return O


def _test_correctness():

    from benchmark import report_error

    def reference_dense(w0_w2: torch.Tensor, x: torch.Tensor, _):
        w0_w2 = w0_w2[:, 0, :, :]
        w0, w2 = w0_w2.chunk(2, dim=1)
        y0 = torch.bmm(w0, x.transpose(1, 2))
        y2 = torch.bmm(w2, x.transpose(1, 2))
        y = torch.nn.functional.silu(y0) * y2
        return y

    def reference_grouped_swiglu_fwd_example(
        w0_w2: torch.Tensor,
        x: torch.Tensor,
        group_sizes: torch.Tensor,
    ):
        B, E, Hidden_times_2, D = w0_w2.shape

        all_batch_y_list = []
        for b_index in range(B):
            start_index = 0
            single_batch_y_list = []
            for e_index in range(E):
                group_size = group_sizes[b_index, e_index]

                x_i = x[b_index, start_index : start_index + group_size, :]

                start_index += group_size

                w0_w2_i = w0_w2[b_index, e_index, :, :]

                w0_i = w0_w2_i[:D, :]
                w2_i = w0_w2_i[D:, :]

                y1 = w0_i @ x_i.T
                y2 = w2_i @ x_i.T

                y = torch.nn.functional.silu(y1) * y2

                single_batch_y_list.append(y)

            single_batch_y = torch.cat(single_batch_y_list, dim=1)
            all_batch_y_list.append(single_batch_y)

        # [b, D, N]
        all_y = torch.stack(all_batch_y_list, dim=0)
        return all_y

    def make_inputs(B, E, Hidden_times_2, N, D):
        w0_w2 = torch.randn(
            B, E, Hidden_times_2, D, device="cuda", dtype=torch.bfloat16
        )
        x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16)

        mean_size = N // topK
        group_sizes = torch.randint(
            1, mean_size, (B, E), device="cuda", dtype=torch.int32
        )

        sum_group_sizes = group_sizes.sum(dim=1)

        extra_size = N - sum_group_sizes
        group_sizes[:, -1] += extra_size

        return w0_w2, x, group_sizes

    B, E, Hidden_times_2, N, D, topK = 4, 4, 2048, 4096, 1024, 2
    _inputs = make_inputs(B, E, Hidden_times_2, N, D)

    fp32_inputs = [
        _inputs[0].to(torch.float32),
        _inputs[1].to(torch.float32),
        _inputs[2],
    ]
    ref_y = reference_grouped_swiglu_fwd_example(*fp32_inputs)
    triton_y = fused_two_mm_swiglu_triton(*_inputs)

    report_error(ref_y, triton_y, "output")

    run_speed_benchmark = True

    if run_speed_benchmark:
        from benchmark import run_benchmark_fwd_only

        name_2_fn = {
            "triton": fused_two_mm_swiglu_triton,
            "triton_compile": torch.compile(fused_two_mm_swiglu_triton),
            # "reference": reference_grouped_swiglu_fwd_example,
            "reference_dense": reference_dense,
            "reference_dense_compile": torch.compile(reference_dense),
        }

        B_list = [4]
        E_list = [2, 4, 8]

        N_list = [2048, 4096, 8192]
        D_list = [384, 512, 1024]

        for B in B_list:
            for E in E_list:
                for N in N_list:
                    for D in D_list:
                        H = D * 2
                        _inputs = make_inputs(B, E, H * 2, N, D)
                        FLOPS = B * N * D * H * 4

                        print(f"Running B={B}, E={E}, N={N}, D={D}, H={H}")
                        run_benchmark_fwd_only(
                            name_2_fn,
                            _inputs,
                            repeats=10,
                            warmup=10,
                            FLOPS=FLOPS,
                        )
                        print("--------------------------------")


if __name__ == "__main__":
    _test_correctness()
