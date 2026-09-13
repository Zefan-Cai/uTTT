import torch

from torch.autograd.function import once_differentiable

try:
    from triton_grouped_swiglu_bwd_with_lr_kernels import (
        swiglu_backward_three_bmm_grouped_with_lr_triton,
    )

    from triton_grouped_gemm import (
        grouped_gemm_transpose_triton,
        grouped_gemm_bwd_triton,
    )
    from triton_pointwise_kernels import (
        triton_swiglu_bwd_bwd_fused_cat_inp_out,
        ref_pytorch_swiglu_bwd_bwd_fused_cat_inp_out,
    )
    from triton_lact_bwd_grouped_four_mm import (
        grouping_four_grouped_gemm_triton,
    )
    from triton_lact_bwd_grouped_grad_kv import (
        fused_grouped_grad_kv,
    )
    from triton_lact_bwd_grouped_two_mm_grad_fw import (
        grouped_two_gemm_fw_bwd_triton,
    )
except ImportError:
    from .triton_grouped_swiglu_bwd_with_lr_kernels import (
        swiglu_backward_three_bmm_grouped_with_lr_triton,
    )

    from .triton_grouped_gemm import (
        grouped_gemm_transpose_triton,
        grouped_gemm_bwd_triton,
    )
    from .triton_pointwise_kernels import (
        triton_swiglu_bwd_bwd_fused_cat_inp_out,
        ref_pytorch_swiglu_bwd_bwd_fused_cat_inp_out,
    )
    from .triton_lact_bwd_grouped_four_mm import (
        grouping_four_grouped_gemm_triton,
    )
    from .triton_lact_bwd_grouped_grad_kv import (
        fused_grouped_grad_kv,
    )
    from .triton_lact_bwd_grouped_two_mm_grad_fw import (
        grouped_two_gemm_fw_bwd_triton,
    )


class FusedLactSwiGLUFFNBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W0_W2, W1, K, V, lr0, lr1, lr2, group_sizes):
        """
        Args:
            W0_W2:    [B, E, 2 * Hidden, D]
            W1:        [B, E, K, M] or [B, E, D, Hidden]
            K, V:      [M, N, K] or [B, num_Tokens, D]
            lr0, lr1, lr2:    [B, N]
            group_sizes: [B, E]

        Outs:
            Hidden: [B, N, K] or [B, num_tokens, Hidden]
            dW0_W2: [B, E, 2 * Hidden, D]
            dW1: [B, E, D, Hidden]
        Total FLOPS: 12 * B * Hidden * D * num_tokens
        """

        ## This fuse three GEMMs togeather, and does element-wise epilogues
        ## to compute DY0, DY2, and Hidden (multiplied with lr0, lr1, lr2).
        # This kernel has high register pressure!
        # W0 = W0.contiguous()
        # W1 = W1.contiguous()
        # W2 = W2.contiguous()
        # K = K.contiguous()
        # V = V.contiguous()
        # lr0 = lr0.contiguous()
        # lr1 = lr1.contiguous()
        # lr2 = lr2.contiguous()

        group_sizes = group_sizes.to(torch.int32).contiguous()

        #### without this triton kernel, we will materize Y0, Y2, Dhidden;  DY0_with_LR0, DY2_with_LR2, Hidden_with_LR1;
        #### 3 + 3 + 3.   read, write.
        DY0_DY2, Hidden = swiglu_backward_three_bmm_grouped_with_lr_triton(
            W0_W2,
            W1,
            K,
            V,
            lr0,
            lr1,
            lr2,
            group_sizes,
        )

        # groupping below two GEMM togeather can futher reduce launching overhead.
        # [B, 2 * Hidden, num_tokens] @ [B, num_tokens, D] -> [B, E, 2 * Hidden, D]
        # DW0_DW2 = grouped_gemm_bwd_triton(DY0_DY2, K, group_sizes)
        # # [B, D, Hidden] = [B, D, num_tokens] @ [B, num_tokens, Hidden]
        # DW1 = grouped_gemm_bwd_triton(
        #     V.transpose(1, 2).contiguous(),
        #     Hidden.transpose(1, 2).contiguous(),
        #     group_sizes,
        # )

        DW0_DW2, DW1 = grouped_two_gemm_fw_bwd_triton(
            DY0_DY2, K, Hidden, V, group_sizes
        )

        # we don't need to save DY0, DY2, and Hidden, because we will compute them again in the backward pass.
        ctx.save_for_backward(W0_W2, W1, K, V, lr0, lr1, lr2, group_sizes)

        return DW0_DW2, DW1

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dw0_dw2, grad_dw1):
        """
        Args:
            grad_dw0_dw2: [B, 2 * Hidden, D]
            grad_dw1: [B, D, Hidden]
        Outs:
            grad_W0: [B, Hidden, D]
            grad_W1: [B, D, Hidden]
            grad_W2: [B, Hidden, D]
            grad_K: [B, D, num_tokens]
            grad_V: [B, num_tokens, D]
            grad_lr0: [B, 1]
            grad_lr1: [B, 1]
            grad_lr2: [B, 1]

        Total FLOPS: 24 * B * Hidden * D * num_tokens + 6 * B * Hidden * D * num_tokens
        # 24 for backward matmuls, and 6 for forward recomputation.
        """

        W0_W2, W1, K, V, lr0, lr1, lr2, group_sizes = ctx.saved_tensors

        # -> [B, 2 * Hidden, num_tokens]
        # W0_W2 @ K.transpose(1, 2)
        # Y0_Y2 = (
        #     grouped_gemm_transpose_triton(
        #         W0_W2, K.transpose(1, 2).contiguous(), group_sizes
        #     )
        #     .transpose(1, 2)
        #     .contiguous()
        # )
        # # W1.transpose(2, 3) @ V.transpose(1, 2)
        # DHidden = (
        #     grouped_gemm_transpose_triton(
        #         W1.transpose(2, 3).contiguous(),
        #         V.transpose(1, 2).contiguous(),
        #         group_sizes,
        #     )
        #     .transpose(1, 2)
        #     .contiguous()
        # )

        # grad_Hidden_with_lr1 = (
        #     grouped_gemm_transpose_triton(
        #         grad_dw1.transpose(2, 3).contiguous(),
        #         V.transpose(1, 2).contiguous(),
        #         group_sizes,
        #     )
        #     .transpose(1, 2)
        #     .contiguous()
        # )

        # [B, Hidden, num_tokens] = [B, Hidden, D] @ [B, num_tokens, D].T
        # grad_DY0_with_lr0, grad_DY2_with_lr2 = torch.ops.lact.two_mm_same_inp(
        #     grad_dw0.contiguous(), grad_dw2.contiguous(), K.contiguous(), False, True
        # )

        # [B, 2 * Hidden, D]
        # grad_DY0_with_lr0_and_grad_DY2_with_lr2 = (
        #     grouped_gemm_transpose_triton(
        #         grad_dw0_dw2.contiguous(), K.transpose(1, 2).contiguous(), group_sizes
        #     )
        #     .transpose(1, 2)
        #     .contiguous()
        # )

        (
            Y0_Y2,
            grad_DY0_with_lr0_and_grad_DY2_with_lr2,
            DHidden,
            grad_Hidden_with_lr1,
        ) = grouping_four_grouped_gemm_triton(
            W0_W2,
            grad_dw0_dw2.contiguous(),
            W1,
            grad_dw1.contiguous(),
            K,
            V,
            group_sizes,
        )

        #### Next, we do tones of element-wise ops.
        #### These element-wise ops are compiled with torch.compile. one graph?
        # print("lr0.shape: ", lr0.shape)
        # print("lr1.shape: ", lr1.shape)
        # print("lr2.shape: ", lr2.shape)
        # print(
        #     "grad_DY0_with_lr0_and_grad_DY2_with_lr2.shape: ",
        #     grad_DY0_with_lr0_and_grad_DY2_with_lr2.shape,
        # )
        # print("grad_Hidden_with_lr1.shape: ", grad_Hidden_with_lr1.shape)
        # print("Y0_Y2.shape: ", Y0_Y2.shape)
        # print("DHidden.shape: ", DHidden.shape)
        # print("grad_dw0_dw2.shape: ", grad_dw0_dw2.shape)
        # print("grad_dw1.shape: ", grad_dw1.shape)
        # print("group_sizes.shape: ", group_sizes.shape)

        (
            grad_DHidden,  # [B, Hidden, num_tokens]
            grad_Y0_Y2,  # [B, 2 * Hidden, num_tokens]
            grad_lr0,  # [B, L]
            grad_lr1,  # [B, L]
            grad_lr2,  # [B, L]
            DY0_with_lr0_and_DY2_with_lr2,  # [B, 2 * Hidden, L]
            Hidden_with_lr1,  # [B, Hidden, L]
            # ) = triton_swiglu_bwd_bwd_fused_cat_inp_out(
        ) = ref_pytorch_swiglu_bwd_bwd_fused_cat_inp_out(
            DHidden,  # [B, Hidden, num_tokens]
            Y0_Y2,  # [B, 2 * Hidden, num_tokens]
            lr0.contiguous(),  # [B, L]
            lr1.contiguous(),  # [B, L]
            lr2.contiguous(),  # [B, L]
            grad_DY0_with_lr0_and_grad_DY2_with_lr2,  # [B, 2 * Hidden, num_tokens]
            grad_Hidden_with_lr1,  # [B, Hidden, num_tokens]
        )

        # grad_K = torch.bmm(
        #     DY0_with_lr0_and_DY2_with_lr2.transpose(1, 2), grad_dw0_dw2
        # ) + torch.bmm(grad_Y0_Y2.transpose(1, 2), W0_W2)

        # grad_K = grouped_gemm_transpose_triton(
        #     grad_dw0_dw2.contiguous().transpose(2, 3).contiguous(),
        #     DY0_with_lr0_and_DY2_with_lr2.contiguous(),
        #     group_sizes,
        # )

        # grad_K = grad_K + grouped_gemm_transpose_triton(
        #     W0_W2.contiguous().transpose(2, 3).contiguous(),
        #     grad_Y0_Y2.contiguous(),
        #     group_sizes,
        # )

        # grad_V = grouped_gemm_transpose_triton(
        #     W1, grad_DHidden.contiguous(), group_sizes
        # )
        # grad_V = grad_V + grouped_gemm_transpose_triton(
        #     grad_dw1.contiguous(),
        #     Hidden_with_lr1.contiguous(),
        #     group_sizes,
        # )
        grad_K, grad_V = fused_grouped_grad_kv(
            grad_dw0_dw2.contiguous(),
            W0_W2,
            DY0_with_lr0_and_DY2_with_lr2,
            grad_Y0_Y2,
            W1,
            grad_dw1.contiguous(),
            grad_DHidden,
            Hidden_with_lr1,
            group_sizes,
        )

        #### For below three matmuls, occupancy is the key, cause their dimension might be small.

        # [B, D, num_tokens] @ [B, num_tokens, Hidden].T -> [B, D, Hidden]
        # grad_W1 = torch.bmm(V.transpose(1, 2), grad_DHidden.transpose(1, 2))
        # grad_W1 = grouped_gemm_bwd_triton(
        #     V.transpose(1, 2).contiguous(),
        #     grad_DHidden.transpose(1, 2).contiguous(),
        #     group_sizes,
        # )

        # grad_W0_W2 = grouped_gemm_bwd_triton(
        #     grad_Y0_Y2.contiguous(), K.contiguous(), group_sizes
        # )

        grad_W0_W2, grad_W1 = grouped_two_gemm_fw_bwd_triton(
            grad_Y0_Y2,
            K,
            grad_DHidden,
            V,
            group_sizes,
        )

        return (
            grad_W0_W2,
            grad_W1,
            grad_K,
            grad_V,
            grad_lr0,
            grad_lr1,
            grad_lr2,
            None,
        )


# fused_lact_swiglu_ffn_fast_weight_grads = FusedLactSwiGLUFFNBwd.apply


def grouped_lact_swiglu_ffn_fast_weight_grads(
    W0_W2, W1, K, V, lr0, lr1, lr2, group_sizes
):
    return FusedLactSwiGLUFFNBwd.apply(W0_W2, W1, K, V, lr0, lr1, lr2, group_sizes)


########################################################
# Test and Benchmark code below
########################################################


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

            w0_i = w0_w2_i[:D, :]
            w2_i = w0_w2_i[D:, :]
            w1_i = w1[b_index, e_index, :, :]  # [D, Hidden]

            y1 = w0_i @ x_i.T
            y2 = w2_i @ x_i.T

            dh = w1_i.T @ v_i.T

            sigma = torch.sigmoid(y1)
            swish = sigma * y1

            hidden = lr1_i * swish * y2

            DY1 = lr0_i * sigma * y2 * dh * (1.0 + y1 * (1.0 - sigma))
            DY2 = lr2_i * swish * dh

            dy0_dy2 = torch.cat([DY1, DY2], dim=0)  # [2hidden, group_size]

            d_w0_w2 = dy0_dy2 @ x_i
            d_w1 = (hidden @ v_i).T

            dw0_dw2_ret[b_index, e_index, :, :] = d_w0_w2
            dw1_ret[b_index, e_index, :, :] = d_w1

    return dw0_dw2_ret, dw1_ret


def profile_backward_function_test():
    def _make_inputs(B, M, K, N, lr_dtype=torch.float32):
        W0_W2 = torch.randn(
            B, 2 * M, K, device="cuda", dtype=torch.bfloat16, requires_grad=False
        )
        W0_W2 = W0_W2.requires_grad_(True)
        W1 = torch.randn(
            B, K, M, device="cuda", dtype=torch.bfloat16, requires_grad=False
        )
        W1 = W1.requires_grad_(True)
        K_input = torch.randn(
            B, N, K, device="cuda", dtype=torch.bfloat16, requires_grad=False
        )
        K_input = K_input.requires_grad_(True)
        V = torch.randn(
            B, N, K, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        lr0 = (
            torch.randn(B, N, device="cuda", dtype=lr_dtype, requires_grad=False) * 0.1
        )
        lr1 = (
            torch.randn(B, N, device="cuda", dtype=lr_dtype, requires_grad=False) * 0.1
        )
        lr2 = (
            torch.randn(B, N, device="cuda", dtype=lr_dtype, requires_grad=False) * 0.1
        )
        grad_dw0_dw2 = torch.randn(
            B, 2 * M, K, device="cuda", dtype=torch.bfloat16, requires_grad=False
        )
        grad_dw1 = torch.randn(
            B, K, M, device="cuda", dtype=torch.bfloat16, requires_grad=False
        )
        return W0_W2, W1, K_input, V, lr0, lr1, lr2, grad_dw0_dw2, grad_dw1

    import torch._dynamo as dynamo

    _inputs = _make_inputs(1, 512, 1024, 8192)

    explanation = dynamo.explain(torch.compile(_backward_function_test))(*_inputs)
    with open("compile_explanation_fw_grad_triton_pointwise_backward.txt", "w") as f:
        f.write(str(explanation))
    import torch.profiler as profiler

    with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
        output = torch.compile(_backward_function_test)(*_inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    with open("compile_profiling_fw_grad_triton_pointwise_backward.txt", "w") as f:
        f.write(str(prof.key_averages().table(sort_by="cuda_time_total")))
    print("=> Done profiling")


def make_inputs(B, E, D, H, N, lr_dtype=torch.float32, require_grad=True):
    w0_w2 = torch.randn(
        B, E, 2 * H, D, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )
    w1 = torch.randn(
        B, E, D, H, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )
    x = torch.randn(
        B, N, D, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )
    v = torch.randn(
        B, N, D, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )
    lr0 = torch.randn(B, N, device="cuda", dtype=lr_dtype, requires_grad=require_grad)
    lr1 = torch.randn(B, N, device="cuda", dtype=lr_dtype, requires_grad=require_grad)
    lr2 = torch.randn(B, N, device="cuda", dtype=lr_dtype, requires_grad=require_grad)
    mean_size = N // E
    group_sizes = torch.randint(1, mean_size, (B, E), device="cuda", dtype=torch.int32)
    sum_group_sizes = group_sizes.sum(dim=1)
    extra_size = N - sum_group_sizes
    group_sizes[:, -1] += extra_size

    return w0_w2, w1, x, v, lr0, lr1, lr2, group_sizes


def test_fwd_bwd_match(
    B=4,
    E=4,
    M=512,  # H
    K=1024,  # D
    N=8192,
    seed=0,
):
    from benchmark import report_error

    """
    Compare FusedSwiGLUFFN with the reference formula:
        O = W1 @ (SiLU(W0 @ X) * (W2 @ X))
    for consistency in forward output and backward gradients.

    Notes:
    - The reference implementation uses float32 for computation and gradients.
    - The Fused path uses:
        W0/W2/X -> bfloat16, W1 -> float32
      and triggers the custom backward in this file via autograd.
    - Prints whether forward/grad match and the maximum absolute error.
    """
    import torch

    assert torch.cuda.is_available(), "CUDA environment is required to run this test"
    # torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())

    lr_dtype = torch.float32
    W0_W2, W1, K_input, V, lr0, lr1, lr2, group_sizes = make_inputs(
        B, E, K, M, N, lr_dtype=lr_dtype
    )

    ref_dtype = torch.float32
    W0_W2_ref = W0_W2.detach().to(ref_dtype).requires_grad_(True)
    W1_ref = W1.detach().to(ref_dtype).requires_grad_(True)
    K_input_ref = K_input.detach().to(ref_dtype).requires_grad_(True)
    V_ref = V.detach().to(ref_dtype).requires_grad_(True)
    lr0_ref = lr0.detach().to(lr_dtype).requires_grad_(True)
    lr1_ref = lr1.detach().to(lr_dtype).requires_grad_(True)
    lr2_ref = lr2.detach().to(lr_dtype).requires_grad_(True)

    DW0_DW2, DW1 = grouped_lact_swiglu_ffn_fast_weight_grads(
        W0_W2, W1, K_input, V, lr0, lr1, lr2, group_sizes
    )  # [B, N, K]
    loss_fused = DW0_DW2.sum() + DW1.sum() * 2
    loss_fused.backward()

    DW0_DW2_ref, DW1_ref = reference_grouped_swiglu_bwd_with_lr_example(
        W0_W2_ref,
        W1_ref,
        K_input_ref,
        V_ref,
        lr0_ref,
        lr1_ref,
        lr2_ref,
        group_sizes,
    )
    loss_ref = DW0_DW2_ref.sum() + DW1_ref.sum() * 2
    loss_ref.backward()

    print("Shape of DW0_DW2, DW1: ", DW0_DW2.shape, DW1.shape)
    print(
        "Shape of DW0_DW2_ref, DW1_ref: ",
        DW0_DW2_ref.shape,
        DW1_ref.shape,
    )

    # Backward comparison

    report_error(DW0_DW2_ref, DW0_DW2, "DW0_DW2")
    report_error(DW1_ref, DW1, "DW1")

    report_error(W0_W2_ref.grad, W0_W2.grad, "W0_W2_grad")
    report_error(W1_ref.grad, W1.grad, "W1_grad")
    report_error(K_input_ref.grad, K_input.grad, "K_grad")
    report_error(V_ref.grad, V.grad, "V_grad")
    report_error(lr0_ref.grad, lr0.grad, "lr0_grad")
    report_error(lr1_ref.grad, lr1.grad, "lr1_grad")
    report_error(lr2_ref.grad, lr2.grad, "lr2_grad")

    # do bf16 fwd and bwd for pytorch reference
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        bf16_W0_W2_ref = W0_W2_ref.detach().to(torch.bfloat16).requires_grad_(True)
        bf16_W1_ref = W1_ref.detach().to(torch.bfloat16).requires_grad_(True)
        bf16_K_input_ref = K_input_ref.detach().to(torch.bfloat16).requires_grad_(True)
        bf16_V_ref = V_ref.detach().to(torch.bfloat16).requires_grad_(True)
        bf16_lr0_ref = lr0_ref.detach().to(torch.float32).requires_grad_(True)
        bf16_lr1_ref = lr1_ref.detach().to(torch.float32).requires_grad_(True)
        bf16_lr2_ref = lr2_ref.detach().to(torch.float32).requires_grad_(True)
        DW0_DW2_ref_bf16, DW1_ref_bf16 = reference_grouped_swiglu_bwd_with_lr_example(
            bf16_W0_W2_ref,
            bf16_W1_ref,
            bf16_K_input_ref,
            bf16_V_ref,
            bf16_lr0_ref,
            bf16_lr1_ref,
            bf16_lr2_ref,
            group_sizes,
        )
    loss_ref_bf16 = DW0_DW2_ref_bf16.sum() + DW1_ref_bf16.sum() * 2
    loss_ref_bf16.backward()

    print("---- reference_bf16_vs_reference_fp32 ---- \n")
    report_error(
        DW0_DW2_ref_bf16, DW0_DW2_ref, "reference_bf16_vs_reference_fp32_DW0_DW2_bf16"
    )
    report_error(DW1_ref_bf16, DW1_ref, "reference_bf16_vs_reference_fp32_DW1_bf16")
    report_error(
        W0_W2_ref.grad,
        bf16_W0_W2_ref.grad,
        "reference_bf16_vs_reference_fp32_W0_W2_grad_bf16",
    )
    report_error(
        W1_ref.grad, bf16_W1_ref.grad, "reference_bf16_vs_reference_fp32_W1_grad_bf16"
    )
    report_error(
        K_input_ref.grad,
        bf16_K_input_ref.grad,
        "reference_bf16_vs_reference_fp32_K_grad_bf16",
    )
    report_error(
        V_ref.grad, bf16_V_ref.grad, "reference_bf16_vs_reference_fp32_V_grad_bf16"
    )
    report_error(
        lr0_ref.grad,
        bf16_lr0_ref.grad,
        "reference_bf16_vs_reference_fp32_lr0_grad_bf16",
    )
    report_error(
        lr1_ref.grad,
        bf16_lr1_ref.grad,
        "reference_bf16_vs_reference_fp32_lr1_grad_bf16",
    )
    report_error(
        lr2_ref.grad,
        bf16_lr2_ref.grad,
        "reference_bf16_vs_reference_fp32_lr2_grad_bf16",
    )


def run_benchmark(name_2_fn, inputs, repeats=10, warmup=10, FLOPS=None):

    try:
        from benchmark import (
            benchmark_forward,
            benchmark_combined,
            benchmark_all,
            benchmark_memory_forward,
            benchmark_memory_combined,
        )
    except ImportError:
        from benchmark import (
            benchmark_forward,
            benchmark_combined,
            benchmark_all,
            benchmark_memory_forward,
            benchmark_memory_combined,
        )

    if FLOPS is None:
        FLOPS = 0

    ret_dict = {}
    for name, fn in name_2_fn.items():
        m_fwd, m_all = benchmark_all(
            fn,
            *inputs,
            repeats=repeats,
            amp=True,
            amp_dtype=torch.bfloat16,
            verbose=False,
        )

        fwd_s = m_fwd[1].median
        all_s = m_all[1].median
        fwd_tflops = (FLOPS * 1e-12) / fwd_s
        all_tflops = (FLOPS * 3 * 1e-12) / all_s
        print(f"{name} - Forward: {fwd_s * 1e3:.2f} ms, {fwd_tflops:.1f} TFLOPS")
        print(f"{name} - Combined: {all_s * 1e3:.2f} ms, {all_tflops:.1f} TFLOPS")

        #### memory
        memory_fwd = benchmark_memory_forward(
            fn,
            *inputs,
            amp=True,
            amp_dtype=torch.bfloat16,
        )
        memory_all = benchmark_memory_combined(
            fn,
            *inputs,
            amp=True,
            amp_dtype=torch.bfloat16,
        )
        print(f"{name} - Forward memory: {memory_fwd:.2f} GB")
        print(f"{name} - Combined memory: {memory_all:.2f} GB")

        ret = [fwd_tflops, all_tflops, memory_fwd, memory_all]
        ret_dict[name] = ret
    return ret_dict


def benchmark_fwdbwd():
    # profile_backward_function_test()
    # exit()
    # for D in [256, 384, 512, 1024]:
    for D in [768, 1024]:
        for H in [2.0]:
            H = int(D * H)
            print(f"Test Correctness: Running D={D}, H={H}")
            # test_fwd_bwd_match(M=H, K=D)
            print("--------------------------------")
    # test_fwd_bwd_match()
    # return

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="lact_fw_grad_benchmark")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    name_2_fn = {
        "triton_lact_fw_grad_compile": torch.compile(
            grouped_lact_swiglu_ffn_fast_weight_grads
        ),
        # "triton_lact_fw_grad": fused_lact_swiglu_ffn_fast_weight_grads,
        "torch_compile": torch.compile(reference_grouped_swiglu_bwd_with_lr_example),
    }
    B_list = [4, 8, 16]
    B_list = [2]
    E = 4
    # B_list = [2]
    L_list = [2048, 4096, 8192, 16384]
    D_list = [256, 384, 512, 1024]
    D_list = [512, 1024]
    D_list = [256, 384, 512, 1024]
    # D_list = [384]
    D_list = [384]
    # D_list = [1024]
    H_ratio_list = [2.0]
    H_ratio_list = [1.0]

    for B in B_list:
        for D in D_list:
            for H_ratio in H_ratio_list:
                H = int(D * H_ratio)

                title = f"B={B}, D={D}, H={H}"
                single_name_to_stats_dict_list = []
                for L in L_list:
                    print(f"Running B={B}, L={L}, D={D}, H={H}")
                    inputs = make_inputs(B, E, D, H, L)
                    FLOPS = 12 * L * D * H * B
                    single_name_to_stats_dict = run_benchmark(
                        name_2_fn,
                        inputs,
                        repeats=10,
                        warmup=10,
                        FLOPS=FLOPS,
                    )
                    single_name_to_stats_dict_list.append(single_name_to_stats_dict)
                    print("--------------------------------")

                import matplotlib.pyplot as plt

                # four subplots, one for fwd_tflops, one for all_tflops, one for memory_fwd, one for memory_all
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))
                plt.title(title)
                for name in name_2_fn.keys():

                    a = [x[name][0] for x in single_name_to_stats_dict_list]

                    ax1.plot(
                        L_list,
                        [x[name][0] for x in single_name_to_stats_dict_list],
                        label=name,
                    )
                    ax2.plot(
                        L_list,
                        [x[name][1] for x in single_name_to_stats_dict_list],
                        label=name,
                    )
                    ax3.plot(
                        L_list,
                        [x[name][2] for x in single_name_to_stats_dict_list],
                        label=name,
                    )
                    ax4.plot(
                        L_list,
                        [x[name][3] for x in single_name_to_stats_dict_list],
                        label=name,
                    )
                ax1.legend()
                ax2.legend()
                ax3.legend()
                ax4.legend()
                plt.savefig(os.path.join(args.save_dir, f"B={B}_D={D}_H={H}.png"))
                plt.close()


if __name__ == "__main__":
    # benchmark_fwdbwd()

    # test_fwd_bwd_match(B=4, E=8, M=512, K=512, N=4096)
    # test_fwd_bwd_match(B=8, E=4, M=512, K=512, N=4096)

    benchmark_fwdbwd()
