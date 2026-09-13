import torch


try:
    from triton_grouped_swiglu_bwd_kernels import (
        swiglu_backward_three_bmm_grouped_triton,
    )
    from triton_grouped_swiglu_kernels import fused_two_mm_swiglu_triton
    from triton_grouped_gemm import (
        grouped_gemm_transpose_triton,
        grouped_gemm_bwd_triton,
    )
    from triton_lact_bwd_grouped_two_mm_grad_fw import (
        grouped_two_gemm_fw_bwd_triton,
    )
except ImportError:
    from .triton_grouped_swiglu_bwd_kernels import (
        swiglu_backward_three_bmm_grouped_triton,
    )
    from .triton_grouped_swiglu_kernels import fused_two_mm_swiglu_triton
    from .triton_grouped_gemm import (
        grouped_gemm_transpose_triton,
        grouped_gemm_bwd_triton,
    )
    from .triton_lact_bwd_grouped_two_mm_grad_fw import (
        grouped_two_gemm_fw_bwd_triton,
    )
from torch.autograd.function import once_differentiable


class FusedSwiGLUFFNFwd(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(
        cast_inputs=torch.bfloat16, device_type="cuda"
    )  # let autocast cast once
    def forward(ctx, W0_W2, W1, X, group_sizes):
        """
        Args:
            W0_W2: [B, E, 2 * Hidden, D]
            W1:     [B, E, K, M] or [B, E, D, Hidden]
            X:      [M, N, K] or [B, num_Tokens, D]
            group_sizes: [B, E]
        Outs:
            Hidden: [B, N, K] or [B, num_tokens, Hidden]

        W1 @ [SiLU(W0 @ X.T) * (W2 @ X.T)]
        """

        # [B, Hidden, num_tokens]
        #### Without this triton kernel, we will materize Y2, SiLU(Y0) * Y2.
        #### 2 + 1 read and write.
        # Here we only have one write.
        # [B, Hidden, num_tokens]
        Hidden = fused_two_mm_swiglu_triton(W0_W2, X, group_sizes)

        # -> [B, num_tokens, D]
        # output = torch.bmm(W1, Hidden).transpose(1, 2)
        output = grouped_gemm_transpose_triton(W1, Hidden, group_sizes)
        ctx.save_for_backward(W0_W2, W1, X, group_sizes)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: [B, num_tokens, D]
        Outs:
            grad_W0_W2: [B, E, 2 * Hidden, D]
            grad_W1: [B, E, K, M] or [B, E, D, Hidden]
            grad_X: [B, D, num_tokens]
        """
        W0_W2, W1, X, group_sizes = ctx.saved_tensors
        # [B, 2 * Hidden, num_tokens]
        DY0_DY2, Hidden = swiglu_backward_three_bmm_grouped_triton(
            W0_W2, W1, X, grad_out.contiguous(), group_sizes
        )

        # grad_W0_W2 = grouped_gemm_bwd_triton(DY0_DY2, X, group_sizes)
        # grad_W1 = grouped_gemm_bwd_triton(
        #     grad_out.transpose(1, 2).contiguous(),
        #     Hidden.transpose(1, 2).contiguous(),
        #     group_sizes,
        # )

        grad_W0_W2, grad_W1 = grouped_two_gemm_fw_bwd_triton(
            DY0_DY2, X, Hidden, grad_out.contiguous(), group_sizes
        )

        grad_X = grouped_gemm_transpose_triton(
            W0_W2.transpose(2, 3).contiguous(), DY0_DY2.contiguous(), group_sizes
        )

        # # [B, D, num_tokens] @ [B, num_tokens, Hidden] -> [B, D, Hidden]
        # grad_W1 = torch.bmm(grad_out.transpose(1, 2), Hidden.transpose(1, 2))

        # # [B, 2 * Hidden, num_tokens] @ [B, num_tokens, D] -> [B, 2 * Hidden, D]
        # grad_W0_W2 = torch.bmm(DY0_DY2, X)

        # # [B, 2 * Hidden, num_tokens].T @ [B, 2 * Hidden, D] -> [B, 2 * Hidden, D]
        # grad_X = torch.bmm(DY0_DY2.transpose(1, 2), W0_W2)

        return (grad_W0_W2, grad_W1, grad_X, None)


def grouped_swiglu_ffn_fwd(W0_W2, W1, X, group_sizes):
    return FusedSwiGLUFFNFwd.apply(W0_W2, W1, X, group_sizes)


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


def reference_grouped_swiglu_ffn_fwd_example(
    w0_w2: torch.Tensor,
    w1: torch.Tensor,
    x: torch.Tensor,
    group_sizes: torch.Tensor,
):
    B, E, Hidden_times_2, D = w0_w2.shape
    Hidden = Hidden_times_2 // 2

    all_batch_o_list = []
    for b_index in range(B):
        start_index = 0
        single_batch_o_list = []
        for e_index in range(E):
            group_size = group_sizes[b_index, e_index]

            x_i = x[b_index, start_index : start_index + group_size, :]

            start_index += group_size

            w0_w2_i = w0_w2[b_index, e_index, :, :]

            w0_i = w0_w2_i[:Hidden, :]
            w2_i = w0_w2_i[Hidden:, :]
            w1_i = w1[b_index, e_index, :, :]

            y1 = w0_i @ x_i.T
            y2 = w2_i @ x_i.T

            y = torch.nn.functional.silu(y1) * y2

            o = w1_i @ y
            o = o.transpose(-1, -2)
            single_batch_o_list.append(o)

        single_batch_o = torch.cat(single_batch_o_list, dim=0)
        all_batch_o_list.append(single_batch_o)

    # [b, D, N]
    all_o = torch.stack(all_batch_o_list, dim=0)
    return all_o


def benchmark_ffn_fwd():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=str, default="triton_swiglu_ffn_fwd_benchmark"
    )

    def make_inputs_ffn(B, E, D, H, N, require_grad=False):
        w0_w2 = torch.randn(
            B,
            E,
            2 * H,
            D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=require_grad,
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
        mean_size = N // E
        group_sizes = torch.randint(
            1, mean_size, (B, E), device="cuda", dtype=torch.int32, requires_grad=False
        )
        sum_group_sizes = group_sizes.sum(dim=1)
        extra_size = N - sum_group_sizes
        group_sizes[:, -1] += extra_size

        return w0_w2, w1, x, group_sizes

    B, E, D, H, N = 4, 4, 1024, 1024, 4096
    inputs = make_inputs_ffn(B, E, D, H, N)

    # reference_out = reference_swiglu_ffn_fwd(*inputs)

    fp32_inputs = [_.to(torch.float32) for _ in inputs]
    fp32_inputs[-1] = fp32_inputs[-1].to(torch.int32)
    reference_out = reference_grouped_swiglu_ffn_fwd_example(*fp32_inputs)
    triton_out = grouped_swiglu_ffn_fwd(*inputs)

    from benchmark import report_error

    report_error(reference_out, triton_out, "fused_swiglu_ffn_fwd")

    # test backward as well
    inputs_that_require_grad = make_inputs_ffn(B, E, D, H, N, require_grad=True)

    triton_out_that_requires_grad = grouped_swiglu_ffn_fwd(*inputs_that_require_grad)
    loss = triton_out_that_requires_grad.sum()
    loss.backward()

    triton_w0_w2_grad = inputs_that_require_grad[0].grad.clone().detach()
    triton_w1_grad = inputs_that_require_grad[1].grad.clone().detach()
    triton_x_grad = inputs_that_require_grad[2].grad.clone().detach()

    fp32_inputs_that_require_grad = [
        _.to(torch.float32).detach().clone().requires_grad_(True)
        for _ in inputs_that_require_grad
    ]
    fp32_inputs_that_require_grad[-1] = (
        fp32_inputs_that_require_grad[-1].to(torch.int32).detach().clone()
    )

    reference_output = reference_grouped_swiglu_ffn_fwd_example(
        *fp32_inputs_that_require_grad
    )
    loss = reference_output.sum()
    loss.backward()
    reference_w0_w2_grad = fp32_inputs_that_require_grad[0].grad.clone().detach()
    reference_w1_grad = fp32_inputs_that_require_grad[1].grad.clone().detach()
    reference_x_grad = fp32_inputs_that_require_grad[2].grad.clone().detach()
    report_error(reference_w0_w2_grad, triton_w0_w2_grad, "w0_w2_grad")
    report_error(reference_w1_grad, triton_w1_grad, "w1_grad")
    report_error(reference_x_grad, triton_x_grad, "x_grad")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    name_2_fn = {
        "triton_swiglu_ffn_compile": torch.compile(
            grouped_swiglu_ffn_fwd  #  , mode="reduce-overhead"
        ),
        "triton_swiglu_ffn": grouped_swiglu_ffn_fwd,
        "torch_compile": torch.compile(reference_grouped_swiglu_ffn_fwd_example),
    }
    B_list = [4, 8, 16]
    B_list = [1]
    # B_list = [2]
    L_list = [2048, 4096, 8192, 16384]
    D_list = [256, 384, 512, 1024]
    D_list = [512, 1024]
    D_list = [256, 384, 512, 1024]
    # D_list = [384]
    D_list = [1024]
    H_ratio_list = [2.0]
    # H_ratio_list = [2.0]

    for B in B_list:
        for D in D_list:
            for H_ratio in H_ratio_list:
                H = int(D * H_ratio)
                title = f"B={B}, D={D}, H={H}"

                single_name_to_stats_dict_list = []
                for L in L_list:
                    print(f"Running B={B}, L={L}, D={D}, H={H}")
                    inputs = make_inputs_ffn(B, E, D, H, L, require_grad=True)
                    FLOPS = 6 * L * D * H * B
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
    benchmark_ffn_fwd()
