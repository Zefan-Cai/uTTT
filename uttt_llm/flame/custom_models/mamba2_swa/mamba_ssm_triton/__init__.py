"""Pure-Triton SSD kernels copied from state-spaces/mamba v2.2.4 (mamba_ssm/ops/triton/*, Apache-2.0,
see LICENSE here), imports made relative. Used by layer_mamba2_swa.py only when mamba_ssm itself
cannot be imported (its package __init__ needs the compiled selective_scan_cuda)."""
