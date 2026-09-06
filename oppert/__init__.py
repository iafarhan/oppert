# Register Intel XPU kernels on Dawn HPC (torch 2.3 + IPEX 2.3.110).
# Must happen before ANY XPU tensor is created. No-op on CPU/CUDA-only hosts.
try:
    import intel_extension_for_pytorch as _ipex  # noqa: F401
except ImportError:
    pass
