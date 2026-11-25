"""Device policy for supplementary studies; canonical CPU training is untouched."""
import os

# This must be set before any CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def configure_device(requested="cpu"):
    import torch
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    selected = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if selected == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return selected


def environment_identity(device, libraries):
    if device not in ("cpu", "cuda"):
        raise ValueError("Invalid actual device")
    keys = ("numpy", "torch", "scikit-learn", "xgboost")
    if device == "cuda":
        keys += ("cuda", "cudnn", "gpu_name")
    return {"device": device, **{key: libraries.get(key) for key in keys}}


def effective_settings():
    import torch
    return {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}
