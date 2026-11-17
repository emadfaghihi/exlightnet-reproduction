"""Per-output-channel symmetric W8 storage; inference remains floating point."""
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import ExLightNet


def quantize_per_channel(weight):
    w = weight.detach().cpu().float().numpy()
    axes = tuple(range(1, w.ndim))
    scales = np.maximum(np.max(np.abs(w), axis=axes) / 127., np.float32(1e-12)).astype(np.float32)
    broad = scales.reshape((-1,) + (1,) * (w.ndim - 1))
    return np.clip(np.rint(w / broad), -127, 127).astype(np.int8), scales


def int8_size_breakdown(path):
    """Raw array payloads and actual NPZ members/container; no estimated sizes."""
    path = Path(path)
    raw = {"weights": 0, "scales": 0, "rest": 0, "metadata": 0}
    compressed = {key: 0 for key in raw}
    def component(key):
        return "weights" if key.startswith("q::") else "scales" if key.startswith("s::") else "rest" if key.startswith("f::") else "metadata"
    with np.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            raw[component(key)] += arrays[key].nbytes
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            compressed[component(member.filename.removesuffix(".npy"))] += member.compress_size
    return {"weights_only_int8_bytes": raw["weights"], "scales_fp32_bytes": raw["scales"], "rest_fp32_bytes": raw["rest"],
            "metadata_array_bytes": raw["metadata"], "total_array_payload_bytes": sum(raw.values()),
            "compressed_members_bytes": compressed, "zip_container_bytes": path.stat().st_size - sum(compressed.values()),
            "full_artifact_bytes": path.stat().st_size,
            "definition": "weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container"}


def export_int8(model, path, metadata=None):
    arrays = {}
    quantized = set()
    errors = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            key = name + ".weight"
            q, scale = quantize_per_channel(module.weight)
            arrays["q::" + key], arrays["s::" + key] = q, scale
            restored = q * scale.reshape((-1,) + (1,) * (q.ndim - 1))
            errors.append(float(np.max(np.abs(module.weight.detach().numpy() - restored))))
            quantized.add(key)
    for key, value in model.state_dict().items():
        if key not in quantized:
            arrays["f::" + key] = value.detach().cpu().numpy().astype(np.float32)
    meta = {"format_version": 1, "num_classes": model.num_classes, "use_se": model.use_se,
            "standard": model.standard, "compute": "FP32 with dequantized W8 weights",
            "bn_eps": 1e-5, **(metadata or {})}
    arrays["__meta__"] = np.array(json.dumps(meta))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"max_abs_weight_error": max(errors), "int8_bytes": path.stat().st_size,
            "quantized_weight_tensors": len(quantized), "compute": meta["compute"], "size_breakdown": int8_size_breakdown(path)}


def load_int8(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__meta__"]))
        model = ExLightNet(meta["num_classes"], use_se=meta["use_se"], standard=meta["standard"])
        state = {}
        for key, original in model.state_dict().items():
            if "q::" + key in archive:
                q, s = archive["q::" + key], archive["s::" + key]
                value = q.astype(np.float32) * s.reshape((-1,) + (1,) * (q.ndim - 1))
            else:
                value = archive["f::" + key]
            state[key] = torch.as_tensor(value.copy(), dtype=original.dtype)
    model.load_state_dict(state, strict=True)
    return model.eval()
