"""Check ExLightNet widths with synthetic CPU inputs and count structural costs."""

import json

import torch
from torch import nn

from exlightnet.model import ExLightNet, model_gate, parameter_count


def check_length(length, classes):
    model = ExLightNet(classes).cpu().eval()
    weighted_layers = {
        name: layer for name, layer in model.named_modules() if isinstance(layer, (nn.Conv1d, nn.Linear))
    }
    flops = 0

    def count(module, _inputs, output):
        nonlocal flops
        positions = output.shape[-1] if isinstance(module, nn.Conv1d) else 1
        flops += 2 * positions * module.weight.numel()

    hooks = [layer.register_forward_hook(count) for layer in weighted_layers.values()]
    try:
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, length))
    finally:
        for hook in hooks:
            hook.remove()
    if output.shape != (1, classes) or not torch.isfinite(output).all():
        raise RuntimeError(f"Invalid logits for length={length}, classes={classes}")

    weight_names = {name + ".weight" for name in weighted_layers}
    weights = sum(layer.weight.numel() for layer in weighted_layers.values())
    scales = sum(layer.weight.shape[0] for layer in weighted_layers.values())
    rest = sum(tensor.numel() for name, tensor in model.state_dict().items() if name not in weight_names)
    return {
        "length": length,
        "classes": classes,
        "output_shape": list(output.shape),
        "trainable_parameters": parameter_count(model),
        "conv_and_linear_flops": flops,
        "w8_weights_only_bytes": weights,
        "w8_fp32_scales_bytes": 4 * scales,
        "w8_fp32_rest_bytes": 4 * rest,
        "w8_model_array_bytes": weights + 4 * (scales + rest),
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    gate = model_gate()["actual"]
    expected = {12: gate["multiclass"], 2: gate["binary"]}
    rows = [check_length(length, classes) for length in (8, 40, 43, 93) for classes in (12, 2)]
    if any(row["trainable_parameters"] != expected[row["classes"]] for row in rows):
        raise RuntimeError("M1 parameter count changed across feature-set widths")
    print(json.dumps({"scope": "synthetic forward and analytical counts", "device": "cpu",
                      "torch": str(torch.__version__), "models": rows}, indent=2))


if __name__ == "__main__":
    main()
