"""Quoted reference values ONLY. Never used by training, selection or scoring."""

PAPER_REPORTED = {
    "source": "supplied paper/exlightnet.pdf, Tables 6, 7, 11, 12 and Section 4.2; quoted, not reproduced",
    "binary": {
        "exlightnet": {"accuracy": .9949, "macro_f1": .9860, "auc": .9985},
        "mlp": {"accuracy": .9926, "macro_f1": .9800},
        "bilstm": {"accuracy": .9744, "macro_f1": .9354},
        "std_cnn": {"accuracy": .9934, "macro_f1": .9820},
        "random_forest": {"accuracy": .9976, "macro_f1": .9934},
        "xgboost": {"accuracy": .9984, "macro_f1": .9957},
        "exlightnet_w8": {"accuracy": .9949, "macro_f1": .9860},
    },
    "multiclass": {
        "exlightnet": {"accuracy": .9770, "macro_f1": .8621},
        "mlp": {"accuracy": .9705, "macro_f1": .7933},
        "bilstm": {"accuracy": .9663, "macro_f1": .6380},
        "std_cnn": {"accuracy": .9794, "macro_f1": .8529},
        "random_forest": {"accuracy": .9797, "macro_f1": .8600},
        "xgboost": {"accuracy": .9821, "macro_f1": .8632},
        "exlightnet_w8": {"accuracy": .9770, "macro_f1": .8555},
        "no_se": {"macro_f1": .7963}, "standard_conv": {"macro_f1": .8259, "params": 59484},
        "balanced": {"macro_f1": .7807}, "no_sampler": {"macro_f1": .8697}, "top20": {"macro_f1": .6048},
    },
    "exlightnet_multiclass_three_seed": {"mean_macro_f1": .835, "std_macro_f1": .023, "headline_best_seed_macro_f1": .8621},
    "quoted_multiclass_bytes_approx": {"fp32_kb": 162, "w8_kb": 37},
}
