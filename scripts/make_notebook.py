#!/usr/bin/env python3
"""Build the notebook using nbformat; all experiment logic is imported."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def section(title, prose, code):
    cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{prose}"))
    cells.append(nbf.v4.new_code_cell(code))


cells.append(nbf.v4.new_markdown_cell("# ExLightNet: an honest CPU reproduction\n\nFast mode trains seed 42 for one epoch per final deep model and alpha candidate, with reduced SHAP. Full mode only loads cached results from the parallel pipeline (`python scripts/run_parallel.py`), including its final isolated benchmark pass. Paper results are quotations, never generated outputs."))
section("0. Environment", "Set RUN_MODE here; restart and run all. The first fast run performs live training; later runs reuse frozen evaluations.", '''import os, sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / "src/exlightnet").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exlightnet-mpl")
RUN_MODE = os.environ.get("EXLIGHTNET_RUN_MODE", "fast")
from IPython.display import display
import pandas as pd
from exlightnet.bench import environment_info
from exlightnet.io import read_json
from exlightnet.preprocess import FlowPreprocessor
from exlightnet import notebook as presentation
display(environment_info())
print("This notebook executes on the CPU. The displayed library versions and processor describe this environment.")''')
section("1. Data and D1/D2", "The genuine corpus is required in fast mode; no synthetic replacement is used.", '''if RUN_MODE == "fast":
    from exlightnet.run_all import prepare_data
    _, _, _, data_context = prepare_data(ROOT)
gates = read_json(ROOT / "results/data_gates.json")
display(pd.DataFrame(gates))
print("D1/D2 status comes from the actual row counts, split sizes and class supports shown above. A failed hard gate stops dependent training.")''')
section("2. Train-only preprocessing and MI", "Feature identity follows computed descending MI order, with the vocabulary and all statistics frozen from train.", '''pp = FlowPreprocessor.load(ROOT / "artifacts")
display(pp.compare_table4())
display(pd.DataFrame(pp.ranking_[:40]))
print(f"The computed selection shares {pp.compare_table4()['overlap_count']} of 40 paper features. The displayed rank correlation measures ordering agreement and was not forced.")''')
section("3. Architecture and M1", "Conv lengths are 40 → 38 → 36 → 34; convolutions have no padding or biases.", '''from exlightnet.model import ExLightNet, model_gate
display(ExLightNet(12))
display(model_gate())
print("M1 checks the runtime parameter counts for both heads and the no-SE variant. The per-stage counts come from the constructed graph.")''')
section("4. Live training in fast mode", "Alpha selection and early stopping use validation only. Full mode does not train in this notebook.", '''manifest = presentation.experiment(ROOT, RUN_MODE)
display(manifest)
display(read_json(ROOT / "results" / RUN_MODE / "alpha_selection.json"))
print("The recorded seed list and epoch limits define the scope of these results. No best seed or test-selected epoch is reported.")''')
section("5. Binary results", "Attack is class 1; decisions use two-logit argmax at a fixed threshold.", 'presentation.show_task(ROOT, RUN_MODE, "binary")')
section("6. Twelve-class results", "Per-class supports and row-normalised confusion expose failures hidden by accuracy.", 'presentation.show_task(ROOT, RUN_MODE, "multiclass")')
section("7. Baselines and seed variation", "All headline models share the fixed partition and features; full mode uses five seeds.", 'presentation.show_baselines(ROOT, RUN_MODE)')
section("8. Ablations", "The full model is reused as the reference rather than retrained for a second test pass.", 'presentation.show_ablations(ROOT, RUN_MODE)')
section("9. Weight-only quantisation", "Per-channel INT8 storage is evaluated using dequantised floating-point weights.", 'presentation.show_quantization(ROOT, RUN_MODE)')
section("10. Measured latency", "Full-mode latency comes from the final benchmark-only pass: one process, one thread, batch one, with CPU-idle evidence recorded before each model.", 'presentation.show_latency(ROOT, RUN_MODE)')
section("11. SHAP and faithfulness", "The primary attribution target is the FP32 model's logits; deletion measures predicted-class probabilities.", 'presentation.show_shap(ROOT, RUN_MODE)')
section("12. Ours versus paper (quoted)", "Reference values come only from PAPER_REPORTED; one-seed paper headlines are not multi-seed targets.", 'presentation.show_comparison(ROOT, RUN_MODE)')
section("13. Deviations and limitations", "REPORT.md contains exact assumptions, gate evidence, hardware, runtime and stopped branches.", '''display(pd.DataFrame(read_json(ROOT / "results" / RUN_MODE / "failures.json")))
print("Fast mode cannot establish five-seed means and standard deviations; full-mode results remain separate. The random flow split may share attack-run artifacts across partitions, and four/five-flow rare-class supports limit inference.")
print("Baseline LSTM depth and CNN pooling are declared implementation choices. No Pi measurement, true INT8 arithmetic or campaign-disjoint generalisation is claimed.")''')
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3 (ExLightNet)", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.11"}}
path = ROOT / "notebooks/ExLightNet_reproduction.ipynb"
nbf.write(nb, path)
print(path)
