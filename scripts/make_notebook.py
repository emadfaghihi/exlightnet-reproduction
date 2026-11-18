#!/usr/bin/env python3
"""Generate the cached analysis notebook; all scientific logic stays in package imports."""
import argparse
from pathlib import Path

import nbformat as nbf


def make_notebook(root, output=None):
    root = Path(root).resolve()
    cells = []

    def section(title, prose, code):
        cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{prose}"))
        cells.append(nbf.v4.new_code_cell(code))

    cells.append(nbf.v4.new_markdown_cell("# ExLightNet: cached CPU reproduction and statistics\n\nThe default scope is the completed full CPU reproduction. This notebook reads persisted results, predictions-derived tables and benchmark evidence. It never launches training, test inference or a benchmark. Historical reduced-run evidence appears in its own Fast demonstration section. Paper values are labelled quotations."))
    section("0. Environment", "Restart and run all to inspect the cached analysis. RUN_MODE defaults to full.", '''import os, sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / "src/exlightnet").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exlightnet-mpl")
RUN_MODE = os.environ.get("EXLIGHTNET_RUN_MODE", "full")
from IPython.display import display
import pandas as pd
from exlightnet.bench import environment_info
from exlightnet.io import read_json
from exlightnet.preprocess import FlowPreprocessor
from exlightnet import notebook as presentation
display(environment_info())
print("This analysis runs on the CPU. The displayed library versions and processor describe the current environment.")''')
    section("1. Data and D1/D2", "Gate evidence comes from the genuine corpus and its persisted class-stratified partition.", '''gates = read_json(ROOT / "results/data_gates.json")
display(pd.DataFrame(gates))
print("D1/D2 status is read from saved row counts, split sizes and class supports. This notebook does not rewrite gates or rebuild data caches.")''')
    section("2. Train-only preprocessing and MI", "Feature identity follows descending MI order; category vocabularies and all fitted statistics remain frozen from training.", '''pp = FlowPreprocessor.load(ROOT / "artifacts")
display(pp.compare_table4())
display(pd.DataFrame(pp.ranking_[:40]))
print(f"The persisted selection shares {pp.compare_table4()['overlap_count']} of 40 paper features. Ranking agreement was measured, not forced.")''')
    section("3. Architecture and M1", "Conv lengths are 40 → 38 → 36 → 34; convolutions have no padding or biases.", '''from exlightnet.model import ExLightNet, model_gate
display(ExLightNet(12))
display(model_gate())
print("M1 checks constructed parameter counts for both heads and the no-SE variant; no model is trained or evaluated here.")''')
    section("4. Cached campaign and effective status", "Validation-only model selection is preserved. Current status is derived alongside the unchanged historical pipeline flag.", '''manifest = presentation.experiment(ROOT, RUN_MODE)
display(manifest)
display(read_json(ROOT / "results" / RUN_MODE / "alpha_selection.json"))
presentation.show_execution_status(ROOT, RUN_MODE)
print("Archived CPU-idle failures remain evidence. A later successful benchmark recovery is reported separately; raw status files are never edited.")''')
    section("5. Binary results", "Attack is class 1; decisions use two-logit argmax at a fixed threshold. All requested seeds and seed 42 are shown.", 'presentation.show_task(ROOT, RUN_MODE, "binary")')
    section("6. Twelve-class results", "Per-class support accompanies every class metric. Tiny supports make macro-F1 comparisons noisy.", 'presentation.show_task(ROOT, RUN_MODE, "multiclass")')
    section("7. Baselines and seed variation", "n=5, indicative only. Models share the fixed split and train-fitted features; instability and zero variance limit interpretation.", 'presentation.show_baselines(ROOT, RUN_MODE)')
    section("8. Paired ablations", "n=5, indicative only. Cached paired tests replace SD-overlap comparisons. Differences use matched seeds and task-wide Holm correction.", 'presentation.show_ablations(ROOT, RUN_MODE)')
    section("9. Generated CPU statistics", "n=5, indicative only. Paired tests, flow bootstrap uncertainty, per-class supports, post-hoc sensitivity and descriptive dynamics are loaded from generated CSVs.", 'presentation.show_statistics(ROOT, RUN_MODE)')
    section("10. Weight-only quantisation", "Per-channel INT8 storage is evaluated using dequantised floating-point weights.", 'presentation.show_quantization(ROOT, RUN_MODE)')
    section("11. Measured latency", "Timing comes from persisted isolated CPU benchmark evidence. Eager batch-one timing includes framework dispatch; the dispatch fraction is NOT MEASURED.", 'presentation.show_latency(ROOT, RUN_MODE)')
    section("12. SHAP and faithfulness", "Attributions explain FP32 logits; deletion measures predicted-class probabilities. The actual method, fallback and residuals remain visible.", 'presentation.show_shap(ROOT, RUN_MODE)')
    section("13. Ours versus paper (quoted)", "n=5, indicative only. Paper headlines are quotations, never optimization targets; rare-class supports limit macro-F1 comparisons.", 'presentation.show_comparison(ROOT, RUN_MODE)')
    section("14. Limitations", "Fixed split, small seed count, rare-class supports, unstable baselines and hardware-specific eager timing constrain conclusions. Stopping/cosine causality remains a Stage B hypothesis.", '''presentation.show_limitations(ROOT)
display(pd.DataFrame(read_json(ROOT / "results" / RUN_MODE / "failures.json")))''')
    section("Fast demonstration (historical)", "This optional reduced-run evidence is separate from the full reproduction. It is read from its existing cache, with no live training.", 'presentation.show_fast_demonstration(ROOT)')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (ExLightNet)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "exlightnet": {"default_run_mode": "full", "execution": "cached analysis only"}}
    from exlightnet.publication import validate_notebook_content
    validate_notebook_content(nb, "full")
    path = Path(output) if output is not None else root / "notebooks/ExLightNet_reproduction.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="generated notebook path; default ROOT/notebooks/ExLightNet_reproduction.ipynb")
    args = parser.parse_args()
    print(make_notebook(args.root, args.output))


if __name__ == "__main__":
    main()
