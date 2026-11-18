"""Presentation uses persisted paired statistics and preserves historical evidence."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest

from exlightnet.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def publication():
    return importlib.import_module("exlightnet.publication")


def stale_phrase():
    return "fast" + " mode"


def obsolete_flag():
    return "exceeds" + "_full_seed_std"


def test_mode_validation_rejects_stale_captions_and_rendered_outputs():
    module = publication()
    with pytest.raises(ValueError, match="demonstration"):
        module.validate_publication_text("## Full CPU results\n" + stale_phrase())
    module.validate_publication_text("## Fast demonstration (historical)\n" + stale_phrase() + "\n### Details\n" + stale_phrase())
    with pytest.raises(ValueError):
        module.validate_publication_text("## Fast demonstration\n" + stale_phrase() + "\n## Full CPU results\n" + stale_phrase())
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("## Full CPU results"),
        nbformat.v4.new_code_cell("print('cached results')", outputs=[nbformat.v4.new_output("stream",name="stdout",text=stale_phrase())])])
    with pytest.raises(ValueError, match="demonstration"):
        module.validate_notebook_content(nb, "full")
    nb.cells[0].source = "## Fast demonstration (historical)"
    module.validate_notebook_content(nb, "full")


def test_active_scope_scanner_excludes_history_but_includes_new_outputs(tmp_path):
    paths = ["src/exlightnet/new.py", "scripts/new.py", "tests/new.py", "REPORT.md", "notebooks/new.ipynb",
             "artifacts/stats_cpu_v1/new.csv", "results/full/ablations.json", "archive/old.json", "docs/STATS_GPU_BRIEF.txt", "AGENTS.md"]
    for relative in paths:
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(obsolete_flag())
    found = publication().active_scope_legacy_flags(tmp_path)
    assert {item["path"] for item in found} == set(paths[:6])


def test_active_scope_also_covers_supplementary_outputs(tmp_path):
    for name in ('artifacts/stats_gpu_v2/new.csv', 'artifacts/stats_gpu_v2_smoke/new.json', 'artifacts/stats_gpu_v2_smoke_verified/new.json', 'REPORT_stats_v2.md'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(obsolete_flag())
    assert len(publication().active_scope_legacy_flags(tmp_path)) == 4


def status_fixture(root):
    directory = root / "results/full"
    write_json(directory / "manifest.json", {"full_five_seed_headlines_complete":True,"failures":[],"seeds":[42,43,44,45,46]})
    write_json(directory / "failures.json", [])
    write_json(directory / "parallel_execution.json", {"status":"complete_with_stopped_branches","finished_at":"2026-11-16T15:00:00+00:00","invocations":[{"returncode":0}]})
    failure={"branch":"benchmark/binary/exlightnet/42","error_type":"IdleCPUUnavailable","reason":"original idle gate timeout"}
    write_json(directory / "benchmarks/old/manifest.json", {"campaign":"old","finished_at":"2026-11-16T15:00:00+00:00","failures":[failure],"failed_models":1,"measured_models":0})
    latest={"campaign":"new","finished_at":"2026-11-17T05:39:01+00:00","failures":[],"attempted_models":95,"measured_models":95,"failed_models":0,"skipped_models":0,"gated_out_models":0}
    write_json(directory / "benchmark_manifest.json", latest)
    write_json(directory / "benchmark_recovery.json", {"status":"complete","campaign":"new","returncode":0,"failures":[]})
    return directory


def test_effective_status_explains_archive_and_recovery_without_mutating(tmp_path):
    directory = status_fixture(tmp_path)
    before={path:path.read_bytes() for path in directory.rglob("*.json")}
    status=publication().execution_status(tmp_path)
    assert status["effective_status"] == "complete"
    assert status["raw_pipeline_status"] == "complete_with_stopped_branches"
    assert status["active_failure_count"] == 0
    assert status["archived_idle_failure_count"] == 1
    assert "idle" in status["explanation"].lower() and "recovery" in status["explanation"].lower()
    assert before == {path:path.read_bytes() for path in before}
    write_json(directory / "failures.json", [{"branch":"training","reason":"a new failure","error_type":"RuntimeError"}])
    assert publication().execution_status(tmp_path)["effective_status"] != "complete"


def test_stale_pipeline_cannot_be_declared_recovered_without_source_evidence(tmp_path):
    directory = status_fixture(tmp_path)
    (directory / "benchmark_recovery.json").unlink()
    assert publication().execution_status(tmp_path)["effective_status"] == "requires_review"


def test_future_ablation_comparison_uses_paired_family_and_preserves_rows():
    from exlightnet.run_all import ablation_comparison
    rows=[]
    for seed,ref,value in zip(range(42,47),[.80,.82,.81,.84,.83],[.75,.80,.79,.81,.80]):
        rows.extend([dict(task="multiclass",model="exlightnet",seed=seed,macro_f1=ref),
                     dict(task="multiclass",model="no_se",seed=seed,macro_f1=value)])
    original=copy.deepcopy(rows)
    result=ablation_comparison(rows)
    assert rows == original
    assert len(result) == 1
    assert result[0]["n_pairs"] == 5
    assert result[0]["mean_difference"] == pytest.approx(.03)
    assert result[0]["p_holm"] < .05
    assert result[0]["sample_label"] == "n=5, indicative only"
    assert not any(key.startswith("exceeds" + "_") for key in result[0])


def test_canonical_run_rejects_cuda_before_any_lock_or_output(tmp_path):
    from exlightnet.run_all import run
    with pytest.raises(ValueError, match="CPU|cpu"):
        run(tmp_path / "never_created", device="cuda")
    assert not (tmp_path / "never_created").exists()


def guarded_json(original):
    def guard(path):
        if Path(path).name == "ablations" + ".json":
            pytest.fail("presentation must never read historical ablation JSON")
        return original(path)
    return guard


def test_report_renders_generated_statistics_without_old_ablation_reads(tmp_path, monkeypatch):
    import exlightnet.report as report
    monkeypatch.setattr(report,"read_json",guarded_json(report.read_json))
    output=tmp_path/"REPORT.md"
    report.build_report(ROOT, output=output)
    text=output.read_text()
    publication().validate_publication_text(text)
    assert obsolete_flag() not in text
    for expected in ("n=5, indicative only", "p_holm", "cohens_dz", "PAIRED", "post-hoc sensitivity", "test_support", "f1_sd", "macro_f1_seed42", "cosine_lr_factor", "descriptive", "NOT MEASURED", "## Limitations"):
        assert expected in text
    assert "nmap_fin_scan" in text and "metasploit_brute_force_ssh" in text
    assert "read-only .git" not in text
    assert "mostly" not in text.lower()
    assert "effective_status" in text and "complete_with_stopped_branches" in text


def test_notebook_ablation_and_statistics_display_use_new_tables(monkeypatch, capsys):
    import exlightnet.notebook as presentation
    monkeypatch.setattr(presentation,"read_json",guarded_json(presentation.read_json))
    displayed=[]
    monkeypatch.setattr(presentation,"display",displayed.append)
    presentation.show_ablations(ROOT,"full")
    presentation.show_statistics(ROOT,"full")
    assert any(isinstance(frame,pd.DataFrame) and "p_holm" in frame.columns for frame in displayed)
    assert any(isinstance(frame,pd.DataFrame) and "test_support" in frame.columns for frame in displayed)
    assert "n=5, indicative only" in capsys.readouterr().out
    assert not any(obsolete_flag() in str(frame) for frame in displayed)


def test_notebook_generator_defaults_to_full_without_publishing(tmp_path):
    output=tmp_path/"generated.ipynb"
    published=ROOT/"notebooks/ExLightNet_reproduction.ipynb"
    before=published.read_bytes()
    script_root=tmp_path/"isolated"
    (script_root/"scripts").mkdir(parents=True)
    (script_root/"notebooks").mkdir()
    isolated_script=script_root/"scripts/make_notebook.py"
    isolated_script.write_bytes((ROOT/"scripts/make_notebook.py").read_bytes())
    completed=subprocess.run([sys.executable,str(isolated_script),"--output",str(output)],capture_output=True,text=True)
    assert completed.returncode == 0, completed.stderr
    assert published.read_bytes() == before
    nb=nbformat.read(output,as_version=4)
    assert nb.metadata["exlightnet"]["default_run_mode"] == "full"
    publication().validate_notebook_content(nb,"full")
    assert any("show_statistics" in cell.source for cell in nb.cells)
    assert all("prepare_data(" not in cell.source for cell in nb.cells)
    assert all(not cell.get("outputs") for cell in nb.cells if cell.cell_type == "code")


def test_execution_evidence_uses_new_directory_and_refuses_history_overwrite(tmp_path):
    import importlib.util
    (tmp_path/"scripts").mkdir()
    (tmp_path/"notebooks").mkdir()
    isolated_script=tmp_path/"scripts/execute_notebook.py"
    isolated_script.write_bytes((ROOT/"scripts/execute_notebook.py").read_bytes())
    nbformat.write(nbformat.v4.new_notebook(cells=[]),tmp_path/"notebooks/ExLightNet_reproduction.ipynb")
    spec=importlib.util.spec_from_file_location("execute_nb_test",isolated_script)
    script=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    old=tmp_path/"results/notebook_execution.json"
    write_json(old,{"historical":True})
    directory=tmp_path/"artifacts/stats_cpu_v1/verification/this_run"
    evidence={"mode":"full","error_outputs":0,"executed_code_cells":1,"seconds":.1}
    script.write_execution_evidence(tmp_path,directory,evidence)
    assert read_json(directory/"notebook_execution.json") == evidence
    assert read_json(old) == {"historical":True}
    with pytest.raises(FileExistsError):
        script.write_execution_evidence(tmp_path,directory,evidence)
    with pytest.raises(ValueError,match="historical|results"):
        script.write_execution_evidence(tmp_path,tmp_path/"results",evidence)


def test_status_counts_current_benchmark_failure_even_if_general_list_is_empty(tmp_path):
    directory=status_fixture(tmp_path)
    benchmark=read_json(directory/"benchmark_manifest.json")
    benchmark["failures"]=[{"branch":"benchmark/current","error_type":"IdleCPUUnavailable","reason":"new timeout"}]
    benchmark["failed_models"]=1
    write_json(directory/"benchmark_manifest.json",benchmark)
    status=publication().execution_status(tmp_path)
    assert status["active_failure_count"] == 1
    assert status["effective_status"] != "complete"


def test_active_sources_have_no_obsolete_statistical_identifiers():
    flags=publication().active_scope_legacy_flags(ROOT)
    assert not [item for item in flags if item["path"].startswith(("src/","scripts/","tests/"))]


def test_notebook_experiment_never_calls_training_for_cached_demonstrations(tmp_path):
    from exlightnet.notebook import experiment
    expected={"status":"cached demonstration"}
    write_json(tmp_path/"results/fast/manifest.json",expected)
    assert experiment(tmp_path,"fast") == expected
    with pytest.raises(FileNotFoundError,match="only reads"):
        experiment(tmp_path,"full")
    assert not (tmp_path/"results/full").exists()


def test_presentation_rejects_tampered_or_mixed_campaign_tables(tmp_path):
    import hashlib
    import shutil
    source=ROOT/"artifacts/stats_cpu_v1"
    copied=tmp_path/"statistics"
    copied.mkdir()
    for path in [source/"statistics_manifest.json",*source.glob("*.csv")]:
        shutil.copyfile(path,copied/path.name)
    table=copied/"model_summary.csv"
    table.write_text(table.read_text().replace("cpu_v1","untrusted_campaign"))
    with pytest.raises(ValueError,match="checksum"):
        publication().load_statistics(ROOT,copied)
    manifest=read_json(copied/"statistics_manifest.json")
    manifest["tables"]["model_summary"]["sha256"]=hashlib.sha256(table.read_bytes()).hexdigest()
    write_json(copied/"statistics_manifest.json",manifest)
    with pytest.raises(ValueError,match="campaign"):
        publication().load_statistics(ROOT,copied)
