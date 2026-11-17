import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("minimum_ram_mb", [0, 3072])
def test_real_worker_processes_overlap_and_finish_before_aggregation(tmp_path, monkeypatch, minimum_ram_mb):
    spec = importlib.util.spec_from_file_location("pipeline", Path("scripts/run_parallel.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    readings = iter([1280, 4096])
    monkeypatch.setattr(module, "_available_memory_mb", lambda: next(readings, 4096))
    sleeps = []
    real_sleep = module.time.sleep
    def sleep(seconds):
        if seconds == 10:
            sleeps.append(seconds)
        else:
            real_sleep(min(seconds, .02))
    monkeypatch.setattr(module.time, "sleep", sleep)
    interpreter = tmp_path / "fixture-python"
    interpreter.write_text('''#!/usr/bin/python3
import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
root=Path(args[args.index("--root")+1]); d=root/"results/full"; d.mkdir(parents=True,exist_ok=True)
if "--alpha-only" in args:
 (root/"artifacts").mkdir(exist_ok=True)
 (root/"artifacts/cache_manifest.json").write_text(json.dumps({"fingerprint":"fixture"}))
 for task in ("binary","multiclass"):
  p=d/"alpha"/task/"selection.json"; p.parent.mkdir(parents=True,exist_ok=True)
  p.write_text(json.dumps({"chosen":"none","sweep_max_epochs":8,"data_fingerprint":"fixture"}))
elif "--seed-worker" in args:
 seed=int(args[args.index("--seeds")+1]); started=time.time()
 (d/f"started-{seed}").write_text(str(os.getpid()))
 deadline=time.time()+10
 while len(list(d.glob("started-*")))<5 and time.time()<deadline: time.sleep(.01)
 time.sleep(.05)
 (d/f"finished-{seed}.json").write_text(json.dumps({"pid":os.getpid(),"started":started,"finished":time.time(),"threads":os.environ["OMP_NUM_THREADS"]}))
elif "--aggregate-only" in args:
 assert len(list(d.glob("finished-*.json")))==5
 (d/"manifest.json").write_text(json.dumps({"full_five_seed_headlines_complete":True,"failures":[]}))
 (d/"aggregate-fixture").write_text("once")
elif "--benchmark-only" in args:
 assert (d/"aggregate-fixture").exists()
 (d/"benchmark_manifest.json").write_text(json.dumps({"failures":[]}))
''')
    interpreter.chmod(0o755)
    result = module.run_pipeline(tmp_path, python=str(interpreter), min_available_ram_mb=minimum_ram_mb)
    workers = [json.loads(p.read_text()) for p in (tmp_path / "results/full").glob("finished-*.json")]
    assert len(workers) == len({r["pid"] for r in workers}) == 5
    assert max(r["started"] for r in workers) < min(r["finished"] for r in workers)
    assert {r["threads"] for r in workers} == {"1"}
    stages = [e["stage"] for e in result["invocations"]]
    assert stages.count("alpha_only") == stages.count("aggregate_only") == stages.count("benchmark_only") == 1
    assert stages.count("seed_worker") == 5
    assert result["status"] == "complete"
    assert bool(sleeps) == (minimum_ram_mb > 0)
    assert result["worker_launch_memory"]["available_mb"] >= minimum_ram_mb
