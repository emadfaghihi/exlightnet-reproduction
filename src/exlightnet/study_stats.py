"""Verified, cache-only supplementary study aggregation from unextracted ZIPs.

No PyTorch, pickle, model fitting or prediction is imported or executed here.
The prepared manifest is the local trust anchor; archived records and every
referenced artifact are independently verified before any output is written.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .campaign import projected_neural_hours, run_fingerprint, study_runs
from .device import environment_identity
from .stats import (campaign_device_label, class_f1_from_confusion, flow_confusion,
                    holm_adjust, interpretation, load_campaign_rows,
                    macro_f1_from_confusion, paired_flow_bootstrap, paired_summary,
                    paired_tests)


STUDIES = ("B1", "B3", "B2", "B4", "B5")
INFRASTRUCTURE_STUDIES = ("probe", "determinism")
B4_ENVIRONMENT_LABEL = "cross-environment (Kaggle B1 references vs Colab uncapped60)"
TREES = {"random_forest", "xgboost"}
VECTOR_FIELDS = ("study", "arm", "task", "model")
RUN_FIELDS = (*VECTOR_FIELDS, "seed", "split_seed")
MAX_JSON = 16*1024*1024
MAX_ARRAY = 256*1024*1024
EMPTY_COLUMNS = {
    "per_run": ["campaign","study","source_study","arm","task","model","device","seed","split_seed","checkpoint","macro_f1","accuracy","best_val_macro_f1","epochs_run","best_epoch","max_epochs","lr","run_fingerprint","inference_eligible"],
    "summaries": ["campaign","study","arm","task","model","checkpoint","device","device_label","n","n_eligible","sample_label","observation_scope","macro_f1_mean","macro_f1_sd","macro_f1_seed42","zero_seed_variance","best_val_macro_f1_mean","epochs_run_mean","n_epoch_observed","n_hit_cap","unstable"],
    "paired_tests": ["campaign","study","arm","task","comparison_type","reference","model","reference_arm","comparison_arm","checkpoint","reference_device","comparison_device","device_label","seeds","n_pairs","mean_difference","difference_sd","t_statistic","p_value","ci_low","ci_high","cohens_dz","p_holm","interpretation"],
    "paired_flow_bootstrap": ["campaign","study","task","model","reference","reference_arm","comparison_arm","checkpoint","comparison_type","reference_device","comparison_device","device_label","mean_difference","ci_low","ci_high","pairing","replicates","n_seeds","sample_label","estimand"],
    "correlations": ["campaign","study","task","model","device","device_label","n","spearman_rho","p_value","ci_low","ci_high","bootstrap_replicates","defined_replicates","bootstrap_unit","interpretation"],
    "lr_selection": ["campaign","study","task","model","arm","device","device_label","n","lr","best_val_macro_f1_mean","selected_lr","selected","status"],
    "probe": ["campaign","device","study","seconds_per_epoch","projected_neural_hours","assumptions"],
    "determinism": ["campaign","device","checkpoint","max_abs_logit_difference","abs_macro_f1_difference"],
}


def _hash_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(name):
    if not isinstance(name,str) or not name or name.startswith("/") or "\\" in name or "\0" in name or re.match(r"^[A-Za-z]:",name):
        raise ValueError("Unsafe archive path")
    if any(part in ("", ".", "..") for part in name.rstrip("/").split("/")):
        raise ValueError("Unsafe archive path component")
    return str(PurePosixPath(name))


def _json(data):
    if len(data)>MAX_JSON:
        raise ValueError("JSON exceeds safe size")
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key]=value
        return result
    try:
        return json.loads(data,object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {value}")))
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise ValueError("Invalid archived JSON") from exc


def _array_file(data, keys):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            if sum(info.file_size for info in inner.infolist())>MAX_ARRAY:
                raise ValueError("Expanded NPZ exceeds safe size")
        with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
            return {key:arrays[key].copy() for key in keys}
    except (OSError,KeyError,zipfile.BadZipFile,EOFError) as exc:
        raise ValueError("Invalid prediction NPZ") from exc


class _ArchiveIndex:
    def __init__(self, archives, stack):
        self.members={}
        self.archive_evidence={}
        for path in dict.fromkeys(map(Path,archives)):
            archive=stack.enter_context(zipfile.ZipFile(path))
            self.archive_evidence[str(path)]={"sha256":_hash_file(path),"size_bytes":path.stat().st_size}
            seen=set()
            for item in archive.infolist():
                name=_safe_path(item.filename)
                mode=item.external_attr>>16
                if stat.S_ISLNK(mode):
                    raise ValueError("Archive symlink is forbidden")
                if stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR):
                    raise ValueError("Archive special file is forbidden")
                if name in seen:
                    raise ValueError("Duplicate archive member path")
                seen.add(name)
                if not item.is_dir():
                    self.members.setdefault(name,[]).append((archive,item))

    @staticmethod
    def read_member(archive,item,limit):
        if item.file_size>limit:
            raise ValueError("Archived data exceeds safe size")
        with archive.open(item) as stream:
            data=stream.read(limit+1)
        if len(data)>limit:
            raise ValueError("Archived data exceeds safe size")
        return data

    def verified(self,name,expected,*,read=True):
        name=_safe_path(name)
        if not isinstance(expected,str) or not re.fullmatch("[0-9a-f]{64}",expected):
            raise ValueError("Invalid artifact SHA256")
        found=False;result=None
        for archive,item in self.members.get(name,[]):
            digest=hashlib.sha256()
            content=bytearray() if read else None
            if read and item.file_size>MAX_ARRAY:
                raise ValueError("Archived artifact exceeds safe size")
            with archive.open(item) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b""):
                    digest.update(chunk)
                    if read:
                        content.extend(chunk)
                        if len(content)>MAX_ARRAY:
                            raise ValueError("Archived artifact exceeds safe size")
            if digest.hexdigest()!=expected:
                raise ValueError(f"Conflicting or SHA256-mismatched completed artifact: {name}")
            found=True
            if read and result is None:
                result=bytes(content)
        if not found:
            raise ValueError(f"Missing or SHA256-mismatched artifact: {name}")
        return result


def _identity(record):
    identity = (record["campaign"], *_run_identity(record))
    if record["study"] in INFRASTRUCTURE_STUDIES:
        environment = environment_identity(record["device"], record["libraries"])
        return (*identity, run_fingerprint(environment))
    return identity


def _run_identity(record):
    return tuple(record[key] for key in RUN_FIELDS)


def _vector_identity(record):
    return tuple(record[key] for key in VECTOR_FIELDS)


def _plan(record):
    plan={k:record[k] for k in ("study","task","model","seed","split_seed","arm")}
    if record["study"]=="B5":
        plan["lr"]=record["lr"]
    expected=[r for r in study_runs(record["study"]) if "reuse_study" not in r]
    if plan not in expected:
        raise ValueError("Record is outside the predeclared study plan")
    return plan


def _validate_recipe(record, manifest):
    config=record["config"]
    expected=dict(seed=record["seed"],max_epochs=3 if record["study"]=="probe" else 60 if record["study"]=="B4" else 20 if record["task"]=="binary" else 30,
        batch_size=512,patience=10 if record["arm"]=="patience10_min15" else 6,
        lr=record.get("lr",.0015),weight_decay=.0001,
        alpha_mode=manifest["frozen_alpha"][record["task"]],sampler="tempered",threads=1)
    if record["campaign"]!="gpu_v2":
        # Smoke has an explicitly separate campaign and shorter epoch cap.
        expected["max_epochs"]=config.get("max_epochs")
        if not isinstance(expected["max_epochs"],int) or expected["max_epochs"]<1:
            raise ValueError("Invalid supplementary epoch cap")
    if config!=expected:
        raise ValueError("Record configuration differs from the frozen study recipe")


def _validate_history(record):
    if record["model"] in TREES:
        if record["history"] or record["epochs_run"] is not None or record["best_epoch"] is not None:
            raise ValueError("Tree record has neural epoch metadata")
        return
    history=record["history"]
    if not history or record["epochs_run"]!=len(history) or [r["epoch"] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError("History and epochs_run disagree")
    scores=np.asarray([r["val_macro_f1"] for r in history],dtype=float)
    times=np.asarray([r["seconds"] for r in history],dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(times).all() or np.any(times<0):
        raise ValueError("Nonfinite history or negative epoch duration")
    best=int(np.argmax(scores))+1
    if record["best_epoch"]!=best or not np.isclose(record["best_val_macro_f1"],scores[best-1],atol=1e-12,rtol=0):
        raise ValueError("Best checkpoint was not selected by validation history")
    if len(history)>record["config"]["max_epochs"]:
        raise ValueError("History exceeds epoch cap")
    cap=record["config"]["max_epochs"]
    if record["arm"]=="fixed30" or record["study"]=="probe":
        if len(history)!=cap:
            raise ValueError(f"Completed {record['arm']} arm must run its full epoch budget")
        return
    # Replay fit_resumable exactly: strict validation improvement resets stale;
    # ties increment it. At the start of the NEXT epoch, stale>=patience and
    # epoch>min_epochs stop execution. Thus a legal terminal epoch is >=min.
    minimum=15 if record["arm"]=="patience10_min15" else 0
    patience=record["config"]["patience"]
    running_best=-1.;stale=0;earliest_stop=None
    for epoch,score in enumerate(scores,1):
        if score>running_best:
            running_best=float(score);stale=0
        else:
            stale+=1
        if stale>=patience and epoch>=minimum:
            earliest_stop=epoch
            break
    if earliest_stop is not None and len(history)>earliest_stop:
        raise ValueError("History continues past the earliest legal stopping epoch")
    if len(history)<cap and earliest_stop!=len(history):
        raise ValueError("Completed history terminates prematurely before stopping rule or cap")


def read_study_archives(archives,prepared_dir):
    """Return verified unique records plus loaded class labels/predictions.

    No archive member is extracted. Identical infrastructure records can
    deduplicate within an environment. Inferential duplicates and conflicting
    completed identities fail. Incomplete records are counted but never scored.
    Model files are hash-checked as opaque streams only.
    """
    archives=list(archives)
    if not archives:
        return {"records":[],"campaign":"gpu_v2","input_archives":{},"n_incomplete_records":0,"prepared_manifest_sha256":None}
    prepared_dir=Path(prepared_dir).resolve()
    manifest_path=prepared_dir/"manifest.json"
    manifest_bytes=manifest_path.read_bytes()
    manifest=_json(manifest_bytes)
    manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
    trusted={}
    with ExitStack() as stack:
        index=_ArchiveIndex(archives,stack)
        unique={};campaigns=set();incomplete=0
        for name,sources in sorted(index.members.items()):
            if PurePosixPath(name).name!="record.json":
                continue
            for archive,item in sources:
                record=_json(index.read_member(archive,item,MAX_JSON))
                if not isinstance(record,dict):
                    raise ValueError("Run record must be a JSON object")
                if record.get("campaign"):
                    campaigns.add(record["campaign"])
                if record.get("status")!="complete":
                    incomplete+=1
                    continue
                try:
                    key=_identity(record)
                except KeyError as exc:
                    raise ValueError("Incomplete completed-run identity") from exc
                if key in unique:
                    if unique[key][0]!=record:
                        raise ValueError("Conflicting duplicate completed identity")
                    if record["study"] not in INFRASTRUCTURE_STUDIES:
                        raise ValueError("Duplicate completed inferential identity")
                expected_path=f'{record["device"]}/{record["run_fingerprint"]}/record.json'
                if name!=expected_path:
                    raise ValueError("Record path/run fingerprint mismatch")
                unique[key]=(record,name)
        if len(campaigns)>1:
            raise ValueError("Mixed campaigns in archives are forbidden")
        campaign=next(iter(campaigns),"gpu_v2")
        if not re.fullmatch("[A-Za-z0-9_-]+",campaign) or campaign=="cpu_v1":
            raise ValueError("Invalid supplementary campaign")
        records=[]
        artifact_hashes={}
        for record,record_path in unique.values():
            required={"schema_version","status","campaign","study","arm","task","model","seed","split_seed","device","config","data_fingerprint","payload_sha256","prepared_sha256","run_fingerprint","libraries","history","best_epoch","best_val_macro_f1","epochs_run","headline_checkpoint","checkpoints"}
            if not required<=record.keys() or not all(type(record[k]) is int for k in ("seed","split_seed")):
                raise ValueError("Completed run is missing required fields or integer seed identities")
            if record.get("schema_version")!=1 or record.get("headline_checkpoint")!="best" or set(record.get("checkpoints",{}))!={"best","last"}:
                raise ValueError("Unsupported record schema/checkpoint declaration")
            if record["device"]!="cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?",record["device"]):
                raise ValueError("Record device must be resolved to CPU or single CUDA")
            if record["model"] in TREES and record["device"]!="cpu":
                raise ValueError("Tree models must run on the host CPU")
            plan=_plan(record)
            split=record["split_seed"]
            if split not in trusted:
                try:
                    entry=manifest["splits"][str(split)]
                    filename=_safe_path(entry.get("path",f"split_{split}.npz"))
                    path=(prepared_dir/filename).resolve()
                    if path.parent!=prepared_dir or _hash_file(path)!=entry["sha256"]:
                        raise ValueError("Trusted prepared file SHA256 mismatch")
                    with np.load(path,allow_pickle=False) as arrays:
                        labels={task:arrays[key].copy() for task,key in [("binary","y2_test"),("multiclass","y12_test")]}
                    trusted[split]=(entry,labels)
                except (KeyError,OSError) as exc:
                    raise ValueError("Prepared manifest split is missing or invalid") from exc
            entry,labels=trusted[split]
            for record_key,entry_key in [("data_fingerprint","data_fingerprint"),("payload_sha256","payload_sha256"),("prepared_sha256","sha256")]:
                if record.get(record_key)!=entry[entry_key]:
                    raise ValueError("Record data/payload/prepared fingerprint mismatch")
            _validate_recipe(record,manifest)
            _validate_history(record)
            identity=dict(campaign=campaign,plan=plan,device=record["device"],config=record["config"],
                data_fingerprint=record["data_fingerprint"],payload_sha256=record["payload_sha256"],
                prepared_sha256=record["prepared_sha256"],libraries=record["libraries"])
            if run_fingerprint(identity)!=record["run_fingerprint"]:
                raise ValueError("Completed run fingerprint is inconsistent with identity")
            names=["normal","attack"] if record["task"]=="binary" else entry["class_names"]
            loaded={}
            for checkpoint,item in record["checkpoints"].items():
                needed={"metrics","prediction_path","prediction_sha256","metrics_path","metrics_sha256","receipt_path","receipt_sha256"}
                if not isinstance(item,dict) or not needed<=item.keys():
                    raise ValueError("Checkpoint evidence is incomplete")
                content={}
                tree_alias=(record["model"] in TREES and checkpoint=="last" and item==record["checkpoints"]["best"])
                checkpoint_directory=PurePosixPath(record_path).parent/("best" if tree_alias else checkpoint)
                for key in ("prediction","metrics","receipt",*( ["state_dict"] if "state_dict_path" in item else [] ),*( ["logits"] if "logits_path" in item else [] )):
                    path=_safe_path(item[f"{key}_path"])
                    if not path.startswith(str(PurePosixPath(record_path).parent)+"/"):
                        raise ValueError("Checkpoint artifact path is outside its run")
                    if key!="state_dict" and PurePosixPath(path).parent!=checkpoint_directory:
                        raise ValueError("Checkpoint artifact path is outside its declared checkpoint")
                    content[key]=index.verified(path,item[f"{key}_sha256"],read=key!="state_dict")
                    artifact_hashes[path]=item[f"{key}_sha256"]
                metrics=_json(content["metrics"]);receipt=_json(content["receipt"])
                expected_fp=f'{record["run_fingerprint"]}:{checkpoint}'
                if tree_alias:
                    expected_fp=f'{record["run_fingerprint"]}:best'
                if receipt.get("status")!="complete" or receipt.get("fingerprint")!=expected_fp or receipt.get("test_prediction_passes")!=1:
                    raise ValueError("Incomplete or inconsistent checkpoint receipt")
                if receipt.get("prediction_sha256")!=item["prediction_sha256"] or receipt.get("metrics_sha256")!=item["metrics_sha256"] or metrics!=item["metrics"]:
                    raise ValueError("Receipt hashes or inline/file metrics disagree")
                arrays=_array_file(content["prediction"],("y","predictions"))
                y,pred=arrays["y"],arrays["predictions"]
                if y.dtype!=np.int8 or pred.dtype!=np.int8 or not np.array_equal(y,labels[record["task"]]):
                    raise ValueError("Prediction labels/dtype do not align with trusted prepared split")
                cm=flow_confusion(y,pred,len(names))
                if not np.array_equal(cm,metrics["confusion_matrix"]) or not np.isclose(macro_f1_from_confusion(cm),metrics["macro_f1"],rtol=0,atol=1e-12) or not np.isclose(np.mean(y==pred),metrics["accuracy"],rtol=0,atol=1e-12):
                    raise ValueError("Saved predictions and scored metrics disagree")
                if list(metrics["per_class"])!=names:
                    # JSON writers sort keys; compare mappings, not insertion order.
                    if set(metrics["per_class"])!=set(names):
                        raise ValueError("Per-class labels differ from prepared class axis")
                for i,class_name in enumerate(names):
                    value=metrics["per_class"][class_name]
                    if value["support"]!=int(cm[i].sum()) or not np.isclose(value["f1"],class_f1_from_confusion(cm)[i],rtol=0,atol=1e-12):
                        raise ValueError("Per-class F1/support differs from predictions")
                loaded[checkpoint]={"y":y,"prediction":pred}
                if "logits" in content:
                    if receipt.get("logits_sha256")!=item["logits_sha256"]:
                        raise ValueError("Logit receipt hash mismatch")
                    logits=_array_file(content["logits"],("logits",))["logits"]
                    if logits.shape!=(len(y),len(names)) or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1),pred):
                        raise ValueError("Logits do not match saved predictions")
                    loaded[checkpoint]["logits"]=logits
            record=copy.deepcopy(record)
            record["_environment"]=environment_identity(record["device"],record["libraries"])
            if record["device"].startswith("cuda") and any(value is None or value=="" for value in record["_environment"].values()):
                raise ValueError("CUDA execution environment lacks recorded hardware/runtime identity")
            record["_environment_id"]=run_fingerprint(record["_environment"])
            record["_predictions"]=loaded
            record["_record_path"]=record_path
            record["_classes"]=names
            records.append(record)
        # Device policy applies across studies. Environment policy applies to
        # each inferential vector, with reuse checked after references expand.
        model_devices={}
        vector_environments={}
        for record in records:
            model_devices.setdefault((record["task"],record["model"]),set()).add(record["device"])
            if record["study"] in STUDIES:
                vector_environments.setdefault(_vector_identity(record),set()).add(record["_environment_id"])
        if any(len(v)>1 for v in model_devices.values()):
            raise ValueError("Multiple devices for the same model seed vector")
        if any(len(v)>1 for v in vector_environments.values()):
            raise ValueError("Mixed actual execution environment for the same study/arm/task/model seed vector")
        device_rows=[];seen=set()
        for r in records:
            key=(r["task"],r["model"],r["seed"])
            if key not in seen:
                device_rows.append(dict(campaign=campaign,device=r["device"],task=r["task"],model=r["model"],seed=r["seed"],macro_f1=r["checkpoints"]["best"]["metrics"]["macro_f1"]))
                seen.add(key)
        device_label=campaign_device_label(device_rows) if device_rows else campaign
        if _hash_file(manifest_path)!=manifest_sha or any(_hash_file(path)!=value["sha256"] for path,value in index.archive_evidence.items()):
            raise ValueError("Input archive or prepared manifest changed during aggregation")
        return dict(records=records,campaign=campaign,input_archives=index.archive_evidence,
            n_incomplete_records=incomplete,prepared_manifest_sha256=manifest_sha,
            artifact_sha256=artifact_hashes,device_label=device_label,
            execution_environments={r["_environment_id"]:r["_environment"] for r in records})


def _study_plans(study):
    plans=study_runs(study)
    if study=="B4":
        # B4 declares its paper-20 comparisons against the B1 binary records.
        plans.extend([dict(plan,arm="paper",reuse_study="B1") for plan in plans])
    return plans


def _expanded_records(records):
    result=list(records)
    lookup={_run_identity(r):r for r in records if r["study"]=="B1"}
    observed_studies={r["study"] for r in records}
    for study in ("B3","B4"):
        if study not in observed_studies:
            continue
        for plan in _study_plans(study):
            if "reuse_study" in plan:
                source_study=plan["reuse_study"]
                record=lookup.get(_run_identity(dict(plan,study=source_study)))
                if record is not None:
                    result.append(dict(record,study=study,source_study=source_study))
    return result


def _check_b3_reference_environments(vectors):
    for (study,arm,task,model),record in vectors.items():
        if study=="B3" and arm!="paper":
            reference=vectors.get(("B3","paper",task,model))
            if reference is not None and reference["_environment_id"]!=record["_environment_id"]:
                raise ValueError(f"B3 requires the same B1 reference environment for {arm}/{task}/{model}")


def _label_b4_environments(tables,vectors):
    arm_environments={(arm,task,model): r["_environment_id"]
                      for (study,arm,task,model),r in vectors.items() if study=="B4"}
    paired_environments=[
        (environment,arm_environments.get(("uncapped60",task,model)))
        for (arm,task,model),environment in arm_environments.items() if arm=="paper"
    ]
    if not any(uncapped is not None and paper!=uncapped for paper,uncapped in paired_environments):
        return

    for name in ("summaries","paired_tests","paired_flow_bootstrap"):
        frame=tables[name]
        if frame.empty or not frame.study.eq("B4").any():
            continue
        frame["cross_environment"]=[True if row.study=="B4" else None for row in frame.itertuples()]
        for arm,column in (("paper","paper_environment_id"),("uncapped60","uncapped60_environment_id")):
            frame[column]=[arm_environments.get((arm,row.task,row.model)) if row.study=="B4" else None
                           for row in frame.itertuples()]


def _run_rows(records):
    rows=[]
    for r in records:
        for checkpoint in ("best","last"):
            metric=r["checkpoints"][checkpoint]["metrics"]
            rows.append(dict(campaign=r["campaign"],study=r["study"],source_study=r.get("source_study",r["study"]),
                arm=r["arm"],task=r["task"],model=r["model"],device=r["device"],seed=r["seed"],split_seed=r["split_seed"],
                checkpoint=checkpoint,macro_f1=metric["macro_f1"],accuracy=metric["accuracy"],
                best_val_macro_f1=r["best_val_macro_f1"],epochs_run=r["epochs_run"],best_epoch=r["best_epoch"],
                max_epochs=r["config"]["max_epochs"],lr=r["config"]["lr"],run_fingerprint=r["run_fingerprint"],
                environment_id=r["_environment_id"],execution_environment=json.dumps(r["_environment"],sort_keys=True),
                **{key:r["_environment"].get(key) for key in ("gpu_name","torch","cuda","cudnn","numpy","scikit-learn","xgboost")}))
    return rows


def _summary(rows,label,expected_seed_counts):
    output=[]
    frame=pd.DataFrame(rows)
    if frame.empty:
        return frame
    for keys,g in frame.groupby([*VECTOR_FIELDS,"checkpoint","environment_id"],sort=True):
        study,arm,task,model,checkpoint,environment_id=keys
        n=len(g);sd=g.macro_f1.std(ddof=1)
        n_eligible=int(g.inference_eligible.sum())
        observation_scope="descriptive"
        if study in expected_seed_counts:
            observation_scope=("complete; descriptive" if n==n_eligible==expected_seed_counts[study]
                               else "partial; observed runs only; descriptive")
        epochs=g.dropna(subset=["epochs_run"])
        output.append(dict(campaign=g.iloc[0].campaign,study=study,arm=arm,task=task,model=model,checkpoint=checkpoint,
            device=g.iloc[0].device,device_label=label,n=n,sample_label="n=5, indicative only" if n==5 else f"n={n}",
            n_eligible=n_eligible,observation_scope=observation_scope,
            environment_id=environment_id,execution_environment=g.iloc[0].execution_environment,
            macro_f1_mean=g.macro_f1.mean(),macro_f1_sd=sd,
            macro_f1_seed42=g[g.seed==42].macro_f1.iloc[0] if (g.seed==42).any() else np.nan,
            zero_seed_variance=bool(n>1 and g.macro_f1.nunique()==1),
            best_val_macro_f1_mean=g.best_val_macro_f1.mean(),epochs_run_mean=epochs.epochs_run.mean(),
            n_epoch_observed=len(epochs),n_hit_cap=int((epochs.epochs_run>=epochs.max_epochs).sum()) if len(epochs) else np.nan,
            unstable=bool((g.best_epoch.dropna()<=3).any() or (np.isfinite(sd) and sd>.05))))
    return pd.DataFrame(output)


def _pair_tables(records,rows,replicates,random_seed,label):
    output=[];bootstraps=[]
    frame=pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(),pd.DataFrame()
    lookup={(r["study"],r["arm"],r["task"],r["model"],r["seed"]):r for r in records}
    comparisons=[]
    for (study,arm,checkpoint),g in frame[frame.study.isin(STUDIES)].groupby(["study","arm","checkpoint"]):
        data=g.to_dict("records")
        for result in paired_tests(data).to_dict("records"):
            result.update(study=study,arm=arm,checkpoint=checkpoint,comparison_type="model",reference_arm=arm,comparison_arm=arm)
            comparisons.append(result)
    for study,baseline,arms in [("B3","paper",("fixed30","patience10_min15")),("B4","paper",("uncapped60",)),("B5","lr_0.0015",("lr_0.0003","lr_0.001"))]:
        for (task,model,checkpoint),g in frame[frame.study==study].groupby(["task","model","checkpoint"]):
            paper=g[g.arm==baseline].set_index("seed")
            for arm in arms:
                experimental=g[g.arm==arm].set_index("seed")
                seeds=sorted(set(paper.index)&set(experimental.index))
                if not seeds:
                    continue
                summary=paired_summary(experimental.loc[seeds,"macro_f1"],paper.loc[seeds,"macro_f1"])
                comparisons.append(dict(campaign=g.iloc[0].campaign,study=study,task=task,model=model,reference=model,
                    arm=arm,checkpoint=checkpoint,comparison_type="arm",reference_arm=arm,comparison_arm=baseline,
                    seeds=",".join(map(str,seeds)),reference_device=experimental.iloc[0].device,
                    comparison_device=paper.iloc[0].device,device_label=label,**summary))
    for result in comparisons:
        study,task,model,checkpoint=result["study"],result["task"],result["model"],result["checkpoint"]
        seed_caches=[]
        for seed in map(int,result["seeds"].split(",")):
            a=lookup[(study,result["reference_arm"],task,result["reference"],seed)]
            b=lookup[(study,result["comparison_arm"],task,model,seed)]
            if a["split_seed"]!=b["split_seed"] or a["data_fingerprint"]!=b["data_fingerprint"] or a["payload_sha256"]!=b["payload_sha256"]:
                raise ValueError("Paired runs have different split/data fingerprints")
            pa,pb=a["_predictions"][checkpoint],b["_predictions"][checkpoint]
            seed_caches.append(dict(y=pa["y"],reference=pa["prediction"],prediction=pb["prediction"],prediction_y=pb["y"]))
        bootstrap=paired_flow_bootstrap(seed_caches,len(a["_classes"]),replicates,random_seed)
        result["reference_environment_id"]=a["_environment_id"]
        result["comparison_environment_id"]=b["_environment_id"]
        result["reference_environment"]=json.dumps(a["_environment"],sort_keys=True)
        result["comparison_environment"]=json.dumps(b["_environment"],sort_keys=True)
        seed_unit="split_seed (train seed equals split seed)" if study=="B2" else "training_seed"
        bootstraps.append({k:result[k] for k in ("campaign","study","task","model","reference","reference_arm","comparison_arm","checkpoint","comparison_type","reference_device","comparison_device")}|dict(device_label=label,**bootstrap))
        bootstraps[-1]["seed_unit"]=seed_unit
        bootstraps[-1].update({key:result[key] for key in ("reference_environment_id","comparison_environment_id","reference_environment","comparison_environment")})
        result["pairing_unit"]=seed_unit
        result["device_label"]=label
        output.append(result)
    paired=pd.DataFrame(output)
    if not paired.empty:
        paired["p_holm"]=np.nan
        for _,g in paired.groupby(["study","task"]):
            paired.loc[g.index,"p_holm"]=holm_adjust(g.p_value)
        paired["interpretation"]=[interpretation(r.ci_low,r.ci_high,r.p_holm,r.task) for r in paired.itertuples()]
        paired["holm_family"]="all available predeclared comparisons within task and study, best and last included"
    return paired,pd.DataFrame(bootstraps)


def _correlations(records,replicates,random_seed,label):
    rows=[]
    groups={}
    for record in records:
        if record["study"]=="B1":
            groups.setdefault((record["task"],record["model"]),[]).append(record)
    for (task,model),group in sorted(groups.items()):
        observed=sorted((r for r in group if r["epochs_run"] is not None),key=lambda r:r["seed"])
        x=np.asarray([r["epochs_run"] for r in observed],dtype=float)
        y=np.asarray([r["checkpoints"]["best"]["metrics"]["macro_f1"] for r in observed])
        defined=len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0
        rho,p=scipy_stats.spearmanr(x,y) if defined else (np.nan,np.nan)
        rng=np.random.default_rng(random_seed); draws=[]
        if defined:
            for _ in range(replicates):
                indices=rng.integers(0,len(x),len(x));xx,yy=x[indices],y[indices]
                if np.ptp(xx)>0 and np.ptp(yy)>0:
                    draws.append(float(scipy_stats.spearmanr(xx,yy).statistic))
        low,high=np.quantile(draws,[.025,.975]) if draws else (np.nan,np.nan)
        rows.append(dict(campaign=group[0]["campaign"],study="B1",task=task,model=model,device=group[0]["device"],device_label=label,
            n=len(group),n_epoch_observed=len(observed),spearman_rho=rho,p_value=p,ci_low=low,ci_high=high,bootstrap_replicates=replicates,
            defined_replicates=len(draws),bootstrap_unit="paired training-seed (epochs,F1) observations",
            sample_label="n=5, indicative only" if len(group)==5 else f"n={len(group)}",
            interpretation="descriptive only, not causal"))
    return pd.DataFrame(rows)


def _lr_selection(records,label):
    rows=[]
    group=[r for r in records if r["study"]=="B5"]
    if not group:
        return pd.DataFrame(columns=["campaign","task","selected_lr","status"])
    expected=study_runs("B5")
    expected_ids={(p["arm"],p["seed"]) for p in expected}
    observed={(r["arm"],r["seed"]) for r in group}
    complete=observed==expected_ids
    val={}
    for arm in sorted({r["arm"] for r in group}):
        arm_rows=[r for r in group if r["arm"]==arm]
        val[arm]=np.mean([r["best_val_macro_f1"] for r in arm_rows])
    selected=min(val,key=lambda arm:(-val[arm],float(arm.removeprefix("lr_")))) if complete else None
    for arm,value in val.items():
        rows.append(dict(campaign=group[0]["campaign"],study="B5",task="multiclass",model="std_cnn",arm=arm,
            device=group[0]["device"],device_label=label,n=sum(r["arm"]==arm for r in group),
            lr=float(arm.removeprefix("lr_")),best_val_macro_f1_mean=value,
            selected_lr=float(selected.removeprefix("lr_")) if selected else np.nan,
            selected=arm==selected if complete else None,
            observation_scope="complete; descriptive" if complete else "partial; observed validation runs only; descriptive",
            status="validation-only selection; tie chooses lower LR" if complete else "PENDING: complete equal seed sets for all three LR arms required"))
    return pd.DataFrame(rows)


def _completeness(records):
    """Match verified records to whole predeclared study seed blocks."""
    observed={_run_identity(r) for r in records if r["study"] in STUDIES}
    rows=[]
    eligible={}
    for study in STUDIES:
        blocks={}
        for plan in _study_plans(study):
            blocks.setdefault(plan["seed"],[]).append(plan)
        eligible[study]=set()
        seed_rows=[]
        totals=dict(expected_new_runs=0,completed_new_runs=0,
                    expected_b1_references=0,available_b1_references=0)
        for seed,plans in sorted(blocks.items()):
            counts=dict.fromkeys(totals,0)
            missing=[]
            for plan in plans:
                identity=dict(plan,study=plan.get("reuse_study",study))
                present=_run_identity(identity) in observed
                if "reuse_study" in plan:
                    counts["expected_b1_references"]+=1
                    counts["available_b1_references"]+=present
                else:
                    counts["expected_new_runs"]+=1
                    counts["completed_new_runs"]+=present
                if not present:
                    missing.append({key:identity[key] for key in RUN_FIELDS})
            if not missing:
                eligible[study].add(seed)
            for key in totals:
                totals[key]+=counts[key]
            seed_rows.append(dict(study=study,row_type="seed",seed=seed,split_seed=plans[0]["split_seed"],
                eligible_seeds=str(seed) if not missing else "",expected_seed_blocks=1,n_eligible=int(not missing),
                **counts,missing_count=len(missing),missing_identities=json.dumps(missing,sort_keys=True),
                status="incomplete; excluded" if missing else "complete"))
        if totals["completed_new_runs"]==0:
            status="PENDING: study not run"
        else:
            status="complete" if len(eligible[study])==len(blocks) else "PARTIAL: missing planned runs or B1 references"
        rows.append(dict(study=study,row_type="study",seed=None,split_seed=None,
            expected_seed_blocks=len(blocks),n_eligible=len(eligible[study]),
            eligible_seeds=",".join(map(str,sorted(eligible[study]))),**totals,
            missing_count=sum(row["missing_count"] for row in seed_rows),status=status))
        rows.extend(seed_rows)
    frame=pd.DataFrame(rows).astype({"seed":"Int64","split_seed":"Int64"})
    return frame,eligible


def _device_equivalence(records,reference_cpu):
    columns=["task","model","reference_campaign","comparison_campaign","reference_device","comparison_device","n_pairs","cpu_v1_mean","cpu_v1_sd","cpu_v1_seed42","gpu_v2_mean","gpu_v2_sd","gpu_v2_seed42","mean_difference","ci_low","ci_high","sample_label"]
    gpu=[r for r in records if r["campaign"]=="gpu_v2" and r["study"]=="B1" and r["split_seed"]==42 and 42<=r["seed"]<=46]
    if not gpu or reference_cpu is None:
        return pd.DataFrame(columns=columns)
    cpu=load_campaign_rows(reference_cpu)
    if any(r["campaign"]!="cpu_v1" or r["device"]!="cpu" for r in cpu):
        raise ValueError("Device-equivalence requires an explicit cpu_v1 CPU reference")
    lookup={(r["task"],r["model"],r["seed"]):r for r in cpu}
    groups={}
    for record in gpu:
        key=(record["task"],record["model"],record["seed"])
        if key in lookup:
            original=lookup[key]
            if original["data_fingerprint"]!=record["data_fingerprint"]:
                raise ValueError("Device-equivalence reference data differs")
            groups.setdefault(key[:2],[]).append((original,record))
    rows=[]
    for (task,model),group in sorted(groups.items()):
        summary=paired_summary([r["checkpoints"]["best"]["metrics"]["macro_f1"] for _,r in group],[r["macro_f1"] for r,_ in group])
        rows.append(dict(task=task,model=model,reference_campaign="cpu_v1",comparison_campaign=group[0][1]["campaign"],reference_device="cpu",comparison_device=group[0][1]["device"],
            n_pairs=summary["n_pairs"],cpu_v1_mean=summary["comparison_mean"],cpu_v1_sd=summary["comparison_sd"],
            cpu_v1_seed42=next((a["macro_f1"] for a,b in group if a["seed"]==42),np.nan),
            gpu_v2_mean=summary["reference_mean"],gpu_v2_sd=summary["reference_sd"],
            gpu_v2_seed42=next((b["checkpoints"]["best"]["metrics"]["macro_f1"] for a,b in group if b["seed"]==42),np.nan),
            mean_difference=summary["mean_difference"],ci_low=summary["ci_low"],ci_high=summary["ci_high"],sample_label=summary["sample_label"]))
    return pd.DataFrame(rows,columns=columns)


def _stopping_endpoint(fixed,paper,amendment):
    """Compute the paired t interval and percentile interval over paired seeds."""
    bootstrap=amendment["sd_ratio_bootstrap"]
    paired=paired_summary(fixed,paper)
    paper_sd=0. if paired["comparison_zero_seed_variance"] else paired["comparison_sd"]
    fixed_sd=0. if paired["reference_zero_seed_variance"] else paired["reference_sd"]
    higher_mean=bool(paired["ci_low"]>0)
    ratio=low=high=np.nan
    smaller_sd=None
    defined=0
    decision="inconclusive"
    if paper_sd==0:
        reason="paper sample SD is zero"
    else:
        ratio=fixed_sd/paper_sd
        rng=np.random.default_rng(bootstrap["random_seed"])
        indices=rng.integers(0,len(paper),size=(bootstrap["replicates"],len(paper)))
        paper_draws=paper[indices]
        valid=np.ptp(paper_draws,axis=1)>0
        defined=int(valid.sum())
        if not valid.all():
            reason="one or more paired bootstrap draws have zero paper SD"
        else:
            fixed_draws=fixed[indices]
            # Center on the first sampled value to avoid roundoff from offsets
            # between the arms, including a constant fixed30 resample.
            fixed_draw_sd=np.std(fixed_draws-fixed_draws[:,:1],axis=1,ddof=1)
            paper_draw_sd=np.std(paper_draws-paper_draws[:,:1],axis=1,ddof=1)
            alpha=(1-bootstrap["confidence_level"])/2
            low,high=np.quantile(fixed_draw_sd/paper_draw_sd,[alpha,1-alpha])
            smaller_sd=bool(ratio<=amendment["sd_ratio_maximum"] and high<bootstrap["upper_bound_strictly_below"])
            decision="conditions supported" if smaller_sd and higher_mean else "not supported"
            reason="both predeclared conditions met" if decision=="conditions supported" else "one or both predeclared conditions unmet"
    return dict(paper_mean=paired["comparison_mean"],paper_sd=paper_sd,fixed30_mean=paired["reference_mean"],fixed30_sd=fixed_sd,
        mean_difference=paired["mean_difference"],mean_difference_ci_low=paired["ci_low"],mean_difference_ci_high=paired["ci_high"],
        sd_ratio=ratio,sd_ratio_ci_low=low,sd_ratio_ci_high=high,defined_replicates=defined,
        smaller_sd=smaller_sd,higher_mean=higher_mean,endpoint_decision=decision,decision_reason=reason)


def _hypothesis(records,correlations,amendment):
    """Evaluate both designated ExLightNet endpoints on eligible B3 blocks."""
    study,task,model="B3","multiclass","exlightnet"
    lookup={(r["arm"],r["seed"]):r["checkpoints"] for r in records
            if (r["study"],r["task"],r["model"])==(study,task,model)}
    b1={}
    if not correlations.empty:
        matches=correlations[(correlations.task==task)&(correlations.model==model)]
        if not matches.empty:
            b1=matches.iloc[0]
    required=amendment["matched_seeds"]
    bootstrap=amendment["sd_ratio_bootstrap"]
    rows=[]
    for role in ("PRIMARY","SECONDARY"):
        endpoint=amendment[f"{role.lower()}_endpoint"]
        experimental=endpoint["experimental_arm"],endpoint["experimental_checkpoint"]
        reference=endpoint["reference_arm"],endpoint["reference_checkpoint"]
        seeds=[seed for seed in required if all(checkpoint in lookup.get((arm,seed),{})
                                               for arm,checkpoint in (experimental,reference))]
        missing=sorted(set(required)-set(seeds))
        row=dict(study=study,task=task,model=model,endpoint_role=role,**endpoint,
            n_pairs=len(seeds),seeds=",".join(map(str,seeds)),missing_seeds=",".join(map(str,missing)),
            paper_mean=np.nan,paper_sd=np.nan,fixed30_mean=np.nan,fixed30_sd=np.nan,
            mean_difference=np.nan,mean_difference_ci_low=np.nan,mean_difference_ci_high=np.nan,
            sd_ratio=np.nan,sd_ratio_ci_low=np.nan,sd_ratio_ci_high=np.nan,
            bootstrap_replicates=bootstrap["replicates"],bootstrap_random_seed=bootstrap["random_seed"],defined_replicates=0,
            smaller_sd=None,higher_mean=None,endpoint_decision="PENDING",decision_reason="all declared matched seed blocks are required",
            b1_n=b1.get("n",0),b1_spearman_rho=b1.get("spearman_rho",np.nan),
            b1_ci_low=b1.get("ci_low",np.nan),b1_ci_high=b1.get("ci_high",np.nan))
        if not missing:
            fixed,paper=(np.asarray([lookup[(arm,seed)][checkpoint]["metrics"]["macro_f1"] for seed in seeds])
                         for arm,checkpoint in (experimental,reference))
            row.update(_stopping_endpoint(fixed,paper,amendment))
        rows.append(row)
    decisions={row["endpoint_decision"] for row in rows}
    overall=("PENDING" if "PENDING" in decisions else next(iter(decisions))
             if len(decisions)==1 else "inconclusive")
    return pd.DataFrame(rows).assign(overall_decision=overall)


def _probe_determinism(records):
    probes=[];determinism=[];repeats_by_environment={}
    for record in records:
        environment_id=record["_environment_id"]
        if record["study"]=="probe":
            seconds=float(np.mean([r["seconds"] for r in record["history"]]))
            for study in STUDIES:
                probes.append(dict(campaign=record["campaign"],device=record["device"],study=study,seconds_per_epoch=seconds,
                    projected_neural_hours=projected_neural_hours(study,seconds),assumptions="ExLightNet epoch speed for all neural models; ignores trees/model speed differences/early stopping",
                    environment_id=environment_id))
        elif record["study"]=="determinism":
            repeats_by_environment.setdefault(environment_id,{})[record["arm"]]=record
    for environment_id,repeats in repeats_by_environment.items():
        if not {"repeat1","repeat2"}<=repeats.keys():
            continue
        a,b=repeats["repeat1"],repeats["repeat2"]
        for cp in ("best","last"):
            pa,pb=a["_predictions"][cp],b["_predictions"][cp]
            if "logits" not in pa or "logits" not in pb:
                raise ValueError("Determinism study lacks verified logits")
            if a["payload_sha256"]!=b["payload_sha256"] or not np.array_equal(pa["y"],pb["y"]):
                raise ValueError("Determinism repeat data differs")
            determinism.append(dict(campaign=a["campaign"],device=a["device"],checkpoint=cp,
                max_abs_logit_difference=float(np.max(np.abs(pa["logits"]-pb["logits"]))),
                abs_macro_f1_difference=abs(a["checkpoints"][cp]["metrics"]["macro_f1"]-b["checkpoints"][cp]["metrics"]["macro_f1"]),
                environment_id=environment_id))
    return pd.DataFrame(probes),pd.DataFrame(determinism)


def _descriptive_environment_repeats(records):
    references={_run_identity(r):r for r in records if r["study"]=="B1" and r["device"]=="cuda"}
    reference_environments={r["_environment_id"] for r in references.values()}
    rows=[]
    for record in records:
        b2_repeat=record["study"]=="B2" and record["split_seed"]==42
        b5_repeat=record["study"]=="B5" and record["arm"]=="lr_0.0015"
        if record["device"]!="cuda" or not (b2_repeat or b5_repeat):
            continue
        reference=references.get(_run_identity(dict(record,study="B1",arm="paper")))
        if reference is None:
            if reference_environments-{record["_environment_id"]}:
                rows.append({key:record[key] for key in ("study","task","model","seed")}
                            | dict(best="no B1 reference",last="no B1 reference"))
            continue
        if record["_environment_id"]==reference["_environment_id"]:
            continue
        row={key:record[key] for key in ("study","task","model","seed")}
        if any(record[key]!=reference[key] for key in ("config","data_fingerprint","payload_sha256","prepared_sha256")):
            row.update(best="config or data mismatch",last="config or data mismatch")
        else:
            for checkpoint in ("best","last"):
                a,b=(r["_predictions"][checkpoint]["prediction"] for r in (reference,record))
                f1_a,f1_b=(r["checkpoints"][checkpoint]["metrics"]["macro_f1"] for r in (reference,record))
                predictions="exact" if np.array_equal(a,b) else "different"
                macro_f1="exact" if f1_a==f1_b else "different"
                row[checkpoint]=f"{predictions} / {macro_f1}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["study","task","model","seed"]) if rows else pd.DataFrame()


def _markdown(frame):
    if frame.empty:
        return "PENDING: no verified observations"
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value,(float,np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|","\\|").replace("\n"," ")
    lines=["| "+" | ".join(map(str,frame.columns))+" |","| "+" | ".join("---" for _ in frame.columns)+" |"]
    lines.extend("| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)


def _report(protocol,campaign,tables,device_label,stopping,blocks,environment_repeats,multiple_environments):
    lines=[f"# Supplementary accuracy studies: {campaign}","", "Source protocol: [`docs/statistics_protocol.json`](docs/statistics_protocol.json). Rules were recorded before Stage B data. General study summaries headline the best-validation checkpoint and list the last epoch second. The stopping hypothesis follows its designated primary and secondary checkpoints below. No test metric selects an epoch, checkpoint, learning rate or arm.","", "## A-priori interpretation rules",""]
    original=protocol["rules_verbatim"]
    superseded=stopping["supersedes_rule_index"]
    lines.extend("- "+rule for index,rule in enumerate(original) if index!=superseded)
    lines.extend(["",f"### {stopping['id']} ({stopping['date']}) — Stopping hypothesis",""])
    lines.extend("- "+rule for rule in stopping["rules_verbatim"])
    lines.extend(["",f"### {blocks['id']} ({blocks['date']}) — Complete seed blocks","",
                  blocks["rule"],"",blocks["clarification"],"",
                  "### Superseded original stopping-hypothesis rule","",original[superseded]])
    lines.extend(["",protocol["bootstrap_interpretation"],"", "Fixed split/training-seed variation and split-seed variation are reported separately. Tiny-class support makes macro-F1 noisy; unstable controls and zero-variance baselines cannot establish superiority. n=5, indicative only. Correlations are descriptive, not causal.","",f"Device label: {device_label}.",""])
    if campaign!="gpu_v2":
        lines.extend(["**Separate smoke/supplementary campaign. These observations do not fill canonical gpu_v2 studies.**",""])
    completeness=tables["completeness"].query('row_type == "study"').set_index("study")
    for study in STUDIES:
        complete=completeness.loc[study]
        status=complete.status
        lines.extend([f"## {study}: {protocol['studies'][study]['title']}","",status,""])
        lines.extend([f"Complete seed blocks for paired analyses: {complete.n_eligible} of {complete.expected_seed_blocks}. "
                      f"Eligible seeds: {complete.eligible_seeds or 'none'}. Missing run identities are recorded by seed in `completeness.csv`.",""])
        if status=="PENDING: study not run" and study!="B3":
            continue
        lines.extend([_markdown(tables["summaries"][tables["summaries"].study==study]),""])
        paired=tables["paired_tests"]
        if not paired.empty:
            display=["task","comparison_type","reference","model","reference_arm","comparison_arm","checkpoint","reference_device","comparison_device","n_pairs","mean_difference","ci_low","ci_high","p_value","p_holm","interpretation"]
            study_pairs=paired[paired.study==study]
            if study=="B4" and "cross_environment" in paired:
                study_pairs=study_pairs.assign(interpretation=study_pairs.interpretation+f" ({B4_ENVIRONMENT_LABEL})")
                display.extend(["cross_environment","paper_environment_id","uncapped60_environment_id"])
            lines.extend(["Paired reference differences (positive favors the reference); Holm family is task/study, including best and last comparisons:","",_markdown(study_pairs[display]),""])
        flow=tables["paired_flow_bootstrap"]
        if not flow.empty:
            lines.extend(["Paired seed-then-flow bootstrap; single-random-run uncertainty:","",_markdown(flow[flow.study==study].drop(columns=["estimand"],errors="ignore")),""])
        if study=="B1":
            lines.extend(["Descriptive epochs-versus-F1 correlation and paired-seed bootstrap CI:","",_markdown(tables["correlations"]),""])
        if study=="B3":
            endpoints=tables["early_stopping_hypothesis"]
            roles=["endpoint_role","experimental_arm","experimental_checkpoint","reference_arm","reference_checkpoint",
                   "n_pairs","missing_seeds","smaller_sd","higher_mean","endpoint_decision","decision_reason"]
            effects=["endpoint_role","paper_mean","paper_sd","fixed30_mean","fixed30_sd","mean_difference",
                     "mean_difference_ci_low","mean_difference_ci_high","sd_ratio","sd_ratio_ci_low","sd_ratio_ci_high",
                     "defined_replicates"]
            b1=["b1_n","b1_spearman_rho","b1_ci_low","b1_ci_high"]
            lines.extend(["B1 supplies the paper-rule checkpoint for each seed. The mean-difference interval uses the paired Student t distribution. The SD-ratio interval uses paired-seed percentile resampling.","",
                f"Overall stopping-hypothesis decision — **{endpoints.overall_decision.iloc[0]}**.","",
                _markdown(endpoints[roles]),"",_markdown(endpoints[effects]),"",
                "B1 descriptive epochs-versus-F1 correlation for multiclass ExLightNet and its paired-seed bootstrap interval.","",
                _markdown(endpoints[b1].drop_duplicates()),""])
        if study=="B4":
            lines.extend(["The paper-20 reference is reused from B1. Cap counts and best/last results appear above; no reference is regenerated.",""])
            if "cross_environment" in tables["summaries"]:
                lines.extend([f"**{B4_ENVIRONMENT_LABEL}**. The training recipe and execution environment changed together.",""])
        if study=="B5":
            lines.extend(["Learning-rate selection uses validation means only, requires every predeclared seed in all three arms, and breaks a validation tie toward lower LR:","",_markdown(tables["lr_selection"]),""])
    if not environment_repeats.empty:
        lines.extend(["## Descriptive cross-environment repeats","","B2 split 42 and B5 `lr_0.0015` are matched to B1 paper runs by task, model, seed, split, full configuration and data identifiers. Each checkpoint cell shows exact agreement for saved predictions / macro-F1. This comparison is descriptive.","",_markdown(environment_repeats),""])
    if multiple_environments:
        lines.extend(["## Optional B6 leakage PROXY","","NOT RUN",""])
    lines.extend(["## Device-equivalence view", "", "Descriptive side-by-side differences and confidence intervals for matched seeds 42–46 with complete gpu_v2 B1 blocks. Differences are gpu_v2 minus cpu_v1.","",_markdown(tables["device_equivalence"]),"", "## Probe and determinism studies", "",_markdown(tables["probe"]),"",_markdown(tables["determinism"]),"", "CUDA execution is NOT RUN locally. Cached hosted GPU observations, if imported, do not imply local CUDA validation.","", "## Supplementary frozen/exported-model latency", "", "NOT RUN: this task prohibits benchmark reruns. Canonical CPU latency/size/INT8/memory evidence stays separate.",""])
    return "\n".join(lines)


def aggregate_studies(archives,prepared_dir,output,report,reference_cpu=None,*,protocol_path=None,replicates=2000,random_seed=42):
    """Verify ZIP evidence, write CSVs/report/manifest, return path mapping.

    Empty archives produce a canonical PENDING skeleton without reading arrays.
    A non-gpu_v2 campaign cannot write a report named REPORT_stats_v2.md.
    """
    repository=Path(__file__).resolve().parents[2]
    protocol_path=Path(protocol_path or repository/"docs/statistics_protocol.json")
    protocol=_json(protocol_path.read_bytes())
    if not protocol.get("declared_before_stage_b_data") or replicates<1:
        raise ValueError("A predeclared protocol and positive bootstrap count are required")
    amendments={row["id"]:row for row in protocol.get("amendments",[])}
    if not {"PA-20261118-1","PA-20261118-2"}<=amendments.keys():
        raise ValueError("The predeclared stopping-hypothesis and seed-block amendments are required")
    stopping,blocks=(amendments[key] for key in ("PA-20261118-1","PA-20261118-2"))
    loaded=read_study_archives(archives,prepared_dir)
    records=loaded["records"];campaign=loaded["campaign"]
    output,report=Path(output).resolve(),Path(report).resolve()
    if campaign!="gpu_v2" and report.name=="REPORT_stats_v2.md":
        raise ValueError("Non-gpu_v2 observations cannot write the canonical report")
    for forbidden in (repository/"results",Path(prepared_dir).resolve(),repository/"artifacts/stats_cpu_v1"):
        if output==forbidden or forbidden in output.parents:
            raise ValueError("Output directory overlaps immutable/prepared evidence")
        if report==forbidden or forbidden in report.parents:
            raise ValueError("Report path overlaps immutable/prepared evidence")
    if report.suffix.lower()!=".md" or report==repository/"REPORT.md":
        raise ValueError("Use a separate Markdown statistics report path")
    label=loaded.get("device_label","no observations")
    expanded=_expanded_records(records)
    vectors={_vector_identity(r):r for r in expanded if r["study"] in STUDIES}
    _check_b3_reference_environments(vectors)
    completeness,eligible=_completeness(records)
    rows=_run_rows(expanded)
    for row in rows:
        row["device_label"]=label
        row["inference_eligible"]=row["seed"] in eligible.get(row["study"],set())
    inferential=[record for record in expanded if record["seed"] in eligible.get(record["study"],set())]
    expected={row.study:row.expected_seed_blocks for row in completeness.itertuples() if row.row_type=="study"}
    summaries=_summary(rows,label,expected)
    paired,flow=_pair_tables(inferential,[row for row in rows if row["inference_eligible"]],replicates,random_seed,label)
    correlations=_correlations(inferential,replicates,random_seed,label)
    probe,determinism=_probe_determinism(records)
    multiple_environments=len(loaded.get("execution_environments",{}))>1
    if not multiple_environments:
        # A single environment is identified by the manifest in legacy outputs.
        probe=probe.drop(columns="environment_id",errors="ignore")
        determinism=determinism.drop(columns="environment_id",errors="ignore")
    tables=dict(per_run=pd.DataFrame(rows),summaries=summaries,paired_tests=paired,paired_flow_bootstrap=flow,
        correlations=correlations,lr_selection=_lr_selection(records,label),completeness=completeness,
        device_equivalence=_device_equivalence(inferential,reference_cpu),early_stopping_hypothesis=_hypothesis(inferential,correlations,stopping),
        probe=probe,determinism=determinism)
    _label_b4_environments(tables,vectors)
    environments={key:(r["_environment_id"],json.dumps(r["_environment"],sort_keys=True)) for key,r in vectors.items()}
    for name,frame in tables.items():
        if name!="device_equivalence":
            if "device_label" not in frame:
                frame["device_label"]=label
            if "campaign" not in frame:
                frame["campaign"]=campaign
            if not frame.empty and {"task","model"}<=set(frame.columns) and "environment_id" not in frame and "reference_environment_id" not in frame:
                # B1 correlations and the B3 hypothesis use the paper reference.
                sources=[environments.get((r.study,getattr(r,"arm","paper"),r.task,r.model),(None,None)) for r in frame.itertuples()]
                frame["environment_id"]=[environment_id for environment_id,_ in sources]
                frame["execution_environment"]=[metadata for _,metadata in sources]
    for name,columns in EMPTY_COLUMNS.items():
        if tables[name].empty:
            if multiple_environments and name in INFRASTRUCTURE_STUDIES:
                columns=[*columns,"environment_id"]
            tables[name]=tables[name].reindex(columns=list(dict.fromkeys(columns+["device_label","campaign"])))
    environment_repeats=_descriptive_environment_repeats(records)
    report_text=_report(protocol,campaign,tables,label,stopping,blocks,environment_repeats,multiple_environments)
    output.mkdir(parents=True,exist_ok=True);report.parent.mkdir(parents=True,exist_ok=True)
    paths={}
    for name,frame in tables.items():
        path=output/f"{name}.csv";frame.to_csv(path,index=False);paths[name]=str(path)
    report.write_text(report_text);paths["report"]=str(report)
    manifest=dict(schema_version=1,campaign=campaign,n_unique_runs=len(records),n_incomplete_records=loaded["n_incomplete_records"],
        input_archives=loaded["input_archives"],prepared_manifest_sha256=loaded["prepared_manifest_sha256"],
        protocol_sha256=_hash_file(protocol_path),aggregator_source_sha256=_hash_file(__file__),
        artifact_sha256=loaded.get("artifact_sha256",{}),device_label=label,
        execution_environments=loaded.get("execution_environments",{}),
        environment_identity_fields="device,numpy,torch,scikit-learn,xgboost; CUDA additionally cuda,cudnn,gpu_name; CPU ignores unused GPU fields",
        bootstrap_replicates=replicates,random_seed=random_seed,
        headline_checkpoint="best",secondary_checkpoint="last",cuda_executed_here=False,
        stopping_hypothesis_protocol=stopping["id"],seed_block_protocol=blocks["id"],
        outputs={name:dict(path=path,sha256=_hash_file(path)) for name,path in paths.items()})
    if reference_cpu and records:
        manifest["reference_cpu_sha256"]=_hash_file(reference_cpu)
    manifest_path=output/"aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    paths["manifest"]=str(manifest_path)
    return paths
