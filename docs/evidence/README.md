# Verify the canonical CPU evidence

The checksum list preserves the 806 canonical inventory entries from the
[PA-20261123-2 input receipt](../../archive/gpu_v2_before_aggregation/20261123_PA-20261123-2/INPUT_SHA256SUMS),
at their original repository-relative paths. The inventory is a fixed list.
The later Colab runner falls outside its scope.

From the repository root, verify the files supplied by Git:

```bash
sha256sum -c --ignore-missing docs/evidence/cpu_v1_SHA256SUMS
```

Expect exit status 0 and **775 OK** lines on a fresh clone. To check the count
and identify the skipped files, use the commands below. A working copy with
all local evidence can run `sha256sum -c docs/evidence/cpu_v1_SHA256SUMS` and
expect **806 OK** lines.

```bash
set -o pipefail
sha256sum -c --ignore-missing docs/evidence/cpu_v1_SHA256SUMS | grep -c ': OK$'
while IFS= read -r row; do
  path=${row#*  }
  if [ ! -f "$path" ]; then printf '%s\n' "$path"; fi
done < docs/evidence/cpu_v1_SHA256SUMS
```

## Mapping hash

The published mapping SHA-256 is
`6369242b08ea89b9cfb83da8a7b6ddf9397539dca849a52c5b45a4916cea4610`.
Reproduce it by building a JSON object keyed by relative path, sorting keys,
indenting by two spaces and adding one trailing newline before hashing the
UTF-8 bytes. The resulting hash binds the entire recorded inventory. The
file checks above verify the actual bytes available on disk.

```bash
python3.11 - <<'PY'
import hashlib
import json
from pathlib import Path

entries = Path("docs/evidence/cpu_v1_SHA256SUMS").read_text(encoding="utf-8").splitlines()
expected = {}
for entry in entries:
    digest, path = entry.split("  ", 1)
    expected[path] = digest
assert len(entries) == len(expected) == 806
payload = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
```

## Local-only files

The 31 paths below are absent from a fresh clone. They include 20 full-mode
tree checkpoints. The other four ignored tree checkpoints are fast-mode
artifacts outside this inventory.

```
artifacts/stats_cpu_v1/review/source_review.txt
artifacts/stats_cpu_v1/smoke/controller.lock
artifacts/stats_cpu_v1/smoke/first_attempt/gpu_v2_smoke.zip
artifacts/stats_cpu_v1/smoke/first_attempt/gpu_v2_smoke/campaign.lock
artifacts/stats_cpu_v1/smoke/first_attempt/gpu_v2_smoke/cpu/ca5364c9584721e42f13407f86027e260ecd4b91ed960a1b5c367c9c8469d17a/epoch_checkpoint.pt
artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke.zip
artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/campaign.lock
artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/36a31c291e1cdb34c4ac0bad7830ad8324f8eb81301a64ce5fc85a3584592081/epoch_checkpoint.pt
artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/ca5364c9584721e42f13407f86027e260ecd4b91ed960a1b5c367c9c8469d17a/epoch_checkpoint.pt
results/full/.locks/benchmark_recovery.lock
results/full/.locks/pipeline.lock
results/full/binary/random_forest/42/model.pkl
results/full/binary/random_forest/43/model.pkl
results/full/binary/random_forest/44/model.pkl
results/full/binary/random_forest/45/model.pkl
results/full/binary/random_forest/46/model.pkl
results/full/binary/xgboost/42/model.pkl
results/full/binary/xgboost/43/model.pkl
results/full/binary/xgboost/44/model.pkl
results/full/binary/xgboost/45/model.pkl
results/full/binary/xgboost/46/model.pkl
results/full/multiclass/random_forest/42/model.pkl
results/full/multiclass/random_forest/43/model.pkl
results/full/multiclass/random_forest/44/model.pkl
results/full/multiclass/random_forest/45/model.pkl
results/full/multiclass/random_forest/46/model.pkl
results/full/multiclass/xgboost/42/model.pkl
results/full/multiclass/xgboost/43/model.pkl
results/full/multiclass/xgboost/44/model.pkl
results/full/multiclass/xgboost/45/model.pkl
results/full/multiclass/xgboost/46/model.pkl
```
