# Feature-set post-hoc diagnostic

Exploratory comparison for hypothesis generation. All model and preprocessing
fits used train rows, and scores used the saved validation partition. The
RF validation study has five seeds (42–46, n=5, indicative only) in items 1 and
3 and three seeds (42–44) in item 5. Small validation classes, including
`nmap_fin_scan` and `metasploit_brute_force_ssh`, make macro-F1 noisy.

## Run record

- Started at `2026-11-28T06:27:43+00:00`. Diagnostic wall time was
  438.53 s using the CPU on `Linux-6.1.0-56-amd64-x86_64-with-glibc2.36`.
- Python 3.11.2, numpy 1.26.4, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.6.1, torch 2.6.0+cpu. RF used `n_jobs=2`.
- Canonical data fingerprint `dfba90bd116f9288bc72a837160516984ba773b11904e98561c80cda7dedcf5c`.
- CSV SHA-256 `7572b790d53e08babdaf9b08dabaaf20dcc2282e8f1926fe7e9470ae1e7286bb`.
- Split archive SHA-256 `7023c1dad23bd4801e9b19e6214fc805c2317557dbe49e57f511cdb06ad6d1c0`.
- 86,181 train rows and 18,468 validation rows overall.
- Candidate accounting is consistent (94 - 1 constant = 93).
  `FlowPreprocessor.encode` supplies the unscaled `float64` candidates.

| Pair class | Train support | Validation support |
| --- | ---: | ---: |
| `nmap_os_detection` | 1400 | 300 |
| `nmap_xmas_tree_scan` | 1407 | 301 |
| **Total** | **2807** | **601** |

An overlap counts a validation flow once when its entire vector equals any
opposite-class train vector. Observed counts were F40 90/601, F93 0/601. AUC uses
Xmas as positive and reports `max(AUC, 1 - AUC)` to allow either threshold
direction. Both RF-importance lists use one train fit with seed 42 on F93.
The multiclass ranking fit also supplies the matching F93 seed-42 score.
Item 5 evaluates both sets of importance prefixes on the 12-class task. The
rankings used for item 5 are unfiltered.

## My number vs yours

The reference values come from the user's exploratory run with NumPy 2.5.3,
pandas 3.0.5 and scikit-learn 1.9.1. Deltas are yours minus mine. F1 summaries
use the mean and sample SD across the listed seeds.

| Item | Metric | My number | Yours | Delta | Tolerance | n seeds | Sample SD | Seed 42 | Check |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Pair F40 validation macro-F1 | 0.4670 | 0.4671 | +0.0001 | ±0.01 | 5 | 0.0045 | 0.4742 | Within |
| 1 | Pair F40 opposite-class vector overlap | 15.00% | 14.98% | -0.02 pp | ±1 pp | — | — | — | Within |
| 1 | Pair F93 validation macro-F1 | 1.0000 | 1.0000 | +0.0000 | ±0.01 | 5 | 0.0000 | 1.0000 | Within |
| 1 | Pair F93 opposite-class vector overlap | 0.00% | 0.00% | +0.00 pp | ±1 pp | — | — | — | Within |
| 2 | Train pair AUC, `fwd_psh_flag_count` | 1.0000 | 0.9996 | -0.0004 | ±0.01 | — | — | — | Within |
| 2 | Binary-MI rank, `fwd_psh_flag_count` | 60 | 60 | +0 | ±3 | — | — | — | Within |
| 2 | Train pair AUC, `flow_fin_flag_count` | 1.0000 | 0.9996 | -0.0004 | ±0.01 | — | — | — | Within |
| 2 | Binary-MI rank, `flow_fin_flag_count` | 71 | 71 | +0 | ±3 | — | — | — | Within |
| 2 | Train pair AUC, `fwd_urg_flag_count` | 0.9990 | 0.9993 | +0.0003 | ±0.01 | — | — | — | Within |
| 2 | Binary-MI rank, `fwd_urg_flag_count` | 82 | 82 | +0 | ±3 | — | — | — | Within |
| 2 | Best train pair AUC in F40 (`fwd_pkts_per_sec`) | 0.5820 | 0.5817 | -0.0003 | ±0.01 | — | — | — | Within |
| 3 | 12-class F40 validation macro-F1 | 0.8725 | 0.8819 | +0.0094 | ±0.01 | 5 | 0.0037 | 0.8850 | Within |
| 3 | 12-class F40 macro-F1 sample SD | 0.0031 | 0.0037 | +0.0006 | ±0.01 | — | — | — | Within |
| 3 | 12-class F93 validation macro-F1 | 0.9755 | 0.9843 | +0.0088 | ±0.01 | 5 | 0.0007 | 0.9845 | Within |
| 3 | 12-class F93 macro-F1 sample SD | 0.0002 | 0.0007 | +0.0005 | ±0.01 | — | — | — | Within |
| 3 | `nmap_os_detection` F40 F1 (300 val) | 0.4700 | 0.4607 | -0.0093 | ±0.01 | 5 | 0.0031 | 0.4595 | Within |
| 3 | `nmap_os_detection` F93 F1 (300 val) | 1.0000 | 1.0000 | +0.0000 | ±0.01 | 5 | 0.0000 | 1.0000 | Within |
| 3 | `nmap_xmas_tree_scan` F40 F1 (301 val) | 0.4670 | 0.4703 | +0.0033 | ±0.01 | 5 | 0.0039 | 0.4754 | Within |
| 3 | `nmap_xmas_tree_scan` F93 F1 (301 val) | 1.0000 | 1.0000 | +0.0000 | ±0.01 | 5 | 0.0000 | 1.0000 | Within |
| 3 | `ddos_slowloris` F40 F1 (80 val) | 0.9050 | 0.8963 | -0.0087 | ±0.01 | 5 | 0.0040 | 0.9007 | Within |
| 3 | `ddos_slowloris` F93 F1 (80 val) | 0.9730 | 0.9691 | -0.0039 | ±0.01 | 5 | 0.0030 | 0.9677 | Within |
| 3 | `wipro_bulb` F40 F1 (38 val) | 0.9100 | 0.9130 | +0.0030 | ±0.01 | 5 | 0.0081 | 0.9189 | Within |
| 3 | `wipro_bulb` F93 F1 (38 val) | 0.9460 | 0.9431 | -0.0029 | ±0.01 | 5 | 0.0065 | 0.9459 | Within |
| 3 | `metasploit_brute_force_ssh` F40 F1 (6 val) | 0.8840 | 0.9231 | +0.0391 | ±0.01 | 5 | 0.0000 | 0.9231 | **Outside** |
| 3 | `metasploit_brute_force_ssh` F93 F1 (6 val) | 0.9230 | 0.9231 | +0.0001 | ±0.01 | 5 | 0.0000 | 0.9231 | Within |
| 4 | 12-class MI rank, `fwd_psh_flag_count` | 57 | 57 | +0 | ±3 | — | — | — | Within |
| 4 | 12-class MI rank, `flow_fin_flag_count` | 67 | 67 | +0 | ±3 | — | — | — | Within |
| 4 | 12-class MI rank, `fwd_urg_flag_count` | 75 | 75 | +0 | ±3 | — | — | — | Within |
| 4 | Shared features in binary and 12-class MI top-40 | 31 | 31 | +0 | ±2 | — | — | — | Within |
| 5 | multiclass RF-importance rank, `fwd_psh_flag_count` | 5 | 4 | -1 | ±3 | — | — | — | Within |
| 5 | multiclass RF-importance rank, `flow_fin_flag_count` | 8 | 9 | +1 | ±3 | — | — | — | Within |
| 5 | multiclass RF-importance rank, `fwd_urg_flag_count` | 2 | 1 | -1 | ±3 | — | — | — | Within |
| 5 | multiclass RF top-40, 12-class macro-F1 (3/3 flags) | 0.9776 | 0.9845 | +0.0069 | ±0.01 | 3 | 0.0006 | 0.9851 | Within |
| 5 | multiclass RF top-20, 12-class macro-F1 (3/3 flags) | 0.9673 | 0.9791 | +0.0118 | ±0.01 | 3 | 0.0005 | 0.9789 | **Outside** |
| 5 | multiclass RF top-8, 12-class macro-F1 (2/3 flags) | 0.9573 | 0.9712 | +0.0139 | ±0.01 | 3 | 0.0028 | 0.9728 | **Outside** |
| 5 | binary RF-importance rank, `fwd_psh_flag_count` | 66 | 59 | -7 | ±3 | — | — | — | **Outside** |
| 5 | binary RF-importance rank, `flow_fin_flag_count` | 78 | 74 | -4 | ±3 | — | — | — | **Outside** |
| 5 | binary RF-importance rank, `fwd_urg_flag_count` | 91 | 92 | +1 | ±3 | — | — | — | Within |
| 5 | binary RF top-40, 12-class macro-F1 (0/3 flags) | 0.9381 | 0.9410 | +0.0029 | ±0.01 | 3 | 0.0001 | 0.9410 | Within |
| 5 | binary RF top-20, 12-class macro-F1 (0/3 flags) | 0.5940 | 0.5927 | -0.0013 | ±0.01 | 3 | 0.0009 | 0.5917 | Within |
| 5 | binary RF top-8, 12-class macro-F1 (0/3 flags) | 0.5748 | 0.5739 | -0.0009 | ±0.01 | 3 | 0.0012 | 0.5754 | Within |
| 5 | Flags in the 12-class RF top-8 | 3 | 2 | -1 | ±0 | — | — | — | **Outside** |

## Qualitative checks

- Holds — The three flags have train pair AUC ≥ 0.989 and exceed the best F40 AUC.
- Holds — F93 raises pair and 12-class macro-F1 by more than 0.01 versus F40.
- Holds — The 12-class RF top-40 contains all three flags and the binary RF top-40 contains zero.

Proceed to Phase 2. F93 adds many columns alongside the three flags, so this
comparison measures the joint feature-set difference. Perfect separation by
the flags, if observed with F43, may be specific to this dataset and its
traffic generator.

## Commands

Run from the repository root with the pinned environment. The fitter emits
the [measurement record](measurements.json) to stdout and progress to stderr.
The renderer reads only that record.

```bash
diag_dir=$(mktemp -d)
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -B -u -m scripts.diagnostics.feature_sets \
  --data data/RT_IOT2022.csv --split-indices artifacts/split_indices.npz \
  --cache-manifest artifacts/cache_manifest.json --jobs 2 \
  > "$diag_dir/measurements.json" 2> "$diag_dir/progress.log"
.venv/bin/python -B -m scripts.diagnostics.render_feature_sets \
  "$diag_dir/measurements.json" > "$diag_dir/REPORT.md"
```
