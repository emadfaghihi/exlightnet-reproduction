# ExLightNet reproduction report

Generated from saved evidence: 2026-11-17T16:49:36.858856+03:30

The supplied 31-page PDF was fully extracted with `pdftotext -layout` and read before implementation. The supplied brief is authoritative where it conflicts with the paper. All measured results below come from code. Quoted reference metrics appear only in `src/exlightnet/paper.py::PAPER_REPORTED` and comparison tables.

## Execution status

Pytest verification from artifacts/stats_cpu_v1/verification/final/pytest.xml. Warning logs remain beside their original verification records.

| Tests | Failures | Errors | Skipped |
| --- | --- | --- | --- |
| 199 | 0 | 0 | 0 |

Historical notebook verification (wall-clock time may include the original training; this is preserved history):

| Mode | Seconds | Executed cells | Error outputs | Caches present at start |
| --- | --- | --- | --- | --- |
| fast | 695.022 | 14 | 0 | initial live run |
| fast | 9.408 | 14 | 0 | True |
| fast | 9.029 | 14 | 0 | True |
| full | 3.081 | 14 | 0 | True |

Current notebook verification from a new evidence path:

| Source | Mode | Seconds | Executed cells | Error outputs |
| --- | --- | --- | --- | --- |
| artifacts/stats_cpu_v1/verification/final/notebook_restart/notebook_execution.json | full | 3.4318543539993698 | 16 | 0 |

Effective current status (raw historical status is preserved):

| Field | Value |
| --- | --- |
| effective_status | complete |
| raw_pipeline_status | complete_with_stopped_branches |
| active_failure_count | 0 |
| archived_idle_failure_count | 2 |
| latest_benchmark_campaign | 20261117T052645467310-1050022 |
| latest_benchmark_measured_models | 95 |
| explanation | The raw pipeline flag preserves its initial CPU-idle timeout. Archived idle failures remain intact; a later successful benchmark recovery completed all current measurements with no active failures. |
| archived_idle_failures | campaign: 20261116T151119336040-1010228; source: results/full/benchmarks/20261116T151119336040-1010228/manifest.json; branch: benchmark/binary/exlightnet/42; error_type: IdleCPUUnavailable; reason: CPU did not remain idle (<= 5% busy) for 3 windows within 300.0s; stage: benchmark_only, campaign: 20261117T051732374195-1034809; source: results/full/benchmarks/20261117T051732374195-1034809/manifest.json; branch: benchmark/binary/exlightnet/42; error_type: IdleCPUUnavailable; reason: CPU did not remain idle (<= 5% busy) for 3 windows within 300.0s; stage: benchmark_only |

### Cached-statistics and hosted-runner verification

Source: artifacts/stats_cpu_v1/verification/final/gates.json. These gates verify cached analysis, code and the local CPU smoke; CUDA studies remain NOT RUN.

| Stage | Gate | Status | Evidence |
| --- | --- | --- | --- |
| A | Immutable evidence | PASS | files_checked: 916; changed:  |
| A | Paired t/CI agrees with SciPy | PASS | artifacts/stats_cpu_v1/verification/final/pytest.xml |
| A | No active legacy SD flags | PASS | violations: ; excluded: results/, archive/, authoritative brief/instruction docs |
| A | Mode captions and status recovery | PASS | artifacts/stats_cpu_v1/verification/final/pytest.xml |
| A | Generated statistical values trace to checksummed files | PASS | tables_checked: model_summary, paired_tests, flow_bootstrap, paired_flow_bootstrap, per_class_f1, sensitivity, training_dynamics, correlations, binary_caps, per_seed_macro_f1; manifest: artifacts/stats_cpu_v1/statistics_manifest.json; method: source/table SHA and rendering tests; no freehand new metric values |
| A | Executed cached notebook | PASS | artifacts/stats_cpu_v1/verification/final/notebook_restart/notebook_execution.json |
| A | Full pytest suite | PASS | tests: 199; xml: artifacts/stats_cpu_v1/verification/final/pytest.xml |
| B | Canonical CPU default compatibility | PASS | method: read-only validated original CPU cache hit; no retraining or inference; fingerprint: b49fbe2ba153504e2811f7abf991a5d71f5c8e1d682c5af7ba660f3291bf567e; macro_f1: 0.8354536030948337; retrained: NOT RUN: prohibited by local experiment budget |
| B | Sampler-frequency equivalence | PASS | artifacts/stats_cpu_v1/verification/final/pytest.xml |
| B | Prepared split42 matches canonical fingerprint | PASS | fingerprint: dfba90bd116f9288bc72a837160516984ba773b11904e98561c80cda7dedcf5c; evidence: artifacts/stats_cpu_v1/prepared_export_checks.json |
| B | CPU smoke, ZIP restore, idempotent resume and aggregation | PASS | seconds: 10.875827761999972; evidence: artifacts/stats_cpu_v1/smoke/control.json; aggregate: artifacts/stats_gpu_v2_smoke_verified/aggregation_manifest.json |
| B | Mixed campaigns rejected; within-campaign device exception explicit | PASS | artifacts/stats_cpu_v1/verification/final/pytest.xml |
| B | Real studies remain PENDING; protocol verbatim | PASS | REPORT_stats_v2.md |
| B | Full pytest suite | PASS | tests: 199; xml: artifacts/stats_cpu_v1/verification/final/pytest.xml |

Synthetic unit verification is separate from the real-data smoke; earlier focused runs are preserved in their own records. Unit-test seconds are pytest case wall time; smoke seconds are supervisor end-to-end time.

| Scope | Cases/runs | Fits | Epochs | Seconds |
| --- | --- | --- | --- | --- |
| Synthetic tests in the final full suite | 4 | 5 | 7 | 1.3810000000000002 |
| Real-data CPU smoke with ZIP restore and cache restart | 2 | 2 | 4 | 10.875827761999972 |

## Fast demonstration (historical)

**fast:** seeds [42]; 2.41 seconds for the recorded invocation. Full five-seed headline matrix complete: False. Stopped branches: 0.

### Gates

| Gate | Task/seed | Status | Actual evidence |
| --- | --- | --- | --- |
| D1 |   | PASS | rows: 123117; counts: mqtt_publish: 4146; thing_speak: 8108; wipro_bulb: 253; arp_poisioning: 7750; ddos_slowloris: 534; dos_syn_hping: 94659; metasploit_brute_force_ssh: 37; nmap_fin_scan: 28; nmap_os_detection: 2000; nmap_tcp_scan: 1002; nmap_udp_scan: 2590; nmap_xmas_tree_scan: 2010; normal: 12507; attack: 110610 |
| D2 |   | PASS | sizes: train: 86181; val: 18468; test: 18468; supports: train: dos_syn_hping: 66261; arp_poisioning: 5425; nmap_xmas_tree_scan: 1407; nmap_udp_scan: 1813; nmap_os_detection: 1400; mqtt_publish: 2902; nmap_fin_scan: 20; nmap_tcp_scan: 701; thing_speak: 5675; ddos_slowloris: 374; wipro_bulb: 177; metasploit_brute_force_ssh: 26; val: dos_syn_hping: 14199; ddos_slowloris: 80; nmap_os_detection: 300; arp_poisioning: 1163; nmap_udp_scan: 388; thing_speak: 1217; mqtt_publish: 622; nmap_xmas_tree_scan: 301; nmap_tcp_scan: 150; wipro_bulb: 38; nmap_fin_scan: 4; metasploit_brute_force_ssh: 6; test: dos_syn_hping: 14199; arp_poisioning: 1162; thing_speak: 1216; ddos_slowloris: 80; mqtt_publish: 622; nmap_os_detection: 300; nmap_tcp_scan: 151; nmap_udp_scan: 389; nmap_xmas_tree_scan: 302; wipro_bulb: 38; metasploit_brute_force_ssh: 5; nmap_fin_scan: 4 |
| P1 |   | REPORT ONLY | report_only: True; overlap_count: 40; budget: 40; spearman_shared_ranks: 0.9994371482176362; candidate_count_before_pruning: 94; candidate_count_after_pruning: 93; dropped_constant: bwd_urg_flag_count; mi_settings: target: binary; n_neighbors: 3; random_state: 42; discrete_features: one-hot only; ties: stable candidate order |
| M1 |   | PASS | multiclass: 38328; binary: 35438; no_se: 23052 |
| T1 |   | PASS | rarest_train_count: 20; tempered_multiplicity: 29.809225890291543; balanced_multiplicity: 359.08750000000003 |
| Q1 | binary 42 | PASS | fp32_macro_f1: 0.9626731969387561; w8_macro_f1: 0.9623857384118137; delta: -0.0002874585269423413; fp32_bytes: 152312; max_abs_weight_error: 0.0021119117736816406; int8_bytes: 53468; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights |
| Q1 | multiclass 42 | PASS | fp32_macro_f1: 0.5836595981643584; w8_macro_f1: 0.5817009710183444; delta: -0.0019586271460140248; fp32_bytes: 163832; max_abs_weight_error: 0.002306520938873291; int8_bytes: 56234; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights |

### Measured macro-F1 and quoted paper comparison

Standard deviation is the sample SD (ddof=1); a single seed has no estimated SD. This cached demonstration uses reduced training and SHAP.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

| Task | Model | Seeds | Our mean | Our SD | Our seed 42 | Paper (quoted) |
| --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 1 | 0.945036 | — | 0.945036 | 0.9354 |
| binary | exlightnet | 1 | 0.962673 | — | 0.962673 | 0.986 |
| binary | exlightnet_w8 | 1 | 0.962386 | — | 0.962386 | 0.986 |
| binary | mlp | 1 | 0.940930 | — | 0.940930 | 0.98 |
| binary | random_forest | 1 | 0.993791 | — | 0.993791 | 0.9934 |
| binary | std_cnn | 1 | 0.863285 | — | 0.863285 | 0.982 |
| binary | xgboost | 1 | 0.993371 | — | 0.993371 | 0.9957 |
| multiclass | balanced | 1 | 0.477435 | — | 0.477435 | 0.7807 |
| multiclass | bilstm | 1 | 0.468787 | — | 0.468787 | 0.638 |
| multiclass | exlightnet | 1 | 0.583660 | — | 0.583660 | 0.8621 |
| multiclass | exlightnet_w8 | 1 | 0.581701 | — | 0.581701 | 0.8555 |
| multiclass | mlp | 1 | 0.584393 | — | 0.584393 | 0.7933 |
| multiclass | no_sampler | 1 | 0.651598 | — | 0.651598 | 0.8697 |
| multiclass | no_se | 1 | 0.609188 | — | 0.609188 | 0.7963 |
| multiclass | random_forest | 1 | 0.859192 | — | 0.859192 | 0.86 |
| multiclass | standard_conv | 1 | 0.616110 | — | 0.616110 | 0.8259 |
| multiclass | std_cnn | 1 | 0.399955 | — | 0.399955 | 0.8529 |
| multiclass | top20 | 1 | 0.405142 | — | 0.405142 | 0.6048 |
| multiclass | xgboost | 1 | 0.832664 | — | 0.832664 | 0.8632 |

All accuracy, macro precision/recall, weighted F1, macro F1, AUC, per-class support, confusion matrices, parameter counts, sizes and latency observations are in the per-run JSON files and `summary.csv`. Binary JSON additionally records attack recall, misses, false positives and FPR at the fixed argmax threshold.

### Measured size and latency (seed 42)

| Task | Model | Parameters | Serialized bytes | Mean ms | Median ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| binary | exlightnet | 35438 | 152312 | 0.6578377910000001 | 0.6534945 | 0.67747925 |
| binary | mlp | 13634 | 56884 | 0.047644432 | 0.0474425 | 0.0485955 |
| binary | bilstm | 9090 | 39452 | 0.33748025099999995 | 0.33470350000000004 | 0.3515182 |
| binary | std_cnn | 124482 | 506986 | 0.35819965200000004 | 0.35612699999999997 | 0.3680619 |
| binary | random_forest | — | 9659792 | 14.686626285000001 | 14.6771095 | 14.979970199999999 |
| binary | xgboost | — | 706309 | 0.429932266 | 0.425083 | 0.44145825 |
| multiclass | exlightnet | 38328 | 163832 | 0.671806534 | 0.6646810000000001 | 0.7070485499999999 |
| multiclass | mlp | 14284 | 59444 | 0.049293145 | 0.04902 | 0.0505233 |
| multiclass | bilstm | 9740 | 42012 | 0.334713594 | 0.33322850000000004 | 0.34463055000000004 |
| multiclass | std_cnn | 127052 | 517226 | 0.36323393800000003 | 0.36026 | 0.3770669 |
| multiclass | random_forest | — | 121518022 | 14.861059532999999 | 14.837621 | 15.12261605 |
| multiclass | xgboost | — | 7767607 | 0.7604472640000001 | 0.686318 | 1.0806490499999999 |
| multiclass | no_se | 23052 | 100576 | 0.538472234 | 0.53563 | 0.57583375 |
| multiclass | standard_conv | 74760 | 308994 | 0.472354859 | 0.471004 | 0.4812427 |
| multiclass | balanced | 38328 | 163832 | 0.679299397 | 0.6699965 | 0.7277440000000001 |
| multiclass | no_sampler | 38328 | 163832 | 0.672657393 | 0.6655795 | 0.7073035 |
| multiclass | top20 | 38328 | 163832 | 0.638441413 | 0.634451 | 0.6571114 |
| binary | exlightnet_w8 | 35438 | 53468 | 0.6739896270000001 | 0.6682375 | 0.7016203 |
| multiclass | exlightnet_w8 | 38328 | 56234 | 0.669587671 | 0.6615405000000001 | 0.7072548499999999 |

### INT8 size breakdown

Weights-only means the raw INT8 Conv1d/Linear tensor payload. The full NPZ also contains per-channel FP32 scales, FP32 BN/bias state, metadata, NPY headers and the ZIP container; only the full-artifact column is an on-disk file size.

| Task | Seed | FP32 file bytes | INT8 weights-only bytes | FP32 scale bytes | FP32 rest bytes | Full INT8 artifact bytes |
| --- | --- | --- | --- | --- | --- | --- |
| binary | 42 | 152312 | 34560 | 2936 | 5828 | 53468 |
| multiclass | 42 | 163832 | 37440 | 2976 | 5868 | 56234 |

Seed-42 component accounting (compressed member sizes include each NPY header; the full-artifact total also includes the ZIP container):

| Task | Component | Raw array payload bytes | Compressed members/container bytes |
| --- | --- | --- | --- |
| binary | INT8 weights | 34560 | 34742 |
| binary | FP32 scales | 2936 | 3576 |
| binary | FP32 rest | 5828 | 7285 |
| binary | Metadata | 10468 | 1843 |
| binary | ZIP container | — | 6022 |
| binary | Full artifact | 53792 | 53468 |
| multiclass | INT8 weights | 37440 | 37409 |
| multiclass | FP32 scales | 2976 | 3632 |
| multiclass | FP32 rest | 5868 | 7328 |
| multiclass | Metadata | 10472 | 1843 |
| multiclass | ZIP container | — | 6022 |
| multiclass | Full artifact | 56756 | 56234 |

### Alpha selection

Selection uses seed 42 validation macro-F1 only, separately per task, then freezes the selected mode for every deep model, seed and ablation. The small sweep is intentionally capped and does not optimize paper agreement.

| Task | Chosen | Sweep epochs |
| --- | --- | --- |
| binary | none | 1 |
| multiclass | sqrt_inv_freq | 1 |

| Task | Alpha mode | Validation macro-F1 | Best epoch | Selected |
| --- | --- | --- | --- | --- |
| binary | inv_freq | 0.9100821611365608 | 1 | False |
| binary | sqrt_inv_freq | 0.9468740686113915 | 1 | False |
| binary | effective_num | 0.93559350737973 | 1 | False |
| binary | none | 0.9665092947966785 | 1 | True |
| multiclass | inv_freq | 0.5078781769707076 | 1 | False |
| multiclass | sqrt_inv_freq | 0.5819639263753635 | 1 | True |
| multiclass | effective_num | 0.5056573017473512 | 1 | False |
| multiclass | none | 0.5306680428252998 | 1 | False |

### Paired ablations

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

NOT RUN: multi-seed paired ablation statistics are unavailable for this evidence scope.

### Explainability evidence

Background and random test examples use a fixed seed. FP32 additionally includes the local examples listed below.

| Artifact | Method/status | Train backgrounds | Random test flows | Extra local flows | Max abs. additivity error | Mean abs. additivity error |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | GradientExplainer | 16 | 12 | 2 | 2.352273098904563 | 0.32402057832028397 |
| Dequantised INT8 | GradientExplainer | 16 | 12 | 0 | 2.362959878228474 | 0.3354329959236417 |

| Artifact | Fallback reason |
| --- | --- |
| FP32 | AssertionError: The SHAP explanations do not sum up to the model's output! This is either because of a rounding error or because an operator in your computation graph was not fully supported. If the sum difference of %f is significant compared to the scale of your model outputs, please post as a github issue, with a reproducible example so we can debug it. Used framework: pytorch - Max. diff: 1.507496914308831 - Tolerance: 0.01 |
| Dequantised INT8 | AssertionError: The SHAP explanations do not sum up to the model's output! This is either because of a rounding error or because an operator in your computation graph was not fully supported. If the sum difference of %f is significant compared to the scale of your model outputs, please post as a github issue, with a reproducible example so we can debug it. Used framework: pytorch - Max. diff: 0.7400688089746836 - Tolerance: 0.01 |

Deletion probabilities:

| Removed features | Mean original predicted-class probability |
| --- | --- |
| 0 | 0.832952082157135 |
| 10 | 0.6494689583778381 |
| 20 | 0.6027548909187317 |

| SHAP deletion-AUC | Random deletion-AUC | Random permutations |
| --- | --- | --- |
| 0.6035759449005127 | 0.6578243970870972 | 20 |

| Quantisation stability measure | Value |
| --- | --- |
| spearman_top15_fp32_feature_set | 0.9964285714285712 |
| top5_overlap | 4 |

Local examples (signed logit contributions, with correctness stated):

| True class | Predicted class | Correct | Probability | Test position |
| --- | --- | --- | --- | --- |
| dos_syn_hping | dos_syn_hping | True | 0.8430706858634949 | 0 |
| nmap_os_detection | nmap_xmas_tree_scan | False | 0.3322300910949707 | 44 |

| Example true class | Rank | Feature | Signed logit contribution | z-value |
| --- | --- | --- | --- | --- |
| dos_syn_hping | 1 | bwd_iat.avg | 1.9774124628549767 | -0.07581867277622223 |
| dos_syn_hping | 2 | bwd_iat.max | 1.170660817388596 | -0.0933314710855484 |
| dos_syn_hping | 3 | fwd_iat.avg | -0.9937715815889577 | -0.12744927406311035 |
| dos_syn_hping | 4 | flow_iat.avg | -0.6831166287963182 | -0.15764069557189941 |
| dos_syn_hping | 5 | flow_pkts_per_sec | 0.5854046593905473 | 1.874710202217102 |
| nmap_os_detection | 1 | flow_pkts_payload.avg | 1.4606639625940947 | -1.2944403886795044 |
| nmap_os_detection | 2 | flow_pkts_payload.std | 0.4748805698849553 | -1.0291452407836914 |
| nmap_os_detection | 3 | fwd_pkts_payload.avg | 0.40471937656610424 | -2.2081925868988037 |
| nmap_os_detection | 4 | fwd_pkts_payload.max | 0.37075497868490076 | -0.9954687356948853 |
| nmap_os_detection | 5 | flow_pkts_payload.max | 0.2862804773247851 | -0.6824842095375061 |

Attributions explain logits; deletion measures the probability of the original predicted class. Additivity residuals are measured without correction. Deletion curves need not be monotone; neither causality nor guaranteed faithfulness is assumed. W8 uses identical background/test examples and GradientExplainer randomness for a controlled ranking probe.

DeepExplainer failed its additivity check and the prescribed GradientExplainer fallback was used. The residuals above show that these reduced-sample attributions do not establish exact additive efficiency; treat these explanation figures as approximate diagnostics.

### Hardware and runtime

| Field | Value |
| --- | --- |
| platform | Linux-6.1.0-32-amd64-x86_64-with-glibc2.36 |
| cpu | Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz |
| python | 3.11.2 |
| versions.numpy | 1.26.4 |
| versions.pandas | 2.2.3 |
| versions.scipy | 1.14.1 |
| versions.torch | 2.6.0+cpu |
| versions.scikit-learn | 1.6.1 |
| versions.xgboost-cpu | 2.1.4 |
| versions.shap | 0.46.0 |
| versions.nbformat | 5.10.4 |
| versions.nbconvert | 7.16.6 |
| versions.pytest | 8.3.5 |
| cuda_used | False |
| torch_threads | 1 |

Latency uses one library thread, batch one, 100 warmups and 1000 timed inferences on training examples. Mean, median, p95 and peak RSS are saved per run. Inline measurements include preceding training in their process-lifetime RSS; isolated benchmark-only measurements include preceding benchmarks, with no training in that process. Neither is incremental model memory. No multi-core throughput extrapolation is reported. Raspberry Pi measurements require running `scripts/bench_pi.py` on the device; this host is not a Pi.

| Stage | Wall seconds | Processes per stage/worker | Threads per process | Started (UTC) | Finished (UTC) |
| --- | --- | --- | --- | --- | --- |
| Seed 42 worker | 2.183885428999929 | 1 | 1 | 2026-11-16T10:30:57.207205+00:00 | 2026-11-16T10:30:59.367668+00:00 |

Seed workers overlap in wall time, so their elapsed times must not be summed as end-to-end runtime. Aggregate-only publication reuses final probabilities and test receipts.

## Full CPU reproduction

**full:** seeds [42, 43, 44, 45, 46]; 2.67 seconds for the recorded invocation. Full five-seed headline matrix complete: True. Stopped branches: 0.

### Gates

| Gate | Task/seed | Status | Actual evidence |
| --- | --- | --- | --- |
| D1 |   | PASS | rows: 123117; counts: mqtt_publish: 4146; thing_speak: 8108; wipro_bulb: 253; arp_poisioning: 7750; ddos_slowloris: 534; dos_syn_hping: 94659; metasploit_brute_force_ssh: 37; nmap_fin_scan: 28; nmap_os_detection: 2000; nmap_tcp_scan: 1002; nmap_udp_scan: 2590; nmap_xmas_tree_scan: 2010; normal: 12507; attack: 110610 |
| D2 |   | PASS | sizes: train: 86181; val: 18468; test: 18468; supports: train: dos_syn_hping: 66261; arp_poisioning: 5425; nmap_xmas_tree_scan: 1407; nmap_udp_scan: 1813; nmap_os_detection: 1400; mqtt_publish: 2902; nmap_fin_scan: 20; nmap_tcp_scan: 701; thing_speak: 5675; ddos_slowloris: 374; wipro_bulb: 177; metasploit_brute_force_ssh: 26; val: dos_syn_hping: 14199; ddos_slowloris: 80; nmap_os_detection: 300; arp_poisioning: 1163; nmap_udp_scan: 388; thing_speak: 1217; mqtt_publish: 622; nmap_xmas_tree_scan: 301; nmap_tcp_scan: 150; wipro_bulb: 38; nmap_fin_scan: 4; metasploit_brute_force_ssh: 6; test: dos_syn_hping: 14199; arp_poisioning: 1162; thing_speak: 1216; ddos_slowloris: 80; mqtt_publish: 622; nmap_os_detection: 300; nmap_tcp_scan: 151; nmap_udp_scan: 389; nmap_xmas_tree_scan: 302; wipro_bulb: 38; metasploit_brute_force_ssh: 5; nmap_fin_scan: 4 |
| P1 |   | REPORT ONLY | report_only: True; overlap_count: 40; budget: 40; spearman_shared_ranks: 0.9994371482176362; candidate_count_before_pruning: 94; candidate_count_after_pruning: 93; dropped_constant: bwd_urg_flag_count; mi_settings: target: binary; n_neighbors: 3; random_state: 42; discrete_features: one-hot only; ties: stable candidate order |
| M1 |   | PASS | multiclass: 38328; binary: 35438; no_se: 23052 |
| T1 |   | PASS | rarest_train_count: 20; tempered_multiplicity: 29.809225890291543; balanced_multiplicity: 359.08750000000003 |
| Q1 | binary 42 | PASS | fp32_macro_f1: 0.9880544774817068; w8_macro_f1: 0.9877616747181965; delta: -0.00029280276351029144; fp32_bytes: 152312; max_abs_weight_error: 0.0022417306900024414; int8_bytes: 48877; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 34560; scales_fp32_bytes: 2936; rest_fp32_bytes: 5828; metadata_array_bytes: 10468; total_array_payload_bytes: 53792; compressed_members_bytes: weights: 30028; scales: 3620; rest: 7364; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 48877; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | binary 43 | PASS | fp32_macro_f1: 0.9872617039362985; w8_macro_f1: 0.9875520685889918; delta: 0.0002903646526932757; fp32_bytes: 152312; max_abs_weight_error: 0.002089887857437134; int8_bytes: 49694; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 34560; scales_fp32_bytes: 2936; rest_fp32_bytes: 5828; metadata_array_bytes: 10468; total_array_payload_bytes: 53792; compressed_members_bytes: weights: 30824; scales: 3643; rest: 7362; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 49694; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | binary 44 | PASS | fp32_macro_f1: 0.9857062511287118; w8_macro_f1: 0.9854153932381198; delta: -0.00029085789059202227; fp32_bytes: 152312; max_abs_weight_error: 0.002098366618156433; int8_bytes: 50446; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 34560; scales_fp32_bytes: 2936; rest_fp32_bytes: 5828; metadata_array_bytes: 10468; total_array_payload_bytes: 53792; compressed_members_bytes: weights: 31590; scales: 3638; rest: 7353; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 50446; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | binary 45 | PASS | fp32_macro_f1: 0.9870983603678489; w8_macro_f1: 0.9871105427338052; delta: 1.2182365956237007e-05; fp32_bytes: 152312; max_abs_weight_error: 0.002102404832839966; int8_bytes: 49403; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 34560; scales_fp32_bytes: 2936; rest_fp32_bytes: 5828; metadata_array_bytes: 10468; total_array_payload_bytes: 53792; compressed_members_bytes: weights: 30545; scales: 3627; rest: 7366; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 49403; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | binary 46 | PASS | fp32_macro_f1: 0.9872255182066656; w8_macro_f1: 0.9873710677838696; delta: 0.0001455495772040294; fp32_bytes: 152312; max_abs_weight_error: 0.0023192763328552246; int8_bytes: 49892; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 34560; scales_fp32_bytes: 2936; rest_fp32_bytes: 5828; metadata_array_bytes: 10468; total_array_payload_bytes: 53792; compressed_members_bytes: weights: 31048; scales: 3622; rest: 7357; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 49892; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | multiclass 42 | PASS | fp32_macro_f1: 0.824796327203707; w8_macro_f1: 0.8253071812641015; delta: 0.0005108540603945277; fp32_bytes: 163832; max_abs_weight_error: 0.0024102330207824707; int8_bytes: 55271; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 37440; scales_fp32_bytes: 2976; rest_fp32_bytes: 5868; metadata_array_bytes: 10472; total_array_payload_bytes: 56756; compressed_members_bytes: weights: 36417; scales: 3658; rest: 7331; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 55271; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | multiclass 43 | PASS | fp32_macro_f1: 0.8557912091161192; w8_macro_f1: 0.8516139231555822; delta: -0.004177285960537014; fp32_bytes: 163832; max_abs_weight_error: 0.0023038536310195923; int8_bytes: 53843; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 37440; scales_fp32_bytes: 2976; rest_fp32_bytes: 5868; metadata_array_bytes: 10472; total_array_payload_bytes: 56756; compressed_members_bytes: weights: 34968; scales: 3649; rest: 7361; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 53843; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | multiclass 44 | PASS | fp32_macro_f1: 0.8734192489816173; w8_macro_f1: 0.8693312968410636; delta: -0.004087952140553686; fp32_bytes: 163832; max_abs_weight_error: 0.0021494626998901367; int8_bytes: 52876; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 37440; scales_fp32_bytes: 2976; rest_fp32_bytes: 5868; metadata_array_bytes: 10472; total_array_payload_bytes: 56756; compressed_members_bytes: weights: 34013; scales: 3641; rest: 7357; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 52876; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | multiclass 45 | PASS | fp32_macro_f1: 0.8312737283238759; w8_macro_f1: 0.8375948287000966; delta: 0.00632110037622069; fp32_bytes: 163832; max_abs_weight_error: 0.002301514148712158; int8_bytes: 53206; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 37440; scales_fp32_bytes: 2976; rest_fp32_bytes: 5868; metadata_array_bytes: 10472; total_array_payload_bytes: 56756; compressed_members_bytes: weights: 34369; scales: 3641; rest: 7331; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 53206; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |
| Q1 | multiclass 46 | PASS | fp32_macro_f1: 0.847711240259875; w8_macro_f1: 0.8501020098708919; delta: 0.0023907696110169185; fp32_bytes: 163832; max_abs_weight_error: 0.0025502443313598633; int8_bytes: 53706; quantized_weight_tensors: 10; compute: FP32 with dequantized W8 weights; size_breakdown: weights_only_int8_bytes: 37440; scales_fp32_bytes: 2976; rest_fp32_bytes: 5868; metadata_array_bytes: 10472; total_array_payload_bytes: 56756; compressed_members_bytes: weights: 34840; scales: 3652; rest: 7349; metadata: 1843; zip_container_bytes: 6022; full_artifact_bytes: 53706; definition: weights-only is raw int8 tensor payload; full artifact includes scales, FP32 state, metadata, NPY headers and ZIP container |

### Measured macro-F1 and quoted paper comparison

n=5, indicative only. Standard deviation is the sample SD (ddof=1). Seed 42 is always shown separately; no best seed is selected.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

| Task | Model | Seeds | Our mean | Our SD | Our seed 42 | Paper (quoted) |
| --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 5 | 0.963204 | 0.008207 | 0.961016 | 0.9354 |
| binary | exlightnet | 5 | 0.987069 | 0.000850 | 0.988054 | 0.986 |
| binary | exlightnet_w8 | 5 | 0.987042 | 0.000940 | 0.987762 | 0.986 |
| binary | mlp | 5 | 0.986900 | 0.000596 | 0.987123 | 0.98 |
| binary | random_forest | 5 | 0.993733 | 0.000168 | 0.993791 | 0.9934 |
| binary | std_cnn | 5 | 0.979657 | 0.005668 | 0.977757 | 0.982 |
| binary | xgboost | 5 | 0.993371 | 0.000000 | 0.993371 | 0.9957 |
| multiclass | balanced | 5 | 0.806513 | 0.030542 | 0.829804 | 0.7807 |
| multiclass | bilstm | 5 | 0.728210 | 0.028676 | 0.770176 | 0.638 |
| multiclass | exlightnet | 5 | 0.846598 | 0.019463 | 0.824796 | 0.8621 |
| multiclass | exlightnet_w8 | 5 | 0.846790 | 0.016496 | 0.825307 | 0.8555 |
| multiclass | mlp | 5 | 0.835599 | 0.010353 | 0.835454 | 0.7933 |
| multiclass | no_sampler | 5 | 0.844640 | 0.043865 | 0.865035 | 0.8697 |
| multiclass | no_se | 5 | 0.827014 | 0.022346 | 0.821965 | 0.7963 |
| multiclass | random_forest | 5 | 0.859600 | 0.002888 | 0.859192 | 0.86 |
| multiclass | standard_conv | 5 | 0.861879 | 0.018604 | 0.832179 | 0.8259 |
| multiclass | std_cnn | 5 | 0.706276 | 0.100929 | 0.832407 | 0.8529 |
| multiclass | top20 | 5 | 0.577855 | 0.012690 | 0.579804 | 0.6048 |
| multiclass | xgboost | 5 | 0.832664 | 0.000000 | 0.832664 | 0.8632 |

All accuracy, macro precision/recall, weighted F1, macro F1, AUC, per-class support, confusion matrices, parameter counts, sizes and latency observations are in the per-run JSON files and `summary.csv`. Binary JSON additionally records attack recall, misses, false positives and FPR at the fixed argmax threshold.

### Measured size and latency (seed 42)

| Task | Model | Parameters | Serialized bytes | Mean ms | Median ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- |
| binary | exlightnet | 35438 | 152312 | 0.61745241 | 0.611172 | 0.6528393 |
| binary | mlp | 13634 | 56884 | 0.044940519 | 0.0440205 | 0.0458833 |
| binary | bilstm | 9090 | 39452 | 0.32031351900000005 | 0.317287 | 0.33722389999999997 |
| binary | std_cnn | 124482 | 506986 | 0.319258475 | 0.317932 | 0.33922830000000004 |
| binary | random_forest | — | 9659792 | 14.329610262 | 14.253516999999999 | 14.907427199999999 |
| binary | xgboost | — | 706309 | 0.404379716 | 0.39794700000000005 | 0.44540135 |
| multiclass | exlightnet | 38328 | 163832 | 0.6296687059999999 | 0.615424 | 0.6816073999999999 |
| multiclass | mlp | 14284 | 59444 | 0.045479072 | 0.044627 | 0.047012899999999996 |
| multiclass | bilstm | 9740 | 42012 | 0.324458716 | 0.321301 | 0.3527728 |
| multiclass | std_cnn | 127052 | 517226 | 0.316068425 | 0.31089049999999996 | 0.33879145 |
| multiclass | random_forest | — | 121518022 | 14.049123725 | 13.971910000000001 | 14.6569456 |
| multiclass | xgboost | — | 7767607 | 0.609227513 | 0.582931 | 0.7949391499999999 |
| multiclass | no_se | 23052 | 100576 | 0.470730644 | 0.46772400000000003 | 0.50984235 |
| multiclass | standard_conv | 74760 | 308994 | 0.43575864799999997 | 0.43154349999999997 | 0.4606528999999999 |
| multiclass | balanced | 38328 | 163832 | 0.612473156 | 0.6121295 | 0.6329319 |
| multiclass | no_sampler | 38328 | 163832 | 0.6094580639999999 | 0.602599 | 0.6474533499999999 |
| multiclass | top20 | 38328 | 163832 | 0.566922669 | 0.564907 | 0.6036926 |
| binary | exlightnet_w8 | 35438 | 48877 | 0.5921434260000001 | 0.5924935 | 0.6163881 |
| multiclass | exlightnet_w8 | 38328 | 55271 | 0.5937076530000001 | 0.5921345 | 0.63052175 |

### INT8 size breakdown

Weights-only means the raw INT8 Conv1d/Linear tensor payload. The full NPZ also contains per-channel FP32 scales, FP32 BN/bias state, metadata, NPY headers and the ZIP container; only the full-artifact column is an on-disk file size.

| Task | Seed | FP32 file bytes | INT8 weights-only bytes | FP32 scale bytes | FP32 rest bytes | Full INT8 artifact bytes |
| --- | --- | --- | --- | --- | --- | --- |
| binary | 42 | 152312 | 34560 | 2936 | 5828 | 48877 |
| binary | 43 | 152312 | 34560 | 2936 | 5828 | 49694 |
| binary | 44 | 152312 | 34560 | 2936 | 5828 | 50446 |
| binary | 45 | 152312 | 34560 | 2936 | 5828 | 49403 |
| binary | 46 | 152312 | 34560 | 2936 | 5828 | 49892 |
| multiclass | 42 | 163832 | 37440 | 2976 | 5868 | 55271 |
| multiclass | 43 | 163832 | 37440 | 2976 | 5868 | 53843 |
| multiclass | 44 | 163832 | 37440 | 2976 | 5868 | 52876 |
| multiclass | 45 | 163832 | 37440 | 2976 | 5868 | 53206 |
| multiclass | 46 | 163832 | 37440 | 2976 | 5868 | 53706 |

Seed-42 component accounting (compressed member sizes include each NPY header; the full-artifact total also includes the ZIP container):

| Task | Component | Raw array payload bytes | Compressed members/container bytes |
| --- | --- | --- | --- |
| binary | INT8 weights | 34560 | 30028 |
| binary | FP32 scales | 2936 | 3620 |
| binary | FP32 rest | 5828 | 7364 |
| binary | Metadata | 10468 | 1843 |
| binary | ZIP container | — | 6022 |
| binary | Full artifact | 53792 | 48877 |
| multiclass | INT8 weights | 37440 | 36417 |
| multiclass | FP32 scales | 2976 | 3658 |
| multiclass | FP32 rest | 5868 | 7331 |
| multiclass | Metadata | 10472 | 1843 |
| multiclass | ZIP container | — | 6022 |
| multiclass | Full artifact | 56756 | 55271 |

### Alpha selection

Selection uses seed 42 validation macro-F1 only, separately per task, then freezes the selected mode for every deep model, seed and ablation. The small sweep is intentionally capped and does not optimize paper agreement.

| Task | Chosen | Sweep epochs |
| --- | --- | --- |
| binary | effective_num | 8 |
| multiclass | none | 8 |

| Task | Alpha mode | Validation macro-F1 | Best epoch | Selected |
| --- | --- | --- | --- | --- |
| binary | inv_freq | 0.9421064972406856 | 7 | False |
| binary | sqrt_inv_freq | 0.9803234429850337 | 7 | False |
| binary | effective_num | 0.9854911138018254 | 7 | True |
| binary | none | 0.984458429601228 | 7 | False |
| multiclass | inv_freq | 0.6626591114313519 | 8 | False |
| multiclass | sqrt_inv_freq | 0.7718128568257825 | 8 | False |
| multiclass | effective_num | 0.7166850759432505 | 8 | False |
| multiclass | none | 0.8107371672636426 | 6 | True |

### Paired ablations

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

n=5, indicative only. Source: artifacts/stats_cpu_v1/paired_tests.csv. Differences are ExLightNet minus the variant. Holm correction uses the full task family, including baselines and W8.

| reference | model | n_pairs | sample_label | mean_difference | difference_sd | ci_low | ci_high | p_value | cohens_dz | p_holm | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exlightnet | balanced | 5 | n=5, indicative only | 0.040085642 | 0.03972046 | -0.0092338231 | 0.089405107 | 0.087004625 | 1.0091938 | 0.60903237 | inconclusive |
| exlightnet | no_sampler | 5 | n=5, indicative only | 0.0019580848 | 0.053311308 | -0.064236647 | 0.068152817 | 0.93848957 | 0.036729258 | 1 | inconclusive |
| exlightnet | no_se | 5 | n=5, indicative only | 0.019583988 | 0.033123131 | -0.021543811 | 0.060711788 | 0.25667868 | 0.5912481 | 0.95031827 | inconclusive |
| exlightnet | standard_conv | 5 | n=5, indicative only | -0.015280745 | 0.013249887 | -0.031732653 | 0.0011711631 | 0.061405298 | -1.1532736 | 0.49124238 | inconclusive |
| exlightnet | top20 | 5 | n=5, indicative only | 0.26874294 | 0.019631362 | 0.24436738 | 0.29311849 | 6.7854877e-06 | 13.68947 | 7.4640365e-05 | difference supported |

### Explainability evidence

Background and random test examples use a fixed seed. FP32 additionally includes the local examples listed below.

| Artifact | Method/status | Train backgrounds | Random test flows | Extra local flows | Max abs. additivity error | Mean abs. additivity error |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | GradientExplainer | 200 | 300 | 2 | 2.1546996555411786 | 0.22188240032873874 |
| Dequantised INT8 | GradientExplainer | 200 | 300 | 0 | 2.229949461879082 | 0.22238371247660357 |

| Artifact | Fallback reason |
| --- | --- |
| FP32 | AssertionError: The SHAP explanations do not sum up to the model's output! This is either because of a rounding error or because an operator in your computation graph was not fully supported. If the sum difference of %f is significant compared to the scale of your model outputs, please post as a github issue, with a reproducible example so we can debug it. Used framework: pytorch - Max. diff: 1.1255382988370002 - Tolerance: 0.01 |
| Dequantised INT8 | AssertionError: The SHAP explanations do not sum up to the model's output! This is either because of a rounding error or because an operator in your computation graph was not fully supported. If the sum difference of %f is significant compared to the scale of your model outputs, please post as a github issue, with a reproducible example so we can debug it. Used framework: pytorch - Max. diff: 1.193267625531007 - Tolerance: 0.01 |

Deletion probabilities:

| Removed features | Mean original predicted-class probability |
| --- | --- |
| 0 | 0.8982132077217102 |
| 10 | 0.6382999420166016 |
| 20 | 0.6712844371795654 |

| SHAP deletion-AUC | Random deletion-AUC | Random permutations |
| --- | --- | --- |
| 0.6877264976501465 | 0.7114053964614868 | 20 |

| Quantisation stability measure | Value |
| --- | --- |
| spearman_top15_fp32_feature_set | 1.0 |
| top5_overlap | 5 |

Local examples (signed logit contributions, with correctness stated):

| True class | Predicted class | Correct | Probability | Test position |
| --- | --- | --- | --- | --- |
| dos_syn_hping | dos_syn_hping | True | 0.9443291425704956 | 0 |
| nmap_os_detection | nmap_xmas_tree_scan | False | 0.35721495747566223 | 44 |

| Example true class | Rank | Feature | Signed logit contribution | z-value |
| --- | --- | --- | --- | --- |
| dos_syn_hping | 1 | fwd_pkts_per_sec | -0.9050708229038346 | 1.8747349977493286 |
| dos_syn_hping | 2 | payload_bytes_per_second | 0.5766906291432785 | 1.8847445249557495 |
| dos_syn_hping | 3 | flow_pkts_payload.avg | 0.3947726929372381 | -0.09957305341959 |
| dos_syn_hping | 4 | fwd_pkts_payload.avg | 0.3860617163919143 | 0.4295502007007599 |
| dos_syn_hping | 5 | bwd_header_size_max | 0.3459979658671932 | 0.16557185351848602 |
| nmap_os_detection | 1 | flow_pkts_payload.avg | 3.434135264619387 | -1.2944403886795044 |
| nmap_os_detection | 2 | payload_bytes_per_second | 1.8243744299923048 | -0.9157596826553345 |
| nmap_os_detection | 3 | fwd_pkts_per_sec | -1.5401855444052899 | -0.8457872867584229 |
| nmap_os_detection | 4 | fwd_pkts_payload.avg | 1.412658535446821 | -2.2081925868988037 |
| nmap_os_detection | 5 | flow_pkts_payload.std | 0.8227950074643344 | -1.0291452407836914 |

Attributions explain logits; deletion measures the probability of the original predicted class. Additivity residuals are measured without correction. Deletion curves need not be monotone; neither causality nor guaranteed faithfulness is assumed. W8 uses identical background/test examples and GradientExplainer randomness for a controlled ranking probe.

DeepExplainer failed its additivity check and the prescribed GradientExplainer fallback was used. The residuals above show that these reduced-sample attributions do not establish exact additive efficiency; treat these explanation figures as approximate diagnostics.

### Hardware and runtime

| Field | Value |
| --- | --- |
| platform | Linux-6.1.0-32-amd64-x86_64-with-glibc2.36 |
| cpu | Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz |
| python | 3.11.2 |
| versions.numpy | 1.26.4 |
| versions.pandas | 2.2.3 |
| versions.scipy | 1.14.1 |
| versions.torch | 2.6.0+cpu |
| versions.scikit-learn | 1.6.1 |
| versions.xgboost-cpu | 2.1.4 |
| versions.shap | 0.46.0 |
| versions.nbformat | 5.10.4 |
| versions.nbconvert | 7.16.6 |
| versions.pytest | 8.3.5 |
| cuda_used | False |
| torch_threads | 1 |
| torch_interop_threads | 1 |
| threadpools | api: blas; library: openblas; threads: 1, api: openmp; library: openmp; threads: 1, api: blas; library: openblas; threads: 1, api: openmp; library: openmp; threads: 1 |

Latency uses one library thread, batch one, 100 warmups and 1000 timed inferences on training examples. Mean, median, p95 and peak RSS are saved per run. Inline measurements include preceding training in their process-lifetime RSS; isolated benchmark-only measurements include preceding benchmarks, with no training in that process. Neither is incremental model memory. No multi-core throughput extrapolation is reported. Raspberry Pi measurements require running `scripts/bench_pi.py` on the device; this host is not a Pi.

The recorded parallel pipeline resumed after an interrupted interactive invocation, reusing ten completed results and the frozen alpha selections. Its wall time covers the detached resumed invocation. Earlier logs and state are preserved under `results/full/interrupted/`; the alpha sweep was not repeated.

The benchmark-only process runs separately from training and explanations. Successful measurements have consecutive CPU-idle checks recorded in their benchmark context. RSS is the benchmark process lifetime peak across previously loaded models, with no training in that process.

| Benchmark attempted | Successful | Failed | Skipped after prerequisite failure | Gated out |
| --- | --- | --- | --- | --- |
| 95 | 95 | 0 | 0 | 0 |

Benchmark recovery after the initial CPU-idle timeout (the stopped campaign remains in `results/full/benchmarks/`):

| Status | Waiting started (UTC) | Benchmark started (UTC) | Finished (UTC) | Last CPU busy fraction |
| --- | --- | --- | --- | --- |
| complete | 2026-11-17T05:26:43.659348+00:00 | 2026-11-17T05:26:43.659367+00:00 | 2026-11-17T05:39:01.526395+00:00 | — |

| Stage | Wall seconds | Processes per stage/worker | Threads per process | Started (UTC) | Finished (UTC) |
| --- | --- | --- | --- | --- | --- |
| Alpha sweep | 2671.9628458939987 | 1 | 1 | — | — |
| Seed 42 worker | 12822.990029392 | 1 | 1 | 2026-11-16T11:29:07.171982+00:00 | 2026-11-16T15:02:50.162018+00:00 |
| Seed 43 worker | 11759.089673106999 | 1 | 1 | 2026-11-16T11:29:07.177725+00:00 | 2026-11-16T14:45:06.267408+00:00 |
| Seed 44 worker | 13322.728464467 | 1 | 1 | 2026-11-16T11:29:07.168492+00:00 | 2026-11-16T15:11:09.896963+00:00 |
| Seed 45 worker | 9278.26628422 | 1 | 1 | 2026-11-16T11:29:07.173218+00:00 | 2026-11-16T14:03:45.439522+00:00 |
| Seed 46 worker | 12544.516884881003 | 1 | 1 | 2026-11-16T11:29:07.177385+00:00 | 2026-11-16T14:58:11.694309+00:00 |
| Recorded parallel pipeline | 13637.783383412 | up to 5 workers | 1 | 2026-11-16T11:29:02.211256+00:00 | 2026-11-16T15:16:19.994776+00:00 |
| Cache-only aggregation | 6.6919 | 1 | 1 | 2026-11-16T15:11:10.650609+00:00 | 2026-11-16T15:11:17.342509+00:00 |
| Final isolated benchmark | 735.5698863359976 | 1 | 1 | 2026-11-17T05:26:45.467288+00:00 | 2026-11-17T05:39:01.037188+00:00 |

Seed workers overlap in wall time, so their elapsed times must not be summed as end-to-end runtime. Aggregate-only publication reuses final probabilities and test receipts.

## Cached CPU statistics — n=5, indicative only

### Seed summaries

All requested seeds contribute; sample SD uses ddof=1. Seed 42 is separate. Zero observed seed variance is flagged.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/model_summary.csv

| task | model | n_seeds | sample_label | macro_f1_mean | macro_f1_sd | macro_f1_seed42 | zero_seed_variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 5 | n=5, indicative only | 0.96320434 | 0.0082074518 | 0.96101575 | False |
| binary | exlightnet | 5 | n=5, indicative only | 0.98706926 | 0.00085010952 | 0.98805448 | False |
| binary | exlightnet_w8 | 5 | n=5, indicative only | 0.98704215 | 0.00094033702 | 0.98776167 | False |
| binary | mlp | 5 | n=5, indicative only | 0.98690001 | 0.00059585295 | 0.9871227 | False |
| binary | random_forest | 5 | n=5, indicative only | 0.99373275 | 0.0001677344 | 0.99379069 | False |
| binary | std_cnn | 5 | n=5, indicative only | 0.97965714 | 0.0056678274 | 0.97775677 | False |
| binary | xgboost | 5 | n=5, indicative only | 0.99337056 | 0 | 0.99337056 | True |
| multiclass | balanced | 5 | n=5, indicative only | 0.80651271 | 0.03054199 | 0.82980444 | False |
| multiclass | bilstm | 5 | n=5, indicative only | 0.72821022 | 0.028676134 | 0.770176 | False |
| multiclass | exlightnet | 5 | n=5, indicative only | 0.84659835 | 0.019463299 | 0.82479633 | False |
| multiclass | exlightnet_w8 | 5 | n=5, indicative only | 0.84678985 | 0.016495548 | 0.82530718 | False |
| multiclass | mlp | 5 | n=5, indicative only | 0.83559904 | 0.010352938 | 0.8354536 | False |
| multiclass | no_sampler | 5 | n=5, indicative only | 0.84464027 | 0.043865446 | 0.86503481 | False |
| multiclass | no_se | 5 | n=5, indicative only | 0.82701436 | 0.022346486 | 0.82196499 | False |
| multiclass | random_forest | 5 | n=5, indicative only | 0.85960042 | 0.0028884842 | 0.85919185 | False |
| multiclass | standard_conv | 5 | n=5, indicative only | 0.8618791 | 0.018604256 | 0.8321795 | False |
| multiclass | std_cnn | 5 | n=5, indicative only | 0.70627568 | 0.10092912 | 0.83240737 | False |
| multiclass | top20 | 5 | n=5, indicative only | 0.57785541 | 0.012689922 | 0.5798038 | False |
| multiclass | xgboost | 5 | n=5, indicative only | 0.83266412 | 0 | 0.83266412 | True |

### Paired comparisons

Differences are ExLightNet minus the comparison on matched seeds. Two-sided paired t tests, 95% t intervals and signed Cohen's dz use the paired differences. Holm adjustment includes all comparisons within each task, including ablations and W8. Wilcoxon is not used below six pairs. A constant model vector reduces the paired test to a one-sample test against that constant.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/paired_tests.csv

| task | reference | model | reference_device | comparison_device | device_label | n_pairs | sample_label | mean_difference | difference_sd | ci_low | ci_high | p_value | cohens_dz | p_holm | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | exlightnet | bilstm | cpu | cpu | cpu | 5 | n=5, indicative only | 0.023864919 | 0.0083679121 | 0.013474783 | 0.034255054 | 0.0031017077 | 2.8519562 | 0.012406831 | difference supported |
| binary | exlightnet | exlightnet_w8 | cpu | cpu | cpu | 5 | n=5, indicative only | 2.7112812e-05 | 0.00026091233 | -0.00029685263 | 0.00035107825 | 0.82766144 | 0.10391541 | 0.92687815 | practically equivalent |
| binary | exlightnet | mlp | cpu | cpu | cpu | 5 | n=5, indicative only | 0.00016925042 | 0.00046730171 | -0.00041098129 | 0.00074948213 | 0.46343908 | 0.36218661 | 0.92687815 | practically equivalent |
| binary | exlightnet | random_forest | cpu | cpu | cpu | 5 | n=5, indicative only | -0.0066634877 | 0.00087261072 | -0.007746977 | -0.0055799984 | 6.8995607e-05 | -7.6362662 | 0.00041397364 | difference supported |
| binary | exlightnet | std_cnn | cpu | cpu | cpu | 5 | n=5, indicative only | 0.0074121187 | 0.0059868255 | -2.1506991e-05 | 0.014845744 | 0.050412936 | 1.2380716 | 0.15123881 | inconclusive |
| binary | exlightnet | xgboost | cpu | cpu | cpu | 5 | n=5, indicative only | -0.0063013004 | 0.00085010952 | -0.0073568508 | -0.00524575 | 7.7611047e-05 | -7.4123396 | 0.00041397364 | difference supported |
| multiclass | exlightnet | balanced | cpu | cpu | cpu | 5 | n=5, indicative only | 0.040085642 | 0.03972046 | -0.0092338231 | 0.089405107 | 0.087004625 | 1.0091938 | 0.60903237 | inconclusive |
| multiclass | exlightnet | bilstm | cpu | cpu | cpu | 5 | n=5, indicative only | 0.11838813 | 0.037002919 | 0.072442939 | 0.16433332 | 0.0020200844 | 3.1994268 | 0.020200844 | difference supported |
| multiclass | exlightnet | exlightnet_w8 | cpu | cpu | cpu | 5 | n=5, indicative only | -0.00019149719 | 0.0044696414 | -0.00574129 | 0.0053582956 | 0.92828553 | -0.042843972 | 1 | practically equivalent |
| multiclass | exlightnet | mlp | cpu | cpu | cpu | 5 | n=5, indicative only | 0.010999316 | 0.014203773 | -0.0066369984 | 0.02863563 | 0.15838638 | 0.77439391 | 0.95031827 | inconclusive |
| multiclass | exlightnet | no_sampler | cpu | cpu | cpu | 5 | n=5, indicative only | 0.0019580848 | 0.053311308 | -0.064236647 | 0.068152817 | 0.93848957 | 0.036729258 | 1 | inconclusive |
| multiclass | exlightnet | no_se | cpu | cpu | cpu | 5 | n=5, indicative only | 0.019583988 | 0.033123131 | -0.021543811 | 0.060711788 | 0.25667868 | 0.5912481 | 0.95031827 | inconclusive |
| multiclass | exlightnet | random_forest | cpu | cpu | cpu | 5 | n=5, indicative only | -0.013002073 | 0.017637751 | -0.034902233 | 0.0088980878 | 0.17462224 | -0.73717293 | 0.95031827 | inconclusive |
| multiclass | exlightnet | standard_conv | cpu | cpu | cpu | 5 | n=5, indicative only | -0.015280745 | 0.013249887 | -0.031732653 | 0.0011711631 | 0.061405298 | -1.1532736 | 0.49124238 | inconclusive |
| multiclass | exlightnet | std_cnn | cpu | cpu | cpu | 5 | n=5, indicative only | 0.14032267 | 0.10198683 | 0.013689301 | 0.26695604 | 0.03705458 | 1.3758902 | 0.33349122 | inconclusive |
| multiclass | exlightnet | top20 | cpu | cpu | cpu | 5 | n=5, indicative only | 0.26874294 | 0.019631362 | 0.24436738 | 0.29311849 | 6.7854877e-06 | 13.68947 | 7.4640365e-05 | difference supported |
| multiclass | exlightnet | xgboost | cpu | cpu | cpu | 5 | n=5, indicative only | 0.013934229 | 0.019463299 | -0.010232648 | 0.038101107 | 0.18466347 | 0.7159233 | 0.95031827 | inconclusive |

### Flow bootstrap uncertainty

Each replicate selects a training seed uniformly and resamples test flows. The estimand is single-random-run plus flow uncertainty; this is not a confidence interval for the seed mean. The pairing/method column identifies cached-prediction or UNPAIRED confusion-cell resampling.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/flow_bootstrap.csv

| task | model | macro_f1_mean | ci_low | ci_high | pairing | replicates | n_seeds | sample_label | estimand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 0.96328331 | 0.95343901 | 0.97906561 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | exlightnet | 0.98705198 | 0.98375672 | 0.99006245 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | exlightnet_w8 | 0.98702089 | 0.98355899 | 0.98993563 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | mlp | 0.98693485 | 0.98395666 | 0.98971897 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | random_forest | 0.99374066 | 0.99171284 | 0.99556998 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | std_cnn | 0.97963317 | 0.96809791 | 0.98624334 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | xgboost | 0.99337869 | 0.99137499 | 0.99512625 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | balanced | 0.80198411 | 0.7394644 | 0.87004379 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | bilstm | 0.72660459 | 0.67216124 | 0.79026036 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | 0.84243254 | 0.77988036 | 0.88967585 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet_w8 | 0.84233269 | 0.78003684 | 0.88668389 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | mlp | 0.83162694 | 0.77855859 | 0.87455727 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | no_sampler | 0.84030307 | 0.77922199 | 0.89785096 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | no_se | 0.82509695 | 0.77432675 | 0.87608449 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | random_forest | 0.8552125 | 0.7975939 | 0.88958117 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | standard_conv | 0.85729141 | 0.79744618 | 0.89595717 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | std_cnn | 0.70805922 | 0.60011861 | 0.84674503 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | top20 | 0.57663772 | 0.53777549 | 0.59594518 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | xgboost | 0.82879666 | 0.78135615 | 0.86529423 | cached predictions; single-model flow bootstrap | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |

### Paired flow bootstrap differences

PAIRED means both models use identical resampled flows from the same matched training seed; an UNPAIRED fallback cannot support a paired flow comparison. The estimand remains a randomly selected training run plus finite-flow uncertainty.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/paired_flow_bootstrap.csv

| task | reference | model | mean_difference | ci_low | ci_high | pairing | replicates | n_seeds | sample_label | estimand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | exlightnet | bilstm | 0.023750903 | 0.0079421782 | 0.034068215 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | exlightnet | exlightnet_w8 | 3.6941887e-05 | -0.00058702553 | 0.00073046324 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | exlightnet | mlp | 0.00019525131 | -0.0023360339 | 0.0028782232 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | exlightnet | random_forest | -0.0066463616 | -0.010039149 | -0.0034388012 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | exlightnet | std_cnn | 0.0074261148 | 0.00068123388 | 0.018646456 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| binary | exlightnet | xgboost | -0.0062915133 | -0.0097868191 | -0.002887803 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | balanced | 0.040727099 | -0.020957608 | 0.1129099 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | bilstm | 0.11541179 | 0.031292039 | 0.17348365 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | exlightnet_w8 | -0.00020864734 | -0.010618358 | 0.0072622558 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | mlp | 0.0093031227 | -0.033466335 | 0.047186386 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | no_sampler | 0.0017069515 | -0.062060314 | 0.10347411 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | no_se | 0.017266001 | -0.023618599 | 0.078081827 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | random_forest | -0.013180607 | -0.059232295 | 0.025940337 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | standard_conv | -0.015110664 | -0.058733146 | 0.0080031699 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | std_cnn | 0.13355537 | -0.023781399 | 0.24476997 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | top20 | 0.26569583 | 0.20215019 | 0.32041017 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |
| multiclass | exlightnet | xgboost | 0.012881207 | -0.03434633 | 0.055841929 | PAIRED | 2000 | 5 | n=5, indicative only | single-random-run plus flow uncertainty; not CI for the seed mean |

### Per-class F1 and test support

Every class metric is accompanied by its fixed test support. Mean/sample-SD and seed-42 F1 are computed from cached predictions/confusion matrices.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/per_class_f1.csv

| task | model | class_name | test_support | f1_mean | f1_sd | f1_seed42 | n_seeds | sample_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | normal | 1876 | 0.93412088 | 0.014600382 | 0.93019648 | 5 | n=5, indicative only |
| binary | bilstm | attack | 16592 | 0.99228781 | 0.0018151444 | 0.99183501 | 5 | n=5, indicative only |
| binary | exlightnet | normal | 1876 | 0.9767596 | 0.0015273025 | 0.97851772 | 5 | n=5, indicative only |
| binary | exlightnet | attack | 16592 | 0.99737892 | 0.00017318858 | 0.99759123 | 5 | n=5, indicative only |
| binary | exlightnet_w8 | normal | 1876 | 0.97671147 | 0.001691152 | 0.97799249 | 5 | n=5, indicative only |
| binary | exlightnet_w8 | attack | 16592 | 0.99737283 | 0.00018974779 | 0.99753086 | 5 | n=5, indicative only |
| binary | mlp | normal | 1876 | 0.97646428 | 0.0010787193 | 0.97686785 | 5 | n=5, indicative only |
| binary | mlp | attack | 16592 | 0.99733574 | 0.00011305961 | 0.99737754 | 5 | n=5, indicative only |
| binary | random_forest | normal | 1876 | 0.98874379 | 0.00030107414 | 0.98884758 | 5 | n=5, indicative only |
| binary | random_forest | attack | 16592 | 0.99872171 | 3.4395467e-05 | 0.9987338 | 5 | n=5, indicative only |
| binary | std_cnn | normal | 1876 | 0.9634763 | 0.010133533 | 0.96 | 5 | n=5, indicative only |
| binary | std_cnn | attack | 16592 | 0.99583798 | 0.0012031766 | 0.99551353 | 5 | n=5, indicative only |
| binary | xgboost | normal | 1876 | 0.98809839 | 1.2412671e-16 | 0.98809839 | 5 | n=5, indicative only |
| binary | xgboost | attack | 16592 | 0.99864274 | 0 | 0.99864274 | 5 | n=5, indicative only |
| multiclass | balanced | arp_poisioning | 1162 | 0.9240117 | 0.02785882 | 0.91737203 | 5 | n=5, indicative only |
| multiclass | balanced | ddos_slowloris | 80 | 0.87640533 | 0.013717144 | 0.88888889 | 5 | n=5, indicative only |
| multiclass | balanced | dos_syn_hping | 14199 | 0.99997183 | 2.9460008e-05 | 1 | 5 | n=5, indicative only |
| multiclass | balanced | metasploit_brute_force_ssh | 5 | 0.71637427 | 0.19910971 | 1 | 5 | n=5, indicative only |
| multiclass | balanced | mqtt_publish | 622 | 0.99774323 | 0.0010539777 | 0.99838969 | 5 | n=5, indicative only |
| multiclass | balanced | nmap_fin_scan | 4 | 0.56467305 | 0.16388297 | 0.66666667 | 5 | n=5, indicative only |
| multiclass | balanced | nmap_os_detection | 300 | 0.49156811 | 0.16297151 | 0.23592493 | 5 | n=5, indicative only |
| multiclass | balanced | nmap_tcp_scan | 151 | 0.98441456 | 0.0085654928 | 0.99342105 | 5 | n=5, indicative only |
| multiclass | balanced | nmap_udp_scan | 389 | 0.89091464 | 0.076074851 | 0.93188011 | 5 | n=5, indicative only |
| multiclass | balanced | nmap_xmas_tree_scan | 302 | 0.50462611 | 0.13497988 | 0.64891041 | 5 | n=5, indicative only |
| multiclass | balanced | thing_speak | 1216 | 0.94127228 | 0.011286517 | 0.9299308 | 5 | n=5, indicative only |
| multiclass | balanced | wipro_bulb | 38 | 0.78617739 | 0.046995996 | 0.74626866 | 5 | n=5, indicative only |
| multiclass | bilstm | arp_poisioning | 1162 | 0.92527619 | 0.019817943 | 0.94989293 | 5 | n=5, indicative only |
| multiclass | bilstm | ddos_slowloris | 80 | 0.88073713 | 0.025160955 | 0.87431694 | 5 | n=5, indicative only |
| multiclass | bilstm | dos_syn_hping | 14199 | 0.99990143 | 0.00013496955 | 1 | 5 | n=5, indicative only |
| multiclass | bilstm | metasploit_brute_force_ssh | 5 | 0.64058442 | 0.18175874 | 0.71428571 | 5 | n=5, indicative only |
| multiclass | bilstm | mqtt_publish | 622 | 0.99742662 | 0.00087828231 | 0.99758648 | 5 | n=5, indicative only |
| multiclass | bilstm | nmap_fin_scan | 4 | 0.15524476 | 0.23993251 | 0.54545455 | 5 | n=5, indicative only |
| multiclass | bilstm | nmap_os_detection | 300 | 0.34420206 | 0.076262003 | 0.25352113 | 5 | n=5, indicative only |
| multiclass | bilstm | nmap_tcp_scan | 151 | 0.68767554 | 0.053030858 | 0.67002519 | 5 | n=5, indicative only |
| multiclass | bilstm | nmap_udp_scan | 389 | 0.92385984 | 0.014199847 | 0.92802057 | 5 | n=5, indicative only |
| multiclass | bilstm | nmap_xmas_tree_scan | 302 | 0.55220675 | 0.037629209 | 0.57856094 | 5 | n=5, indicative only |
| multiclass | bilstm | thing_speak | 1216 | 0.9424634 | 0.015677953 | 0.95936322 | 5 | n=5, indicative only |
| multiclass | bilstm | wipro_bulb | 38 | 0.68894452 | 0.078135088 | 0.77108434 | 5 | n=5, indicative only |
| multiclass | exlightnet | arp_poisioning | 1162 | 0.95039511 | 0.0052658468 | 0.94741306 | 5 | n=5, indicative only |
| multiclass | exlightnet | ddos_slowloris | 80 | 0.87259115 | 0.013998337 | 0.87912088 | 5 | n=5, indicative only |
| multiclass | exlightnet | dos_syn_hping | 14199 | 0.99998591 | 1.9286685e-05 | 1 | 5 | n=5, indicative only |
| multiclass | exlightnet | metasploit_brute_force_ssh | 5 | 0.92727273 | 0.040655781 | 0.90909091 | 5 | n=5, indicative only |
| multiclass | exlightnet | mqtt_publish | 622 | 0.99790543 | 0.00044206653 | 0.99758259 | 5 | n=5, indicative only |
| multiclass | exlightnet | nmap_fin_scan | 4 | 0.72337662 | 0.18387061 | 0.5 | 5 | n=5, indicative only |
| multiclass | exlightnet | nmap_os_detection | 300 | 0.37060808 | 0.17068273 | 0.52158273 | 5 | n=5, indicative only |
| multiclass | exlightnet | nmap_tcp_scan | 151 | 0.99018092 | 0.0045799995 | 0.99342105 | 5 | n=5, indicative only |
| multiclass | exlightnet | nmap_udp_scan | 389 | 0.94681683 | 0.013020944 | 0.94594595 | 5 | n=5, indicative only |
| multiclass | exlightnet | nmap_xmas_tree_scan | 302 | 0.62221631 | 0.035382751 | 0.5785381 | 5 | n=5, indicative only |
| multiclass | exlightnet | thing_speak | 1216 | 0.95908307 | 0.0039283433 | 0.95819398 | 5 | n=5, indicative only |
| multiclass | exlightnet | wipro_bulb | 38 | 0.79874804 | 0.12941633 | 0.66666667 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | arp_poisioning | 1162 | 0.9508324 | 0.0061180956 | 0.9477707 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | ddos_slowloris | 80 | 0.87165108 | 0.014404291 | 0.87912088 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | dos_syn_hping | 14199 | 0.99998591 | 1.9286685e-05 | 1 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | metasploit_brute_force_ssh | 5 | 0.92727273 | 0.040655781 | 0.90909091 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | mqtt_publish | 622 | 0.99790543 | 0.00044206653 | 0.99758259 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | nmap_fin_scan | 4 | 0.72337662 | 0.18387061 | 0.5 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | nmap_os_detection | 300 | 0.34580924 | 0.1501462 | 0.47443763 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | nmap_tcp_scan | 151 | 0.99018092 | 0.0045799995 | 0.99342105 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | nmap_udp_scan | 389 | 0.94622685 | 0.014276457 | 0.94594595 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | nmap_xmas_tree_scan | 302 | 0.63493303 | 0.025997103 | 0.63098592 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | thing_speak | 1216 | 0.95989485 | 0.0040918668 | 0.95866388 | 5 | n=5, indicative only |
| multiclass | exlightnet_w8 | wipro_bulb | 38 | 0.81340911 | 0.10537623 | 0.66666667 | 5 | n=5, indicative only |
| multiclass | mlp | arp_poisioning | 1162 | 0.95251912 | 0.0037328176 | 0.94944302 | 5 | n=5, indicative only |
| multiclass | mlp | ddos_slowloris | 80 | 0.86647054 | 0.020602736 | 0.84656085 | 5 | n=5, indicative only |
| multiclass | mlp | dos_syn_hping | 14199 | 1 | 0 | 1 | 5 | n=5, indicative only |
| multiclass | mlp | metasploit_brute_force_ssh | 5 | 0.91515152 | 0.083402175 | 0.83333333 | 5 | n=5, indicative only |
| multiclass | mlp | mqtt_publish | 622 | 0.99838969 | 0 | 0.99838969 | 5 | n=5, indicative only |
| multiclass | mlp | nmap_fin_scan | 4 | 0.58909091 | 0.024393469 | 0.6 | 5 | n=5, indicative only |
| multiclass | mlp | nmap_os_detection | 300 | 0.44095678 | 0.17608921 | 0.54948805 | 5 | n=5, indicative only |
| multiclass | mlp | nmap_tcp_scan | 151 | 0.97332586 | 0.029562601 | 0.99342105 | 5 | n=5, indicative only |
| multiclass | mlp | nmap_udp_scan | 389 | 0.93797492 | 0.012905067 | 0.93125811 | 5 | n=5, indicative only |
| multiclass | mlp | nmap_xmas_tree_scan | 302 | 0.56087841 | 0.12231433 | 0.56117455 | 5 | n=5, indicative only |
| multiclass | mlp | thing_speak | 1216 | 0.96462896 | 0.0018586999 | 0.96237458 | 5 | n=5, indicative only |
| multiclass | mlp | wipro_bulb | 38 | 0.82780171 | 0.029762522 | 0.8 | 5 | n=5, indicative only |
| multiclass | no_sampler | arp_poisioning | 1162 | 0.96017896 | 0.0064033026 | 0.9593777 | 5 | n=5, indicative only |
| multiclass | no_sampler | ddos_slowloris | 80 | 0.89459557 | 0.0091329637 | 0.88888889 | 5 | n=5, indicative only |
| multiclass | no_sampler | dos_syn_hping | 14199 | 0.99992254 | 8.4053288e-05 | 0.99978876 | 5 | n=5, indicative only |
| multiclass | no_sampler | metasploit_brute_force_ssh | 5 | 1 | 0 | 1 | 5 | n=5, indicative only |
| multiclass | no_sampler | mqtt_publish | 622 | 0.99806685 | 0.00044206653 | 0.99838969 | 5 | n=5, indicative only |
| multiclass | no_sampler | nmap_fin_scan | 4 | 0.51428571 | 0.46947648 | 0.85714286 | 5 | n=5, indicative only |
| multiclass | no_sampler | nmap_os_detection | 300 | 0.4796501 | 0.13940289 | 0.54419411 | 5 | n=5, indicative only |
| multiclass | no_sampler | nmap_tcp_scan | 151 | 0.99276963 | 0.0014566275 | 0.99342105 | 5 | n=5, indicative only |
| multiclass | no_sampler | nmap_udp_scan | 389 | 0.95553696 | 0.0064779747 | 0.952 | 5 | n=5, indicative only |
| multiclass | no_sampler | nmap_xmas_tree_scan | 302 | 0.57818214 | 0.05389097 | 0.56913183 | 5 | n=5, indicative only |
| multiclass | no_sampler | thing_speak | 1216 | 0.96537155 | 0.0055616653 | 0.96136641 | 5 | n=5, indicative only |
| multiclass | no_sampler | wipro_bulb | 38 | 0.79712316 | 0.080860384 | 0.65671642 | 5 | n=5, indicative only |
| multiclass | no_se | arp_poisioning | 1162 | 0.9371724 | 0.0052341142 | 0.9319026 | 5 | n=5, indicative only |
| multiclass | no_se | ddos_slowloris | 80 | 0.88462492 | 0.019831463 | 0.85106383 | 5 | n=5, indicative only |
| multiclass | no_se | dos_syn_hping | 14199 | 1 | 0 | 1 | 5 | n=5, indicative only |
| multiclass | no_se | metasploit_brute_force_ssh | 5 | 0.98181818 | 0.040655781 | 1 | 5 | n=5, indicative only |
| multiclass | no_se | mqtt_publish | 622 | 0.99774453 | 0.00067550492 | 0.99677419 | 5 | n=5, indicative only |
| multiclass | no_se | nmap_fin_scan | 4 | 0.49117647 | 0.16229858 | 0.5 | 5 | n=5, indicative only |
| multiclass | no_se | nmap_os_detection | 300 | 0.27556816 | 0.1937976 | 0.25531915 | 5 | n=5, indicative only |
| multiclass | no_se | nmap_tcp_scan | 151 | 0.97476354 | 0.026033185 | 0.93209877 | 5 | n=5, indicative only |
| multiclass | no_se | nmap_udp_scan | 389 | 0.91498631 | 0.030448936 | 0.93640054 | 5 | n=5, indicative only |
| multiclass | no_se | nmap_xmas_tree_scan | 302 | 0.63526189 | 0.032274868 | 0.65422886 | 5 | n=5, indicative only |
| multiclass | no_se | thing_speak | 1216 | 0.95528383 | 0.0060961992 | 0.94468085 | 5 | n=5, indicative only |
| multiclass | no_se | wipro_bulb | 38 | 0.87577212 | 0.026464939 | 0.86111111 | 5 | n=5, indicative only |
| multiclass | random_forest | arp_poisioning | 1162 | 0.98732246 | 0.00064442379 | 0.98756965 | 5 | n=5, indicative only |
| multiclass | random_forest | ddos_slowloris | 80 | 0.91717988 | 0.0048770535 | 0.92405063 | 5 | n=5, indicative only |
| multiclass | random_forest | dos_syn_hping | 14199 | 1 | 0 | 1 | 5 | n=5, indicative only |
| multiclass | random_forest | metasploit_brute_force_ssh | 5 | 0.90909091 | 1.2412671e-16 | 0.90909091 | 5 | n=5, indicative only |
| multiclass | random_forest | mqtt_publish | 622 | 0.99919549 | 0 | 0.99919549 | 5 | n=5, indicative only |
| multiclass | random_forest | nmap_fin_scan | 4 | 0.68333333 | 0.0372678 | 0.66666667 | 5 | n=5, indicative only |
| multiclass | random_forest | nmap_os_detection | 300 | 0.47085674 | 0.0053667802 | 0.47018739 | 5 | n=5, indicative only |
| multiclass | random_forest | nmap_tcp_scan | 151 | 0.99669967 | 0 | 0.99669967 | 5 | n=5, indicative only |
| multiclass | random_forest | nmap_udp_scan | 389 | 0.97587306 | 0.0010624408 | 0.97686375 | 5 | n=5, indicative only |
| multiclass | random_forest | nmap_xmas_tree_scan | 302 | 0.47846851 | 0.0048080178 | 0.48366013 | 5 | n=5, indicative only |
| multiclass | random_forest | thing_speak | 1216 | 0.98762791 | 0.00028792199 | 0.98722703 | 5 | n=5, indicative only |
| multiclass | random_forest | wipro_bulb | 38 | 0.90955711 | 0.0090879201 | 0.90909091 | 5 | n=5, indicative only |
| multiclass | standard_conv | arp_poisioning | 1162 | 0.95117455 | 0.011096354 | 0.94932871 | 5 | n=5, indicative only |
| multiclass | standard_conv | ddos_slowloris | 80 | 0.88595167 | 0.023928168 | 0.88888889 | 5 | n=5, indicative only |
| multiclass | standard_conv | dos_syn_hping | 14199 | 1 | 0 | 1 | 5 | n=5, indicative only |
| multiclass | standard_conv | metasploit_brute_force_ssh | 5 | 0.96363636 | 0.04979296 | 1 | 5 | n=5, indicative only |
| multiclass | standard_conv | mqtt_publish | 622 | 0.9980697 | 0.00071552696 | 0.99838969 | 5 | n=5, indicative only |
| multiclass | standard_conv | nmap_fin_scan | 4 | 0.75630252 | 0.22548585 | 0.35294118 | 5 | n=5, indicative only |
| multiclass | standard_conv | nmap_os_detection | 300 | 0.53886188 | 0.1245298 | 0.33497537 | 5 | n=5, indicative only |
| multiclass | standard_conv | nmap_tcp_scan | 151 | 0.99147104 | 0.0029049284 | 0.99016393 | 5 | n=5, indicative only |
| multiclass | standard_conv | nmap_udp_scan | 389 | 0.95462828 | 0.0086384435 | 0.94021739 | 5 | n=5, indicative only |
| multiclass | standard_conv | nmap_xmas_tree_scan | 302 | 0.49815048 | 0.16439347 | 0.65321564 | 5 | n=5, indicative only |
| multiclass | standard_conv | thing_speak | 1216 | 0.9577001 | 0.010474289 | 0.95698052 | 5 | n=5, indicative only |
| multiclass | standard_conv | wipro_bulb | 38 | 0.84660257 | 0.06942571 | 0.92105263 | 5 | n=5, indicative only |
| multiclass | std_cnn | arp_poisioning | 1162 | 0.83796473 | 0.10835504 | 0.95298069 | 5 | n=5, indicative only |
| multiclass | std_cnn | ddos_slowloris | 80 | 0.8505954 | 0.063725834 | 0.85106383 | 5 | n=5, indicative only |
| multiclass | std_cnn | dos_syn_hping | 14199 | 0.99990142 | 0.00012042063 | 1 | 5 | n=5, indicative only |
| multiclass | std_cnn | metasploit_brute_force_ssh | 5 | 0.83333333 | 0.23570226 | 1 | 5 | n=5, indicative only |
| multiclass | std_cnn | mqtt_publish | 622 | 0.99644718 | 0.0020325109 | 0.99838969 | 5 | n=5, indicative only |
| multiclass | std_cnn | nmap_fin_scan | 4 | 0.13815789 | 0.19033521 | 0.31578947 | 5 | n=5, indicative only |
| multiclass | std_cnn | nmap_os_detection | 300 | 0.38382797 | 0.13376816 | 0.52575488 | 5 | n=5, indicative only |
| multiclass | std_cnn | nmap_tcp_scan | 151 | 0.74262663 | 0.23882525 | 0.99342105 | 5 | n=5, indicative only |
| multiclass | std_cnn | nmap_udp_scan | 389 | 0.71078941 | 0.20511552 | 0.94070081 | 5 | n=5, indicative only |
| multiclass | std_cnn | nmap_xmas_tree_scan | 302 | 0.43279595 | 0.25703956 | 0.57232704 | 5 | n=5, indicative only |
| multiclass | std_cnn | thing_speak | 1216 | 0.85041612 | 0.13825527 | 0.96174863 | 5 | n=5, indicative only |
| multiclass | std_cnn | wipro_bulb | 38 | 0.69845213 | 0.11835034 | 0.87671233 | 5 | n=5, indicative only |
| multiclass | top20 | arp_poisioning | 1162 | 0.88399258 | 0.013278986 | 0.87319422 | 5 | n=5, indicative only |
| multiclass | top20 | ddos_slowloris | 80 | 0.8774293 | 0.016028852 | 0.88888889 | 5 | n=5, indicative only |
| multiclass | top20 | dos_syn_hping | 14199 | 0.99981692 | 1.5741968e-05 | 0.99982396 | 5 | n=5, indicative only |
| multiclass | top20 | metasploit_brute_force_ssh | 5 | 0.93566434 | 0.10102415 | 1 | 5 | n=5, indicative only |
| multiclass | top20 | mqtt_publish | 622 | 0.99790595 | 0.00072196882 | 0.99677419 | 5 | n=5, indicative only |
| multiclass | top20 | nmap_fin_scan | 4 | 0 | 0 | 0 | 5 | n=5, indicative only |
| multiclass | top20 | nmap_os_detection | 300 | 0.09202454 | 0.20577313 | 0 | 5 | n=5, indicative only |
| multiclass | top20 | nmap_tcp_scan | 151 | 0 | 0 | 0 | 5 | n=5, indicative only |
| multiclass | top20 | nmap_udp_scan | 389 | 0.1022508 | 0.0077963572 | 0.092457421 | 5 | n=5, indicative only |
| multiclass | top20 | nmap_xmas_tree_scan | 302 | 0.36662386 | 0.20495142 | 0.45941807 | 5 | n=5, indicative only |
| multiclass | top20 | thing_speak | 1216 | 0.94875902 | 0.0115859 | 0.92969749 | 5 | n=5, indicative only |
| multiclass | top20 | wipro_bulb | 38 | 0.72979766 | 0.04979503 | 0.7173913 | 5 | n=5, indicative only |
| multiclass | xgboost | arp_poisioning | 1162 | 0.98799314 | 0 | 0.98799314 | 5 | n=5, indicative only |
| multiclass | xgboost | ddos_slowloris | 80 | 0.91139241 | 1.2412671e-16 | 0.91139241 | 5 | n=5, indicative only |
| multiclass | xgboost | dos_syn_hping | 14199 | 1 | 0 | 1 | 5 | n=5, indicative only |
| multiclass | xgboost | metasploit_brute_force_ssh | 5 | 0.83333333 | 0 | 0.83333333 | 5 | n=5, indicative only |
| multiclass | xgboost | mqtt_publish | 622 | 0.99759036 | 1.2412671e-16 | 0.99759036 | 5 | n=5, indicative only |
| multiclass | xgboost | nmap_fin_scan | 4 | 0.46153846 | 0 | 0.46153846 | 5 | n=5, indicative only |
| multiclass | xgboost | nmap_os_detection | 300 | 0.45360825 | 0 | 0.45360825 | 5 | n=5, indicative only |
| multiclass | xgboost | nmap_tcp_scan | 151 | 1 | 0 | 1 | 5 | n=5, indicative only |
| multiclass | xgboost | nmap_udp_scan | 389 | 0.97164948 | 1.2412671e-16 | 0.97164948 | 5 | n=5, indicative only |
| multiclass | xgboost | nmap_xmas_tree_scan | 302 | 0.47649919 | 6.2063354e-17 | 0.47649919 | 5 | n=5, indicative only |
| multiclass | xgboost | thing_speak | 1216 | 0.98927393 | 1.2412671e-16 | 0.98927393 | 5 | n=5, indicative only |
| multiclass | xgboost | wipro_bulb | 38 | 0.90909091 | 1.2412671e-16 | 0.90909091 | 5 | n=5, indicative only |

### Post-hoc sensitivity

post-hoc sensitivity, not a replacement for macro-F1. Low-support classes are excluded from the macro average only; their flows still contribute false positives to retained classes.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/sensitivity.csv

| task | model | macro_f1_mean | macro_f1_sd | macro_f1_seed42 | n_seeds | sample_label | excluded_classes | min_support | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 0.96320434 | 0.0082074518 | 0.96101575 | 5 | n=5, indicative only | — | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| binary | exlightnet | 0.98706926 | 0.00085010952 | 0.98805448 | 5 | n=5, indicative only | — | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| binary | exlightnet_w8 | 0.98704215 | 0.00094033702 | 0.98776167 | 5 | n=5, indicative only | — | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| binary | mlp | 0.98690001 | 0.00059585295 | 0.9871227 | 5 | n=5, indicative only | — | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| binary | random_forest | 0.99373275 | 0.0001677344 | 0.99379069 | 5 | n=5, indicative only | — | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| binary | std_cnn | 0.97965714 | 0.0056678274 | 0.97775677 | 5 | n=5, indicative only | — | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| binary | xgboost | 0.99337056 | 0 | 0.99337056 | 5 | n=5, indicative only | — | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | balanced | 0.83971052 | 0.013913468 | 0.82909866 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | bilstm | 0.79426935 | 0.010722359 | 0.79823717 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | exlightnet | 0.85085309 | 0.0122087 | 0.8488465 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | exlightnet_w8 | 0.85108288 | 0.0091334931 | 0.84945953 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | mlp | 0.8522946 | 0.014095705 | 0.85921099 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | no_sampler | 0.86213975 | 0.012961562 | 0.85232749 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | no_se | 0.84511777 | 0.021040531 | 0.83635799 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | random_forest | 0.87227808 | 0.0011873611 | 0.87345447 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | standard_conv | 0.86226103 | 0.0085944733 | 0.86332128 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | std_cnn | 0.75038169 | 0.088414444 | 0.8673099 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | top20 | 0.59986006 | 0.0062491766 | 0.59576455 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |
| multiclass | xgboost | 0.86970977 | 0 | 0.86970977 | 5 | n=5, indicative only | metasploit_brute_force_ssh,nmap_fin_scan | 10 | post-hoc sensitivity, not a replacement for macro-F1 |

### Training dynamics

Persisted epoch counts and best-validation epochs define cap/stopping indicators. The cosine factor is evaluated at the last epoch using the recorded maximum epoch budget; instability is flagged rather than used to claim superiority.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/training_dynamics.csv

| task | model | seed | n_seeds | sample_label | macro_f1 | epochs_run | best_epoch | max_epochs | hit_cap | stopped_early | cosine_lr_factor | unstable | early_best_epoch | high_seed_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 42 | 5 | n=5, indicative only | 0.96101575 | 20 | 16 | 20 | True | False | 0 | False | False | False |
| binary | bilstm | 43 | 5 | n=5, indicative only | 0.9559369 | 14 | 8 | 20 | False | True | 0.20610737 | False | False | False |
| binary | bilstm | 44 | 5 | n=5, indicative only | 0.96322661 | 20 | 18 | 20 | True | False | 0 | False | False | False |
| binary | bilstm | 45 | 5 | n=5, indicative only | 0.97706849 | 20 | 19 | 20 | True | False | 0 | False | False | False |
| binary | bilstm | 46 | 5 | n=5, indicative only | 0.95877397 | 20 | 18 | 20 | True | False | 0 | False | False | False |
| binary | exlightnet | 42 | 5 | n=5, indicative only | 0.98805448 | 20 | 20 | 20 | True | False | 0 | False | False | False |
| binary | exlightnet | 43 | 5 | n=5, indicative only | 0.9872617 | 20 | 17 | 20 | True | False | 0 | False | False | False |
| binary | exlightnet | 44 | 5 | n=5, indicative only | 0.98570625 | 17 | 11 | 20 | False | True | 0.054496738 | False | False | False |
| binary | exlightnet | 45 | 5 | n=5, indicative only | 0.98709836 | 20 | 18 | 20 | True | False | 0 | False | False | False |
| binary | exlightnet | 46 | 5 | n=5, indicative only | 0.98722552 | 20 | 20 | 20 | True | False | 0 | False | False | False |
| binary | exlightnet_w8 | 42 | 5 | n=5, indicative only | 0.98776167 | — | — | — | — | — | — | False | False | False |
| binary | exlightnet_w8 | 43 | 5 | n=5, indicative only | 0.98755207 | — | — | — | — | — | — | False | False | False |
| binary | exlightnet_w8 | 44 | 5 | n=5, indicative only | 0.98541539 | — | — | — | — | — | — | False | False | False |
| binary | exlightnet_w8 | 45 | 5 | n=5, indicative only | 0.98711054 | — | — | — | — | — | — | False | False | False |
| binary | exlightnet_w8 | 46 | 5 | n=5, indicative only | 0.98737107 | — | — | — | — | — | — | False | False | False |
| binary | mlp | 42 | 5 | n=5, indicative only | 0.9871227 | 20 | 16 | 20 | True | False | 0 | False | False | False |
| binary | mlp | 43 | 5 | n=5, indicative only | 0.98726771 | 17 | 11 | 20 | False | True | 0.054496738 | False | False | False |
| binary | mlp | 44 | 5 | n=5, indicative only | 0.98586522 | 17 | 11 | 20 | False | True | 0.054496738 | False | False | False |
| binary | mlp | 45 | 5 | n=5, indicative only | 0.98730363 | 19 | 13 | 20 | False | True | 0.0061558297 | False | False | False |
| binary | mlp | 46 | 5 | n=5, indicative only | 0.98694081 | 17 | 11 | 20 | False | True | 0.054496738 | False | False | False |
| binary | random_forest | 42 | 5 | n=5, indicative only | 0.99379069 | — | — | — | — | — | — | False | False | False |
| binary | random_forest | 43 | 5 | n=5, indicative only | 0.99393996 | — | — | — | — | — | — | False | False | False |
| binary | random_forest | 44 | 5 | n=5, indicative only | 0.99379069 | — | — | — | — | — | — | False | False | False |
| binary | random_forest | 45 | 5 | n=5, indicative only | 0.99364434 | — | — | — | — | — | — | False | False | False |
| binary | random_forest | 46 | 5 | n=5, indicative only | 0.99349807 | — | — | — | — | — | — | False | False | False |
| binary | std_cnn | 42 | 5 | n=5, indicative only | 0.97775677 | 11 | 5 | 20 | False | True | 0.42178277 | False | False | False |
| binary | std_cnn | 43 | 5 | n=5, indicative only | 0.9826659 | 16 | 10 | 20 | False | True | 0.095491503 | False | False | False |
| binary | std_cnn | 44 | 5 | n=5, indicative only | 0.98339941 | 20 | 17 | 20 | True | False | 0 | False | False | False |
| binary | std_cnn | 45 | 5 | n=5, indicative only | 0.97052123 | 10 | 4 | 20 | False | True | 0.5 | False | False | False |
| binary | std_cnn | 46 | 5 | n=5, indicative only | 0.98394241 | 14 | 8 | 20 | False | True | 0.20610737 | False | False | False |
| binary | xgboost | 42 | 5 | n=5, indicative only | 0.99337056 | — | — | — | — | — | — | False | False | False |
| binary | xgboost | 43 | 5 | n=5, indicative only | 0.99337056 | — | — | — | — | — | — | False | False | False |
| binary | xgboost | 44 | 5 | n=5, indicative only | 0.99337056 | — | — | — | — | — | — | False | False | False |
| binary | xgboost | 45 | 5 | n=5, indicative only | 0.99337056 | — | — | — | — | — | — | False | False | False |
| binary | xgboost | 46 | 5 | n=5, indicative only | 0.99337056 | — | — | — | — | — | — | False | False | False |
| multiclass | balanced | 42 | 5 | n=5, indicative only | 0.82980444 | 18 | 12 | 30 | False | True | 0.3454915 | False | False | False |
| multiclass | balanced | 43 | 5 | n=5, indicative only | 0.84829328 | 22 | 16 | 30 | False | True | 0.1654347 | False | False | False |
| multiclass | balanced | 44 | 5 | n=5, indicative only | 0.78251075 | 18 | 12 | 30 | False | True | 0.3454915 | False | False | False |
| multiclass | balanced | 45 | 5 | n=5, indicative only | 0.78941638 | 10 | 4 | 30 | False | True | 0.75 | False | False | False |
| multiclass | balanced | 46 | 5 | n=5, indicative only | 0.78253869 | 17 | 11 | 30 | False | True | 0.39604415 | False | False | False |
| multiclass | bilstm | 42 | 5 | n=5, indicative only | 0.770176 | 30 | 27 | 30 | True | False | 0 | False | False | False |
| multiclass | bilstm | 43 | 5 | n=5, indicative only | 0.71700132 | 30 | 29 | 30 | True | False | 0 | False | False | False |
| multiclass | bilstm | 44 | 5 | n=5, indicative only | 0.73230761 | 30 | 26 | 30 | True | False | 0 | False | False | False |
| multiclass | bilstm | 45 | 5 | n=5, indicative only | 0.69105597 | 19 | 13 | 30 | False | True | 0.29663168 | False | False | False |
| multiclass | bilstm | 46 | 5 | n=5, indicative only | 0.7305102 | 18 | 12 | 30 | False | True | 0.3454915 | False | False | False |
| multiclass | exlightnet | 42 | 5 | n=5, indicative only | 0.82479633 | 11 | 5 | 30 | False | True | 0.70336832 | False | False | False |
| multiclass | exlightnet | 43 | 5 | n=5, indicative only | 0.85579121 | 17 | 11 | 30 | False | True | 0.39604415 | False | False | False |
| multiclass | exlightnet | 44 | 5 | n=5, indicative only | 0.87341925 | 27 | 21 | 30 | False | True | 0.024471742 | False | False | False |
| multiclass | exlightnet | 45 | 5 | n=5, indicative only | 0.83127373 | 15 | 9 | 30 | False | True | 0.5 | False | False | False |
| multiclass | exlightnet | 46 | 5 | n=5, indicative only | 0.84771124 | 15 | 9 | 30 | False | True | 0.5 | False | False | False |
| multiclass | exlightnet_w8 | 42 | 5 | n=5, indicative only | 0.82530718 | — | — | — | — | — | — | False | False | False |
| multiclass | exlightnet_w8 | 43 | 5 | n=5, indicative only | 0.85161392 | — | — | — | — | — | — | False | False | False |
| multiclass | exlightnet_w8 | 44 | 5 | n=5, indicative only | 0.8693313 | — | — | — | — | — | — | False | False | False |
| multiclass | exlightnet_w8 | 45 | 5 | n=5, indicative only | 0.83759483 | — | — | — | — | — | — | False | False | False |
| multiclass | exlightnet_w8 | 46 | 5 | n=5, indicative only | 0.85010201 | — | — | — | — | — | — | False | False | False |
| multiclass | mlp | 42 | 5 | n=5, indicative only | 0.8354536 | 19 | 13 | 30 | False | True | 0.29663168 | False | False | False |
| multiclass | mlp | 43 | 5 | n=5, indicative only | 0.8326996 | 16 | 10 | 30 | False | True | 0.44773577 | False | False | False |
| multiclass | mlp | 44 | 5 | n=5, indicative only | 0.85326081 | 23 | 17 | 30 | False | True | 0.12842759 | False | False | False |
| multiclass | mlp | 45 | 5 | n=5, indicative only | 0.827413 | 19 | 13 | 30 | False | True | 0.29663168 | False | False | False |
| multiclass | mlp | 46 | 5 | n=5, indicative only | 0.82916817 | 17 | 11 | 30 | False | True | 0.39604415 | False | False | False |
| multiclass | no_sampler | 42 | 5 | n=5, indicative only | 0.86503481 | 20 | 14 | 30 | False | True | 0.25 | False | False | False |
| multiclass | no_sampler | 43 | 5 | n=5, indicative only | 0.87928656 | 24 | 18 | 30 | False | True | 0.095491503 | False | False | False |
| multiclass | no_sampler | 44 | 5 | n=5, indicative only | 0.78705005 | 18 | 12 | 30 | False | True | 0.3454915 | False | False | False |
| multiclass | no_sampler | 45 | 5 | n=5, indicative only | 0.80876308 | 21 | 15 | 30 | False | True | 0.20610737 | False | False | False |
| multiclass | no_sampler | 46 | 5 | n=5, indicative only | 0.88306683 | 30 | 30 | 30 | True | False | 0 | False | False | False |
| multiclass | no_se | 42 | 5 | n=5, indicative only | 0.82196499 | 11 | 5 | 30 | False | True | 0.70336832 | False | False | False |
| multiclass | no_se | 43 | 5 | n=5, indicative only | 0.79507715 | 11 | 5 | 30 | False | True | 0.70336832 | False | False | False |
| multiclass | no_se | 44 | 5 | n=5, indicative only | 0.82351026 | 19 | 13 | 30 | False | True | 0.29663168 | False | False | False |
| multiclass | no_se | 45 | 5 | n=5, indicative only | 0.83951525 | 15 | 9 | 30 | False | True | 0.5 | False | False | False |
| multiclass | no_se | 46 | 5 | n=5, indicative only | 0.85500417 | 28 | 22 | 30 | False | True | 0.0109262 | False | False | False |
| multiclass | random_forest | 42 | 5 | n=5, indicative only | 0.85919185 | — | — | — | — | — | — | False | False | False |
| multiclass | random_forest | 43 | 5 | n=5, indicative only | 0.85909176 | — | — | — | — | — | — | False | False | False |
| multiclass | random_forest | 44 | 5 | n=5, indicative only | 0.86449912 | — | — | — | — | — | — | False | False | False |
| multiclass | random_forest | 45 | 5 | n=5, indicative only | 0.8583254 | — | — | — | — | — | — | False | False | False |
| multiclass | random_forest | 46 | 5 | n=5, indicative only | 0.85689399 | — | — | — | — | — | — | False | False | False |
| multiclass | standard_conv | 42 | 5 | n=5, indicative only | 0.8321795 | 14 | 8 | 30 | False | True | 0.55226423 | False | False | False |
| multiclass | standard_conv | 43 | 5 | n=5, indicative only | 0.86516868 | 18 | 12 | 30 | False | True | 0.3454915 | False | False | False |
| multiclass | standard_conv | 44 | 5 | n=5, indicative only | 0.88235446 | 19 | 13 | 30 | False | True | 0.29663168 | False | False | False |
| multiclass | standard_conv | 45 | 5 | n=5, indicative only | 0.87007782 | 14 | 8 | 30 | False | True | 0.55226423 | False | False | False |
| multiclass | standard_conv | 46 | 5 | n=5, indicative only | 0.85961502 | 20 | 14 | 30 | False | True | 0.25 | False | False | False |
| multiclass | std_cnn | 42 | 5 | n=5, indicative only | 0.83240737 | 29 | 23 | 30 | False | True | 0.0027390523 | True | False | True |
| multiclass | std_cnn | 43 | 5 | n=5, indicative only | 0.6385524 | 8 | 2 | 30 | False | True | 0.8345653 | True | True | True |
| multiclass | std_cnn | 44 | 5 | n=5, indicative only | 0.79932523 | 26 | 20 | 30 | False | True | 0.043227271 | True | False | True |
| multiclass | std_cnn | 45 | 5 | n=5, indicative only | 0.6227003 | 7 | 1 | 30 | False | True | 0.87157241 | True | True | True |
| multiclass | std_cnn | 46 | 5 | n=5, indicative only | 0.6383931 | 9 | 3 | 30 | False | True | 0.79389263 | True | True | True |
| multiclass | top20 | 42 | 5 | n=5, indicative only | 0.5798038 | 14 | 8 | 30 | False | True | 0.55226423 | False | False | False |
| multiclass | top20 | 43 | 5 | n=5, indicative only | 0.58671769 | 15 | 9 | 30 | False | True | 0.5 | False | False | False |
| multiclass | top20 | 44 | 5 | n=5, indicative only | 0.57573883 | 10 | 4 | 30 | False | True | 0.75 | False | False | False |
| multiclass | top20 | 45 | 5 | n=5, indicative only | 0.5573912 | 9 | 3 | 30 | False | True | 0.79389263 | True | True | False |
| multiclass | top20 | 46 | 5 | n=5, indicative only | 0.58962555 | 12 | 6 | 30 | False | True | 0.6545085 | False | False | False |
| multiclass | xgboost | 42 | 5 | n=5, indicative only | 0.83266412 | — | — | — | — | — | — | False | False | False |
| multiclass | xgboost | 43 | 5 | n=5, indicative only | 0.83266412 | — | — | — | — | — | — | False | False | False |
| multiclass | xgboost | 44 | 5 | n=5, indicative only | 0.83266412 | — | — | — | — | — | — | False | False | False |
| multiclass | xgboost | 45 | 5 | n=5, indicative only | 0.83266412 | — | — | — | — | — | — | False | False | False |
| multiclass | xgboost | 46 | 5 | n=5, indicative only | 0.83266412 | — | — | — | — | — | — | False | False | False |

### Epoch/F1 correlations

Spearman correlations, including the pooled within-model z-scored convolutional family, are descriptive only, not causal. The stopping/cosine explanation remains a hypothesis for the intervention study in Stage B.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/correlations.csv

| task | model | n | sample_label | spearman_rho | p_value | interpretation | members |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 5 | n=5, indicative only | 0.70710678 | 0.18169011 | descriptive only, not causal | bilstm |
| binary | exlightnet | 5 | n=5, indicative only | 0.70710678 | 0.18169011 | descriptive only, not causal | exlightnet |
| binary | exlightnet_w8 | 0 | n=0 | — | — | descriptive only, not causal | exlightnet_w8 |
| binary | mlp | 5 | n=5, indicative only | 0.4472136 | 0.45018486 | descriptive only, not causal | mlp |
| binary | random_forest | 0 | n=0 | — | — | descriptive only, not causal | random_forest |
| binary | std_cnn | 5 | n=5, indicative only | 0.7 | 0.1881204 | descriptive only, not causal | std_cnn |
| binary | xgboost | 0 | n=0 | — | — | descriptive only, not causal | xgboost |
| multiclass | balanced | 5 | n=5, indicative only | 0.46169026 | 0.43376616 | descriptive only, not causal | balanced |
| multiclass | bilstm | 5 | n=5, indicative only | 0.4472136 | 0.45018486 | descriptive only, not causal | bilstm |
| multiclass | exlightnet | 5 | n=5, indicative only | 0.97467943 | 0.0048182305 | descriptive only, not causal | exlightnet |
| multiclass | exlightnet_w8 | 0 | n=0 | — | — | descriptive only, not causal | exlightnet_w8 |
| multiclass | mlp | 5 | n=5, indicative only | 0.46169026 | 0.43376616 | descriptive only, not causal | mlp |
| multiclass | no_sampler | 5 | n=5, indicative only | 0.9 | 0.037386073 | descriptive only, not causal | no_sampler |
| multiclass | no_se | 5 | n=5, indicative only | 0.8720816 | 0.053854218 | descriptive only, not causal | no_se |
| multiclass | random_forest | 0 | n=0 | — | — | descriptive only, not causal | random_forest |
| multiclass | standard_conv | 5 | n=5, indicative only | 0.15389675 | 0.80482882 | descriptive only, not causal | standard_conv |
| multiclass | std_cnn | 5 | n=5, indicative only | 0.9 | 0.037386073 | descriptive only, not causal | std_cnn |
| multiclass | top20 | 5 | n=5, indicative only | 0.7 | 0.1881204 | descriptive only, not causal | top20 |
| multiclass | xgboost | 0 | n=0 | — | — | descriptive only, not causal | xgboost |
| binary | pooled_conv_family | 10 | n=10, pooled descriptive only | 0.61279929 | 0.059610196 | descriptive only, not causal; within-model z scores | exlightnet,std_cnn |
| multiclass | pooled_conv_family | 20 | n=20, pooled descriptive only | 0.79337649 | 2.9951953e-05 | descriptive only, not causal; within-model z scores | exlightnet,no_se,standard_conv,std_cnn |

### Binary epoch caps

Counts describe the recorded binary epoch cap and best-validation epoch; no uncapped local experiment was run.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/binary_caps.csv

| task | model | n_runs | n_epoch_observed | n_hit_cap | n_best_at_cap | sample_label |
| --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 5 | 5 | 4 | 0 | n=5, indicative only |
| binary | exlightnet | 5 | 5 | 4 | 2 | n=5, indicative only |
| binary | exlightnet_w8 | 5 | 0 | — | — | n=5, indicative only |
| binary | mlp | 5 | 5 | 1 | 0 | n=5, indicative only |
| binary | random_forest | 5 | 0 | — | — | n=5, indicative only |
| binary | std_cnn | 5 | 5 | 1 | 0 | n=5, indicative only |
| binary | xgboost | 5 | 0 | — | — | n=5, indicative only |

### Per-seed observations

All persisted seed observations are shown; missing values remain unavailable and no best seed is headlined.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

Source: artifacts/stats_cpu_v1/per_seed_macro_f1.csv

| task | model | seed | n_seeds | sample_label | macro_f1 | accuracy | epochs_run | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | bilstm | 42 | 5 | n=5, indicative only | 0.96101575 | 0.98538012 | 20 | 16 |
| binary | bilstm | 43 | 5 | n=5, indicative only | 0.9559369 | 0.98326836 | 14 | 8 |
| binary | bilstm | 44 | 5 | n=5, indicative only | 0.96322661 | 0.98630063 | 20 | 18 |
| binary | bilstm | 45 | 5 | n=5, indicative only | 0.97706849 | 0.9916071 | 20 | 19 |
| binary | bilstm | 46 | 5 | n=5, indicative only | 0.95877397 | 0.98440546 | 20 | 18 |
| binary | exlightnet | 42 | 5 | n=5, indicative only | 0.98805448 | 0.99566818 | 20 | 20 |
| binary | exlightnet | 43 | 5 | n=5, indicative only | 0.9872617 | 0.9953433 | 20 | 17 |
| binary | exlightnet | 44 | 5 | n=5, indicative only | 0.98570625 | 0.99480182 | 17 | 11 |
| binary | exlightnet | 45 | 5 | n=5, indicative only | 0.98709836 | 0.99528915 | 20 | 18 |
| binary | exlightnet | 46 | 5 | n=5, indicative only | 0.98722552 | 0.9953433 | 20 | 20 |
| binary | exlightnet_w8 | 42 | 5 | n=5, indicative only | 0.98776167 | 0.99555989 | — | — |
| binary | exlightnet_w8 | 43 | 5 | n=5, indicative only | 0.98755207 | 0.99545159 | — | — |
| binary | exlightnet_w8 | 44 | 5 | n=5, indicative only | 0.98541539 | 0.99469352 | — | — |
| binary | exlightnet_w8 | 45 | 5 | n=5, indicative only | 0.98711054 | 0.99528915 | — | — |
| binary | exlightnet_w8 | 46 | 5 | n=5, indicative only | 0.98737107 | 0.99539744 | — | — |
| binary | mlp | 42 | 5 | n=5, indicative only | 0.9871227 | 0.99528915 | 20 | 16 |
| binary | mlp | 43 | 5 | n=5, indicative only | 0.98726771 | 0.9953433 | 17 | 11 |
| binary | mlp | 44 | 5 | n=5, indicative only | 0.98586522 | 0.99485597 | 17 | 11 |
| binary | mlp | 45 | 5 | n=5, indicative only | 0.98730363 | 0.9953433 | 19 | 13 |
| binary | mlp | 46 | 5 | n=5, indicative only | 0.98694081 | 0.995235 | 17 | 11 |
| binary | random_forest | 42 | 5 | n=5, indicative only | 0.99379069 | 0.9977258 | — | — |
| binary | random_forest | 43 | 5 | n=5, indicative only | 0.99393996 | 0.99777994 | — | — |
| binary | random_forest | 44 | 5 | n=5, indicative only | 0.99379069 | 0.9977258 | — | — |
| binary | random_forest | 45 | 5 | n=5, indicative only | 0.99364434 | 0.99767165 | — | — |
| binary | random_forest | 46 | 5 | n=5, indicative only | 0.99349807 | 0.9976175 | — | — |
| binary | std_cnn | 42 | 5 | n=5, indicative only | 0.97775677 | 0.99193199 | 11 | 5 |
| binary | std_cnn | 43 | 5 | n=5, indicative only | 0.9826659 | 0.99366472 | 16 | 10 |
| binary | std_cnn | 44 | 5 | n=5, indicative only | 0.98339941 | 0.99388131 | 20 | 17 |
| binary | std_cnn | 45 | 5 | n=5, indicative only | 0.97052123 | 0.98900801 | 10 | 4 |
| binary | std_cnn | 46 | 5 | n=5, indicative only | 0.98394241 | 0.99415205 | 14 | 8 |
| binary | xgboost | 42 | 5 | n=5, indicative only | 0.99337056 | 0.99756335 | — | — |
| binary | xgboost | 43 | 5 | n=5, indicative only | 0.99337056 | 0.99756335 | — | — |
| binary | xgboost | 44 | 5 | n=5, indicative only | 0.99337056 | 0.99756335 | — | — |
| binary | xgboost | 45 | 5 | n=5, indicative only | 0.99337056 | 0.99756335 | — | — |
| binary | xgboost | 46 | 5 | n=5, indicative only | 0.99337056 | 0.99756335 | — | — |
| multiclass | balanced | 42 | 5 | n=5, indicative only | 0.82980444 | 0.97195148 | 18 | 12 |
| multiclass | balanced | 43 | 5 | n=5, indicative only | 0.84829328 | 0.97649989 | 22 | 16 |
| multiclass | balanced | 44 | 5 | n=5, indicative only | 0.78251075 | 0.96621183 | 18 | 12 |
| multiclass | balanced | 45 | 5 | n=5, indicative only | 0.78941638 | 0.97411739 | 10 | 4 |
| multiclass | balanced | 46 | 5 | n=5, indicative only | 0.78253869 | 0.97298029 | 17 | 11 |
| multiclass | bilstm | 42 | 5 | n=5, indicative only | 0.770176 | 0.97130171 | 30 | 27 |
| multiclass | bilstm | 43 | 5 | n=5, indicative only | 0.71700132 | 0.97059779 | 30 | 29 |
| multiclass | bilstm | 44 | 5 | n=5, indicative only | 0.73230761 | 0.96821529 | 30 | 26 |
| multiclass | bilstm | 45 | 5 | n=5, indicative only | 0.69105597 | 0.96756552 | 19 | 13 |
| multiclass | bilstm | 46 | 5 | n=5, indicative only | 0.7305102 | 0.96539961 | 18 | 12 |
| multiclass | exlightnet | 42 | 5 | n=5, indicative only | 0.82479633 | 0.97649989 | 11 | 5 |
| multiclass | exlightnet | 43 | 5 | n=5, indicative only | 0.85579121 | 0.9763916 | 17 | 11 |
| multiclass | exlightnet | 44 | 5 | n=5, indicative only | 0.87341925 | 0.9785575 | 27 | 21 |
| multiclass | exlightnet | 45 | 5 | n=5, indicative only | 0.83127373 | 0.97579597 | 15 | 9 |
| multiclass | exlightnet | 46 | 5 | n=5, indicative only | 0.84771124 | 0.97687893 | 15 | 9 |
| multiclass | exlightnet_w8 | 42 | 5 | n=5, indicative only | 0.82530718 | 0.97704137 | — | — |
| multiclass | exlightnet_w8 | 43 | 5 | n=5, indicative only | 0.85161392 | 0.97595841 | — | — |
| multiclass | exlightnet_w8 | 44 | 5 | n=5, indicative only | 0.8693313 | 0.9785575 | — | — |
| multiclass | exlightnet_w8 | 45 | 5 | n=5, indicative only | 0.83759483 | 0.97595841 | — | — |
| multiclass | exlightnet_w8 | 46 | 5 | n=5, indicative only | 0.85010201 | 0.97704137 | — | — |
| multiclass | mlp | 42 | 5 | n=5, indicative only | 0.8354536 | 0.97698722 | 19 | 13 |
| multiclass | mlp | 43 | 5 | n=5, indicative only | 0.8326996 | 0.97655404 | 16 | 10 |
| multiclass | mlp | 44 | 5 | n=5, indicative only | 0.85326081 | 0.9787741 | 23 | 17 |
| multiclass | mlp | 45 | 5 | n=5, indicative only | 0.827413 | 0.97617501 | 19 | 13 |
| multiclass | mlp | 46 | 5 | n=5, indicative only | 0.82916817 | 0.97655404 | 17 | 11 |
| multiclass | no_sampler | 42 | 5 | n=5, indicative only | 0.86503481 | 0.97807018 | 20 | 14 |
| multiclass | no_sampler | 43 | 5 | n=5, indicative only | 0.87928656 | 0.97871995 | 24 | 18 |
| multiclass | no_sampler | 44 | 5 | n=5, indicative only | 0.78705005 | 0.97747455 | 18 | 12 |
| multiclass | no_sampler | 45 | 5 | n=5, indicative only | 0.80876308 | 0.97796188 | 21 | 15 |
| multiclass | no_sampler | 46 | 5 | n=5, indicative only | 0.88306683 | 0.98018194 | 30 | 30 |
| multiclass | no_se | 42 | 5 | n=5, indicative only | 0.82196499 | 0.97373836 | 11 | 5 |
| multiclass | no_se | 43 | 5 | n=5, indicative only | 0.79507715 | 0.97368421 | 11 | 5 |
| multiclass | no_se | 44 | 5 | n=5, indicative only | 0.82351026 | 0.97384665 | 19 | 13 |
| multiclass | no_se | 45 | 5 | n=5, indicative only | 0.83951525 | 0.97655404 | 15 | 9 |
| multiclass | no_se | 46 | 5 | n=5, indicative only | 0.85500417 | 0.97601256 | 28 | 22 |
| multiclass | random_forest | 42 | 5 | n=5, indicative only | 0.85919185 | 0.98023608 | — | — |
| multiclass | random_forest | 43 | 5 | n=5, indicative only | 0.85909176 | 0.98029023 | — | — |
| multiclass | random_forest | 44 | 5 | n=5, indicative only | 0.86449912 | 0.97985705 | — | — |
| multiclass | random_forest | 45 | 5 | n=5, indicative only | 0.8583254 | 0.98018194 | — | — |
| multiclass | random_forest | 46 | 5 | n=5, indicative only | 0.85689399 | 0.98007364 | — | — |
| multiclass | standard_conv | 42 | 5 | n=5, indicative only | 0.8321795 | 0.97687893 | 14 | 8 |
| multiclass | standard_conv | 43 | 5 | n=5, indicative only | 0.86516868 | 0.97612086 | 18 | 12 |
| multiclass | standard_conv | 44 | 5 | n=5, indicative only | 0.88235446 | 0.9786658 | 19 | 13 |
| multiclass | standard_conv | 45 | 5 | n=5, indicative only | 0.87007782 | 0.97666233 | 14 | 8 |
| multiclass | standard_conv | 46 | 5 | n=5, indicative only | 0.85961502 | 0.97882824 | 20 | 14 |
| multiclass | std_cnn | 42 | 5 | n=5, indicative only | 0.83240737 | 0.97725796 | 29 | 23 |
| multiclass | std_cnn | 43 | 5 | n=5, indicative only | 0.6385524 | 0.95251245 | 8 | 2 |
| multiclass | std_cnn | 44 | 5 | n=5, indicative only | 0.79932523 | 0.96875677 | 26 | 20 |
| multiclass | std_cnn | 45 | 5 | n=5, indicative only | 0.6227003 | 0.94417371 | 7 | 1 |
| multiclass | std_cnn | 46 | 5 | n=5, indicative only | 0.6383931 | 0.92587178 | 9 | 3 |
| multiclass | top20 | 42 | 5 | n=5, indicative only | 0.5798038 | 0.94395712 | 14 | 8 |
| multiclass | top20 | 43 | 5 | n=5, indicative only | 0.58671769 | 0.94579814 | 15 | 9 |
| multiclass | top20 | 44 | 5 | n=5, indicative only | 0.57573883 | 0.94731427 | 10 | 4 |
| multiclass | top20 | 45 | 5 | n=5, indicative only | 0.5573912 | 0.94363223 | 9 | 3 |
| multiclass | top20 | 46 | 5 | n=5, indicative only | 0.58962555 | 0.94644791 | 12 | 6 |
| multiclass | xgboost | 42 | 5 | n=5, indicative only | 0.83266412 | 0.97969461 | — | — |
| multiclass | xgboost | 43 | 5 | n=5, indicative only | 0.83266412 | 0.97969461 | — | — |
| multiclass | xgboost | 44 | 5 | n=5, indicative only | 0.83266412 | 0.97969461 | — | — |
| multiclass | xgboost | 45 | 5 | n=5, indicative only | 0.83266412 | 0.97969461 | — | — |
| multiclass | xgboost | 46 | 5 | n=5, indicative only | 0.83266412 | 0.97969461 | — | — |

## Limitations

The class-stratified flow split is fixed. The five canonical seeds vary initialization and sampling, not the split; n=5, indicative only. Related attack-run flows can cross partitions because session/campaign identifiers are absent. A clustering PROXY is not campaign-disjoint validation.

Tiny multiclass test supports make macro-F1 comparisons noisy: metasploit_brute_force_ssh (test n=5), nmap_fin_scan (test n=4). Training seeds do not measure split or campaign variability.

The relation between epochs run and macro-F1 is descriptive. Early stopping combined with a cosine schedule that has not reached its horizon is a hypothesis, to be tested by [Stage B stopping-rule interventions](docs/KAGGLE_RUNBOOK.md); correlation alone does not establish causality.

The persisted training-dynamics table flags std_cnn instability (early best-validation epochs or large seed SD). This baseline cannot establish that ExLightNet is superior. Zero observed seed variance, including XGBoost under the paper configuration, is flagged in the generated summary; its paired test remains a test of the paired differences.

PyTorch eager batch-one latency includes framework dispatch overhead. It is not pure compute cost or hardware-independent performance. The dispatch fraction is NOT MEASURED. Exported/frozen-model latency is NOT RUN; no such benchmark is inferred from the eager timings.

Supplementary GPU accuracy/stopping/split studies are separate from cpu_v1; CUDA was NOT EXECUTED here. CPU/GPU device-equivalence views are descriptive and cannot pool campaigns or assert a pass/fail equivalence gate.

## Implementation and assumptions

Implemented: real-data loading and hard gates; fixed class-stratified partition; train-fitted preprocessing/MI; exact ExLightNet; four alpha alternatives and validation-only sweep; shared deterministic training; five external baselines; metrics and evaluation receipts; five architectural/training/feature ablations; per-channel W8 exports and float evaluation; single-thread timing; standalone NumPy inference; SHAP with measured fallback/additivity/deletion/local/stability checks; notebook generation and evidence-based reporting. Full-mode code defaults to five seeds (42–46) for both tasks and all ablations. Optional class-clustering leakage proxy is implemented behind `--leakage-proxy`; it is never labelled campaign-disjoint.

Not implemented: optional knowledge distillation, true INT8 arithmetic/ONNX Runtime, packet capture/Zeek extraction, or multi-process throughput measurement. Full-mode execution status above is separate from implementation status. No Pi hardware was available.

Preprocessing: lower-case labels/columns and category strings; benign Wipro prefixes collapse to wipro_bulb; raw export index is removed. Numeric infinities become NaN, training medians fill them, and an entirely missing training column uses zero then is dropped as constant. proto/service one-hot vocabularies come from train only; unseen categories produce all-zero blocks. Constant pruning precedes binary-target MI. MI uses sklearn n_neighbors=3, random_state=42, discrete one-hot columns, continuous numeric columns and stable tie ordering. Selected features are sorted by descending computed MI; z-scores use population (ddof=0) training SD. The top-20 ablation takes the first 20 of the same ranking/scaler.

Alpha formulas: 1/n_c; 1/sqrt(n_c); (1-beta)/(1-beta**n_c), beta=0.999; and all ones. Each vector is normalized to arithmetic mean one, then focal loss averages per-example alpha*(1-p_y)**2*(-log p_y), with no weighted-loss denominator. Sampling uses per-sample n_c**(-1/2); the fully balanced ablation uses n_c**(-1). Counts and frequencies differ by a common factor that cancels in both normalized schemes.

Baselines: MLP 40–128–64–K, ReLU, biased Linear; one-layer bidirectional LSTM, input size 1, 32 hidden units/direction, final directional hidden states concatenated into a 64-wide head; Std-CNN widths 64/128/256, k=3, no padding, bias-free convolutions, BN/ReLU after each, global average pooling then biased Linear. Standard-convolution ablation replaces each depthwise+pointwise pair by a single no-padding k=3 convolution with the same channel widths and retains SE. Counts are measured, not fitted to paper totals. RF uses 300 balanced trees; XGBoost uses 400 depth-8 hist trees, balanced per-sample weights, CPU, one thread, otherwise pinned-library defaults (learning_rate=0.3). Tree parameter count is null because trainable neural parameters are not a meaningful equivalent; serialized bytes are measured. Official xgboost-cpu supplies the xgboost API without GPU libraries.

Quantisation: Conv1d/Linear weights only, per-output-channel scale max(abs(W))/127 clamped at 1e-12, nearest-even rounding, clipping [-127,127]. BN state and all biases stay FP32; the non-inference BN counter is stored FP32 and restored to its integer dtype. Scales/rest/metadata contribute to the measured compressed NPZ size. Dequantisation occurs once at load; compute remains FP32 and no INT8 speedup is claimed. Q1 binary approximately-zero tolerance is declared as absolute macro-F1 delta <=0.001; multiclass drop <=0.01. Failed Q1 artifacts and metrics remain reviewable but their downstream deployment/stability branch stops.

The test set receives one final probability pass per trained checkpoint (and one per derived W8 artifact). Receipts prevent silent reevaluation after interruption. Confusion matrices, tables and plots reuse cached predictions. Required SHAP and deletion queries are final interpretability analysis, with no model/epoch/alpha/threshold selection. DeepExplainer first checks additivity on two inputs to detect unsupported operators before full-set work; failure invokes the same declared GradientExplainer fallback, without changing background, explanation sample or GradientExplainer settings. Separate fast and full checkpoints have separate receipts.

## Paper inconsistencies and limits

The literal per-class sampler formula conflicts with the per-flow repetition narrative; the brief's per-sample interpretation is used. The sampler-disabled paper ablation does not show an accuracy gain from sampling. Figure 3 stage counts do not sum to its stated total. The Std-CNN descriptions conflict on widths and parameter counts. The same-width standard-convolution ablation count is reported from our graph, not adjusted to match the paper. One-hot candidate accounting is consistent (94 - 1 constant = 93). The prominent SHAP rate/header features occur in the lower half of the MI ranking: marginal binary MI and model-specific multi-class attribution answer different questions, and the actual overlap/ranks are reported rather than forced.

The paper's multiclass headline is its best of three seeds; its quoted three-seed mean/SD is 0.835 +/- 0.023. Our summaries use all requested seeds and never select a best seed. Some paper precision/recall columns track weighted values/accuracy; our precision/recall are explicitly macro-averaged as requested. SHAP efficiency is classwise in logit space, not a sum across class outputs. Weight-only quantisation does not establish integer-compute acceleration. The paper's four-core Pi throughput is an extrapolation, and its cache explanation confuses instruction/data caches; neither assertion is reproduced as measured evidence here.

The flat corpus has no campaign/session IDs. Random flow splitting can put related flows in multiple partitions and optimistically bias metrics and relative rankings. A class-clustering PROXY cannot establish campaign independence. The generated test-support tables quantify rare-class uncertainty; initialization/sampler seeds do not estimate dataset/campaign variability. No clinical, encrypted-traffic, adversarial or cross-dataset generalisation claim follows from these runs.

## Reproduction commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
pytest -q
python -m exlightnet.run_all --mode fast
python -m exlightnet.run_all --mode full --alpha-only --skip-benchmark
python scripts/run_parallel.py --reuse-alpha
# The pipeline runs five single-thread workers, then exactly these final stages:
# python -m exlightnet.run_all --mode full --aggregate-only --skip-benchmark
# python -m exlightnet.run_all --mode full --benchmark-only
python scripts/make_notebook.py
EXLIGHTNET_RUN_MODE=full python scripts/execute_notebook.py --evidence-dir artifacts/stats_cpu_v1/verification/notebook_run
python scripts/package_deliverables.py
# Optional, separately, after full results: python -m exlightnet.run_all --mode full --leakage-proxy --skip-benchmark
# On a Raspberry Pi with NumPy only:
python scripts/bench_pi.py results/full/multiclass/exlightnet/42/model_int8.npz
```

Git is available in the current workspace. Early reproduction entries describe a historical unavailable Git mount; phase evidence is recorded in docs/progress.md. Network setup initially required escalation and retries; the CPU PyTorch wheel was obtained from the official S3 origin when the CDN timed out. Source data came from UCI ucimlrepo id 942. The archive excludes the dataset, virtual environment, cached feature arrays and every *.pkl tree checkpoint. Tree checkpoints remain local for benchmarking; an extracted archive displays their cached metrics but cannot re-benchmark absent model files.

Primary implementation references: [UCI RT-IoT2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022), [SHAP DeepExplainer](https://shap.readthedocs.io/en/stable/generated/shap.DeepExplainer.html), [official XGBoost CPU package](https://pypi.org/project/xgboost-cpu/2.1.4/).
