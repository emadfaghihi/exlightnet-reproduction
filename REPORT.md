# ExLightNet reproduction report

Generated from saved evidence: 2026-11-17T10:08:46.160813+03:30

The supplied 31-page PDF was fully extracted with `pdftotext -layout` and read before implementation. The supplied brief is authoritative where it conflicts with the paper. All measured results below come from code. Quoted reference metrics appear only in `src/exlightnet/paper.py::PAPER_REPORTED` and comparison tables.

## Execution status

Pytest verification from results/pytest.xml. Third-party SHAP/Matplotlib deprecation warnings are retained in results/pytest.log.

| Tests | Failures | Errors | Skipped |
| --- | --- | --- | --- |
| 65 | 0 | 0 | 0 |

Notebook verification history (wall-clock time includes initial live training; later invocations may reuse caches):

| Mode | Seconds | Executed cells | Error outputs | Caches present at start |
| --- | --- | --- | --- | --- |
| fast | 695.022 | 14 | 0 | initial live run |
| fast | 9.408 | 14 | 0 | True |
| fast | 9.029 | 14 | 0 | True |
| full | 3.081 | 14 | 0 | True |

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

Standard deviation is the sample SD (ddof=1); a single seed has no estimated SD. Fast mode is a demonstration with reduced training and SHAP, not a five-seed reproduction. Seed 42 is always shown separately; no best seed is selected.

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

### Ablations

| Variant | Paired seeds | Delta macro-F1 | Paired delta SD | Full-model SD | Exceeds full SD | Exceeds paired SD |
| --- | --- | --- | --- | --- | --- | --- |
| balanced | 42 | -0.10622442071210342 | — | — | — | — |
| no_sampler | 42 | 0.06793822057763987 | — | — | — | — |
| no_se | 42 | 0.02552872443144394 | — | — | — | — |
| standard_conv | 42 | 0.032450016049767716 | — | — | — | — |
| top20 | 42 | -0.1785177351358968 | — | — | — | — |

Differences are compared with both full-model seed SD and paired difference SD when multiple seeds exist. This is descriptive variation, not a statistical significance claim. Full mode uses five seeds; fast mode uses one and cannot establish whether a difference exceeds seed variation.

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

DeepExplainer failed its additivity check and the prescribed GradientExplainer fallback was used. The residuals above show that these reduced-sample attributions do not establish exact additive efficiency; treat the fast explanation figures as approximate diagnostics.

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

Standard deviation is the sample SD (ddof=1); a single seed has no estimated SD. Fast mode is a demonstration with reduced training and SHAP, not a five-seed reproduction. Seed 42 is always shown separately; no best seed is selected.

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

### Ablations

| Variant | Paired seeds | Delta macro-F1 | Paired delta SD | Full-model SD | Exceeds full SD | Exceeds paired SD |
| --- | --- | --- | --- | --- | --- | --- |
| balanced | 42, 43, 44, 45, 46 | -0.040085642036770786 | 0.03972045997260938 | 0.01946329900844065 | True | True |
| no_sampler | 42, 43, 44, 45, 46 | -0.001958084780446656 | 0.053311308250923284 | 0.01946329900844065 | False | False |
| no_se | 42, 43, 44, 45, 46 | -0.01958398847665439 | 0.033123131160565186 | 0.01946329900844065 | True | False |
| standard_conv | 42, 43, 44, 45, 46 | 0.015280744934418111 | 0.013249887279081399 | 0.01946329900844065 | False | True |
| top20 | 42, 43, 44, 45, 46 | -0.268742936751889 | 0.019631361678655503 | 0.01946329900844065 | True | True |

Differences are compared with both full-model seed SD and paired difference SD when multiple seeds exist. This is descriptive variation, not a statistical significance claim. Full mode uses five seeds; fast mode uses one and cannot establish whether a difference exceeds seed variation.

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

DeepExplainer failed its additivity check and the prescribed GradientExplainer fallback was used. The residuals above show that these reduced-sample attributions do not establish exact additive efficiency; treat the fast explanation figures as approximate diagnostics.

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

The flat corpus has no campaign/session IDs. Random flow splitting can put related flows in multiple partitions and optimistically bias metrics and relative rankings. A class-clustering PROXY cannot establish campaign independence. Rare-class supports (four FIN scans, five SSH brute-force examples in test) impose substantial uncertainty; five training seeds do not estimate dataset/campaign variability. No clinical, encrypted-traffic, adversarial or cross-dataset generalisation claim follows from these runs.

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
EXLIGHTNET_RUN_MODE=full python scripts/execute_notebook.py
python scripts/package_deliverables.py
# Optional, separately, after full results: python -m exlightnet.run_all --mode full --leakage-proxy --skip-benchmark
# On a Raspberry Pi with NumPy only:
python scripts/bench_pi.py results/full/multiclass/exlightnet/42/model_int8.npz
```

This workspace contains an empty read-only .git mount rather than a usable Git repository. Per-phase commits were unavailable; phase evidence is recorded in docs/progress.md. Network setup initially required escalation and retries; the CPU PyTorch wheel was obtained from the official S3 origin when the CDN timed out. Source data came from UCI ucimlrepo id 942. The archive excludes the dataset, virtual environment, cached feature arrays and every *.pkl tree checkpoint. Tree checkpoints remain local for benchmarking; an extracted archive displays their cached metrics but cannot re-benchmark absent model files.

Primary implementation references: [UCI RT-IoT2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022), [SHAP DeepExplainer](https://shap.readthedocs.io/en/stable/generated/shap.DeepExplainer.html), [official XGBoost CPU package](https://pypi.org/project/xgboost-cpu/2.1.4/).
