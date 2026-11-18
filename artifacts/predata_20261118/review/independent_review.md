# Independent review of the 2026-11-18 amendments

Reviewed the working tree relative to `cbbea40` before the controller’s fix pass. The scope covered the pre-data protocol, scheduling, archive aggregation, final CPU analysis, presentation, publication checks, handoff packaging, and their tests.

## Findings

There were no critical findings.

| Severity | Location | Finding and recommended action |
| --- | --- | --- |
| Important | `src/exlightnet/campaign.py:361`, `:400` | A CUDA session running all models includes its GPU name in CPU-tree fingerprints. Trees-only sessions on the same host use a null GPU name. Switching selectors can train and evaluate the same tree twice and create conflicting completed records. Derive new identities from run-specific metadata. Preserve and resolve compatible legacy tree fingerprints and receipts, including pending branches, without querying a GPU in a trees-only session. Test the all-CUDA to trees-only transition using stubs. |
| Important | `src/exlightnet/campaign.py:448`, `:460` | Pending-tree recovery compares the entire record, including the requested device, warning ledger, and effective session settings. A change from CPU to auto in a trees-only resume can stop a scientifically identical run. Validate its fingerprint and immutable inputs and keep its original recorded provenance. |
| Important | `src/exlightnet/final_presentation.py:24` | Completed presentation evidence is checked only for filenames listed in its manifest. A missing manifest entry allows an unverified plot to be displayed or referenced. Require the exact declared filenames and their checksums, including the auxiliary outputs, before rendering. |
| Important | `src/exlightnet/report.py:298`, `src/exlightnet/notebook.py:164`, `scripts/make_notebook.py:60` | The final-analysis presentations omit the saved AdaptiveMaxPool1d warning. State that the measured absolute residuals summarize 259 flows by 12 output logits, and that separate output-logit summaries were not retained. Surface the archived warning. The reproduction notebook’s causal-hypothesis wording also needs the dated, never-causal stopping rule and its primary and secondary endpoints. The supplementary statistics report already shows both. |
| Minor | `scripts/package_deliverables.py:83`, `:92`, `:147` | Restrict supplied handoff metadata to tested and not-tested statements so callers cannot replace the measured byte count. Render role labels for plain-text inputs. Document the refresh command with its verification JSON because a default packaging run updates hosted source but skips an existing handoff. |

## Verified strengths

- The preserved original declaration has the same Git blob as the baseline, `c4ec13b67b1414ce05a42a8e7ec92cb034289f20`. Six dated amendments accompany it.
- The stopping analysis uses matched seed pairs, paired t intervals, a fixed-seed 2,000-replicate bootstrap and an inclusive SD-ratio limit with a strict upper-CI limit. Endpoint disagreement and undefined ratios have explicit outcomes. The completeness logic enumerates whole expected study blocks and B1 reference identities before inference.
- All 12 recorded final-analysis output hashes matched. The 200 background positions matched the earlier explanation in order. The 259 selected samples included all five and four flows from the two smallest classes. The deletion cache contained one 41-step SHAP curve and 20 random curves per class. All 240 random orders were valid permutations of 40 features.
- Twenty focused pure tests passed in 2.08 seconds, exit 0. The checks covered protocol preservation, stubbed hosted-command wiring, both stopping endpoints, every B1/B2/B3 plan prefix, baseline plan multisets, rare-class sampling, class weighting and the publication scan. This review used persisted evidence. No training, SHAP execution, CUDA, benchmark or new model inference was needed.

## Declined to judge

None within the assigned review scope. The controller owns post-fix verification and the final generated archives.

## Assessment

Ready after fixes. The tree identity and pending-state issues affect resumability and the evaluation-once guarantee. The measured CPU analysis is intact. Its warning and residual interpretation can be reported from persisted evidence.
