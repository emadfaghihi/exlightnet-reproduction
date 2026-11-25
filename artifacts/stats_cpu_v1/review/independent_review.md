# Independent scientific and hosted-runner review

Review date: 2026-11-17. Verdict: **changes requested**. No Critical findings; four Important findings and one Minor finding below. Stage A mathematics and canonical evidence separation are substantially implemented; the hosted recovery and import-validation contracts are not yet complete.

This is a single independent review of the current working-tree implementation, the supplied brief/design/plan, agent reports, and the recorded smoke evidence. Packaging changes owned by the controller are excluded. No training, model inference, benchmarks, process actions, commits, source edits, or additional agents were used. Only this review artifact was written. CUDA and real hosted studies were **NOT RUN**.

## Important findings

### R1. Tree recovery ZIPs lose the state required to finalize a completed evaluation

Locations: `src/exlightnet/campaign.py:122`, `src/exlightnet/campaign.py:343`, `src/exlightnet/campaign.py:345`, `src/exlightnet/campaign.py:371`.

Trigger: a tree has finished its single evaluation and published a complete receipt, but the process exits before publishing `record.json`. This is a recoverable boundary locally because `tree_in_progress.pkl` still exists. The exit ZIP excludes every PKL, so a fresh hosted session restores the complete receipt without the fitted tree. The runner requires that pickle whenever any receipt exists and raises before it can finalize the record. Trees alias their last checkpoint to best, so there is no remaining test inference that actually requires the fitted model in this case.

Effect: an already evaluated tree becomes permanently stopped after ZIP recovery even though all its final predictions and metrics are available. This breaks the fresh-session resumability contract and, with R2, prevents later planned runs from proceeding. Do not solve this by admitting PKL files into the delivery or deleting receipts. Persist sufficient run metadata before evaluation and finalize a tree with fully verified completed evidence without fitting or predicting again.

Evidence: a pure temporary-file package/restore check produced `receipt_exists=true` and `required_tree_state_exists=false`. The relevant guard then necessarily raises. No pickle was deserialized in this check.

Regression suggestion: inject interruption immediately after a tree's complete receipt and before its run record; package, restore into an empty directory, and finalize using existing evidence. Make fit/predict stubs fail if called and verify receipt bytes remain unchanged. Also retain the existing claimed-receipt rejection.

### R2. One stopped run aborts every later independent run in the study

Locations: `src/exlightnet/campaign.py:304`, `src/exlightnet/campaign.py:343`, `src/exlightnet/campaign.py:441`.

Trigger: the next planned identity has a claimed receipt, missing/corrupted completed evidence, a nonfinite training loss, or another run-local failure. These exceptions escape the whole plan loop; its only handler is the outer `finally` that packages and releases the lock.

Effect: the command terminates without recording a durable stopped-run summary or attempting later independent seeds/models. Restarting the same study reaches the same identity and fails again. Because the CLI has no production seed/model filter or stopped-branch mechanism, the remaining study cannot make progress without editing evidence or source. The project explicitly requires preserving failed branches while continuing independent work. Receipt refusal is correct; its scope is too broad.

Regression suggestion: a two-plan, no-training fixture with the first identity carrying a claimed receipt and the second using a completed cache or stub. Assert that the first branch is recorded as stopped, its receipt is untouched, the independent plan is reached, and the overall result explicitly reports stopped branches. Keep global prepared-data/provenance errors fatal; do not broadly swallow arbitrary exceptions.

### R3. Import validation accepts completed runs that violate the declared stopping rule

Locations: `src/exlightnet/study_stats.py:177`, `src/exlightnet/study_stats.py:192`, `src/exlightnet/study_stats.py:505`.

Trigger: a completed paper-rule B1 record contains only two epochs, best epoch 1, patience 6, and max_epochs 20. `_validate_history` checks the epoch cap, best-validation choice and two special arms, but does not verify that ordinary early stopping was legally reached. Similar gaps apply to B2/B4/B5 and the stopping point of the minimum-epoch arm.

Effect: a truncated or inconsistent run can pass the verified import path, enter paired statistics, and count toward a complete study. There is no scientific reason for this example to terminate after two epochs. Several existing synthetic archive fixtures use exactly this impossible terminal history, so passing those tests does not cover recipe completion.

Evidence: invoking `_validate_history` on the existing `evidence()` fixture returned successfully with `{epochs_run: 2, best_epoch: 1, patience: 6, max_epochs: 20}`. This was a pure validation call, not training.

Regression suggestion: replay the validation improvement/stale logic from the declared rule and reject both premature termination and continuation past the first legal stop; accept the cap and valid patience/minimum-epoch boundaries. Include a probe that must contain all three epochs. Update archive fixtures to legal histories instead of weakening this check.

### R4. Different GPU hardware can silently enter one model's seed vector

Locations: `src/exlightnet/study_stats.py:337`, `src/exlightnet/study_stats.py:341`; related runner identity at `src/exlightnet/campaign.py:279` and `src/exlightnet/campaign.py:323`.

Trigger: one hosted session supplies seed 42 on one GPU model and another session supplies seed 43 on a different GPU model. Both records have `device="cuda"`; their different `libraries.gpu_name` values are correctly included in their individual run fingerprints, but the aggregator checks only the literal device string. The same issue permits different recorded CUDA/cuDNN/PyTorch environments across one vector without any mismatch decision.

Effect: the no-multiple-devices rule is satisfied syntactically but violated scientifically: a vector silently pools distinct hardware/runtime conditions and displays only `cuda`. Kaggle hardware can differ between restored sessions. A run fingerprint distinguishes artifacts but does not establish comparability of an aggregate. The runner also looks up a probe by `probe_cuda.json` alone, so an old hardware probe can be reused for projections on changed hardware.

Evidence: a pure archive-fixture import with two same-model seeds and `libraries.gpu_name` values `GPU A`/`GPU B` returned two accepted records and `device_label="cuda"`.

Regression suggestion: define and validate the actual neural execution-environment identity before pooling, including GPU model and relevant runtime versions; reject changes or require a separate campaign/root. Validate a reused probe against that identity. CPU-tree provenance should use its relevant CPU/software fields, not inherit a requirement that its host's unused GPU be identical. Test distinct GPU models under the same `cuda` string and a consistent-hardware success case.

## Minor finding

### R5. Warnings before an interruption are absent from the resumed final record

Locations: `src/exlightnet/campaign.py:354`, `src/exlightnet/campaign.py:408`, `src/exlightnet/train.py:192`.

The warning list exists only inside the current process's `catch_warnings` context and is copied into the final record. Epoch recovery state contains history/RNG/optimizer but no warning ledger. After interruption, warnings from completed earlier epochs are lost; a resumed run may therefore report an empty warnings list despite a recorded earlier nondeterministic operation. The final record also does not explicitly store the deterministic/TF32/CUBLAS settings required by the design, although `configure_device` sets the intended flags.

Regression suggestion: persist warnings with the epoch recovery boundary (and their epoch/phase), merge them on resume, and record the effective deterministic settings. A synthetic warning before a checkpoint can test this without a GPU or optimization. Do not claim the CUDA warning behavior was executed here.

## Checks and positive observations

- Paired differences use matched seeds, sample SD, a two-sided Student t test, a df=n-1 interval and signed dz. Constant model variance is distinguished from constant paired differences. Holm adjustment preserves input order and unavailable values. No Wilcoxon path was found.
- Joint-cell bootstrap preserves paired flow covariance and uniformly selects one matched seed per replicate. The stated estimand correctly describes a randomly selected run plus flow uncertainty, not the seed-mean interval. B2 pairing verifies the split and prepared payload for each pair.
- Sensitivity retains false-positive contributions from excluded-class flows. Canonical per-class tables currently use the original CPU mapping order; supplementary ingest maps sorted JSON class keys through prepared class names explicitly.
- Study plan counts and references agree with the declared B1/B2/B3/B4/B5 structure. B3 and B4 references are expanded from B1 without an extra training plan. B5 selection uses validation means, requires complete equal seed sets, and uses a deterministic lower-LR tie break.
- Mixed campaign imports fail with no override. The explicit CUDA-neural/CPU-tree exception is restricted to `gpu_v2` and separate device columns are present. The cross-campaign equivalence table has differences/CIs but no p-value or pass/fail column.
- Prepared exports preserve canonical split42 arrays and use train-only fitting for other splits. The offline loader validates recipe/payload/class/partition invariants. The historical `TrainConfig`, `fit_model`, sampler, and CPU prediction implementation remain unchanged; the new loop is separate.
- Source inspection found appropriate CUDA optimizer-state restoration through `optimizer.load_state_dict` after placing the model on the device, plus CPU/sampler and CUDA RNG restoration. This is not a CUDA execution result.
- ZIP import validates paths/types and hashes without extracting or loading model pickles. Claimed receipts are refused, and complete evidence is checked before reuse. The known CPU best/last tensor-alias fix is present.
- Publication consumes new statistical tables rather than old ablation JSON. Missing supplementary studies remain PENDING. No evidence of numerical tuning to the paper was found in the reviewed path.

## Verification limits and evidence

The recorded CPU smoke `artifacts/stats_cpu_v1/smoke/control.json` reports two MLP runs/four epochs, epoch-one interruption, fresh-directory ZIP restore, and a cache-only restart with unchanged receipts. It covers neural CPU epoch recovery; it does not cover R1's tree evaluation boundary or CUDA execution.

The existing first-suite log reports **170 passed, 1 failed**; its failure concerns the newly introduced synthetic-training policy guard. The controller reported updating that registry and a focused eight-test pass. This review did not rerun the suite and does not assert a final all-green suite.

The review's additional pure checks used one-thread `.venv/bin/python`, temporary synthetic artifacts, and the existing archive fixture builders. They completed with exit 0 in 0.56 seconds and confirmed R1/R3/R4. No optimizer step, model prediction, real-corpus read, benchmark or raw-evidence mutation occurred. Source inspection supports R2/R5; their suggested regression tests were not executed in this review.

Spec verdict: Stage A is substantially conformant; Stage B requires fixes R1-R4 before claiming its recovery/provenance gates complete. Quality verdict: the core separation and statistical implementation are sound, but the hosted runner needs stronger branch isolation and terminal-state/environment validation. R5 should be addressed with the other resume work.

## Reviewed source hashes

- `src/exlightnet/campaign.py`: `47fcb168bff9a4683bbeb6b0d7795d3358e42173eab40a4f31aa3866c60feb18`
- `src/exlightnet/train.py`: `e50e3ac4a98efa83f789c3de8e6d1edbd2602f99ec486f60f68b4bf254bd1602`
- `src/exlightnet/study_stats.py`: `8faccc8831e51917b68fae332a2dfb1cdbb5d9653062d155423c15de6116bc32`
- `src/exlightnet/stats.py`: `9dd39592c382daba6d7eabe2f07472300f307e62921c3928981bfb8ddaaab1bf`
- `src/exlightnet/device.py`: `9486d6b2b7452d346b915eb7b4837f0c2750d0c62b023d1d2032f389f884d8ec`
- `src/exlightnet/prepared.py`: `547de09053017e77fc9566070ab3f728509f9c7bf66b9446d7ef68400d50ca2b`
