# Contributing

Use Python 3.11 and the pinned dependencies in `requirements.txt`. Installation,
experiment commands and evaluation constraints are documented in [README.md](README.md).

## Branches

`main` is the publication branch. Start focused work from its current tip:

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/describe-the-change
```

Use `feat/`, `fix/`, `docs/`, `refactor/`, `test/` or `chore/` followed by a short
lowercase, hyphen-separated description. Send changes through a pull request to
`main`. Keep the branch limited to one purpose.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>(<optional scope>): <imperative summary>
```

Write English summaries, aim for 72 characters or fewer, and omit the final
period. Use a body to explain the reason, experimental implications and
validation when those details help a reviewer.

| Type | Use |
| --- | --- |
| `feat` | New functionality |
| `fix` | Correctness or behavior repair |
| `refactor` | Code restructuring without a behavior change |
| `perf` | Performance improvement |
| `test` | Test coverage or test infrastructure |
| `docs` | Documentation and reports |
| `build` | Packaging or dependency changes |
| `ci` | GitHub Actions changes |
| `chore` | Repository maintenance or recorded experiment artifacts |

Useful scopes include `data`, `model`, `train`, `eval`, `quant`, `explain`,
`report`, `notebook`, `results` and `repo`.

Examples:

```text
feat(train): add parallel seed workers and cached aggregation
fix(eval): clear stale timings after a failed benchmark
test(quant): verify INT8 artifact size accounting
docs(report): document five-seed results and measured limitations
chore(results): record the full five-seed campaign evidence
```

Stage explicit paths or use `git add -p` to separate unrelated edits. Review
`git diff --cached --stat` and `git diff --cached` before committing. Each commit
should have one clear purpose; include the related tests with a behavior change.
Mark breaking changes with `!` and explain them in a `BREAKING CHANGE:` footer.
Do not rewrite commits that other people may already have fetched.

An optional local commit template and conservative Git defaults:

```bash
git config --local commit.template .gitmessage
git config --local pull.ff only
git config --local push.default simple
git config --local fetch.prune true
```

## Files and experimental evidence

Track source, tests, pinned requirements, documentation, the executed notebook
and deliberately recorded experiment evidence. Keep fast and full results
separate, and preserve the original bytes of measured CSVs.

The ignore rules keep the dataset, prepared feature-array cache, `*.pkl` tree
checkpoints, virtual environments, credentials, process locks and local agent
settings out of Git. Small FP32/INT8 artifacts, predictions, summaries and audit
records remain versioned so reports can be inspected from a clone. Never use
`git add -f` to bypass these rules for a dataset or large checkpoint. Use a
separate artifact store or a release asset if checkpoint distribution is needed.

Tree metrics can be inspected without the excluded checkpoints. Re-benchmarking
tree models requires the original local `model.pkl` files; Git does not restore
them. Prepared feature arrays can be rebuilt from the genuine dataset using the
documented preprocessing path. Do not rerun training or final test evaluation
just to refresh a Git commit or change recorded evidence to make a gate pass.

## Verification and pull requests

Activate the installed environment and run:

```bash
python -m pytest -q
git diff --check
git diff --cached --check
```

The full local suite includes recovery from the real dataset. GitHub Actions
runs the dataset-independent suite; it deselects
`test_extracted_archive_can_rebuild_missing_array_cache`, which requires the
local CSV and prepared corpus. `test_real_split_if_present` skips itself when
the dataset is absent. CI does not train the full five-seed campaign.

Describe the problem, the resulting behavior, validation actually run and any
remaining limitations in the pull request. If measured results change, identify
the mode, seeds, recipe and evidence paths. Keep notebook outputs consistent
with the report. Rebuild `deliverables_manifest.json` with
`python scripts/package_deliverables.py` whenever its packaged inputs change;
the generated ZIP stays local.

## Push workflow

```bash
git fetch --prune origin
git status --short --branch
git diff --cached --check
git push --dry-run origin HEAD
git push -u origin HEAD
```

Push only the intended branch. Resolve a rejected push by fetching and inspecting
the remote changes. Do not force-push `main`, use `--mirror`, or push all local
branches. After review and passing CI, merge the pull request into `main`.

The original pre-publication `master` branch is retained locally as a historical
backup. Its initial commit contains a large tree checkpoint. It is intentionally
excluded from the clean `main` history and must not be pushed or merged into
`main`.
