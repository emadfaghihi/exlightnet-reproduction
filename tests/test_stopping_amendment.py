"""Verify the predeclared stopping analysis with pure synthetic seed scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exlightnet.study_stats import _hypothesis


PROTOCOL = Path(__file__).resolve().parents[1]/"docs/statistics_protocol.json"
AMENDMENT = next(a for a in json.loads(PROTOCOL.read_text())["amendments"] if a["id"]=="PA-20261118-1")
PAPER = .25 + np.arange(15)/1024


def results(paper, fixed_last, fixed_best=None):
    if fixed_best is None:
        fixed_best = fixed_last
    records=[]
    for seed, a, b, c in zip(range(42,57), paper, fixed_last, fixed_best):
        for arm, best, last in [("paper",a,.99),("fixed30",c,b)]:
            records.append(dict(study="B3",task="multiclass",model="exlightnet",arm=arm,seed=seed,
                checkpoints={"best":{"metrics":{"macro_f1":best}},"last":{"metrics":{"macro_f1":last}}}))
    return records


def hypothesis(paper, fixed_last, fixed_best=None):
    return _hypothesis(results(paper,fixed_last,fixed_best),pd.DataFrame(),AMENDMENT).set_index("endpoint_role")


@pytest.mark.parametrize("scale,expected",[(.75,True),(.751,False)])
def test_ratio_at_threshold_is_inclusive(scale,expected):
    table=hypothesis(PAPER,.5+scale*(PAPER-.25))
    row=table.loc["PRIMARY"]
    assert row.sd_ratio==pytest.approx(scale)
    assert row.sd_ratio_ci_high==pytest.approx(scale)
    assert bool(row.smaller_sd)==expected
    assert row.overall_decision==("conditions supported" if expected else "not supported")


def test_upper_ratio_interval_threshold_is_strict():
    paper=PAPER.copy()
    paper[-1]=.5
    row=hypothesis(paper,PAPER+.25).loc["PRIMARY"]
    assert row.sd_ratio<.75
    assert row.sd_ratio_ci_high==1
    assert row.mean_difference_ci_low>0
    assert not row.smaller_sd
    assert row.endpoint_decision=="not supported"


def test_ratio_percentile_ci_uses_paired_seed_indices_and_fixed_settings():
    paper=.25+np.arange(15)/512
    fixed=.5+20*(paper-.25)**2
    row=hypothesis(paper,fixed).loc["PRIMARY"]
    assert row.sd_ratio==pytest.approx(.5670569179342435)
    assert row.sd_ratio_ci_low==pytest.approx(.44720542)
    assert row.sd_ratio_ci_high==pytest.approx(.68012552)
    assert row.bootstrap_replicates==row.defined_replicates==2000
    assert row.bootstrap_random_seed==42


def test_negative_mean_interval_cannot_support_higher_mean():
    row=hypothesis(PAPER,.125+.5*(PAPER-.25)).loc["PRIMARY"]
    assert row.smaller_sd
    assert row.mean_difference_ci_high<0
    assert not row.higher_mean
    assert row.endpoint_decision==row.overall_decision=="not supported"


def test_zero_paper_sd_makes_ratio_undefined():
    row=hypothesis(np.full(15,.3),np.full(15,.7)).loc["PRIMARY"]
    assert row.paper_sd==0
    assert np.isnan(row.sd_ratio)
    assert row.defined_replicates==0
    assert row.endpoint_decision==row.overall_decision=="inconclusive"


def test_undefined_ratio_draws_cannot_support_the_hypothesis():
    paper=np.full(15,.3)
    paper[-1]=.4
    row=hypothesis(paper,np.full(15,.7)).loc["PRIMARY"]
    assert row.fixed30_sd==row.sd_ratio==0
    assert row.higher_mean
    assert 0<row.defined_replicates<2000
    assert np.isnan(row.sd_ratio_ci_high)
    assert row.endpoint_decision==row.overall_decision=="inconclusive"


def test_zero_fixed_sd_has_defined_zero_ratio_when_paper_varies():
    row=hypothesis(PAPER,np.full(15,.7)).loc["PRIMARY"]
    assert row.fixed30_sd==row.sd_ratio==row.sd_ratio_ci_high==0
    assert row.endpoint_decision==row.overall_decision=="conditions supported"


@pytest.mark.parametrize("swap",[False,True])
def test_both_directions_of_endpoint_disagreement_are_inconclusive(swap):
    fixed_supported=.5+.5*(PAPER-.25)
    fixed_unsupported=.125+.5*(PAPER-.25)
    last,best=(fixed_unsupported,fixed_supported) if swap else (fixed_supported,fixed_unsupported)
    table=hypothesis(PAPER,last,best)
    assert list(table.experimental_arm)==["fixed30","fixed30"]
    assert list(table.experimental_checkpoint)==["last","best"]
    assert list(table.reference_arm)==["paper","paper"]
    assert list(table.reference_checkpoint)==["best","best"]
    assert list(table.fixed30_mean)==pytest.approx([last.mean(),best.mean()])
    assert table.loc["SECONDARY" if swap else "PRIMARY"].endpoint_decision=="conditions supported"
    assert table.loc["PRIMARY" if swap else "SECONDARY"].endpoint_decision=="not supported"
    assert set(table.overall_decision)=={"inconclusive"}


def test_missing_seed_is_pending_and_records_b1_descriptive_correlation():
    records=results(PAPER,.5+.5*(PAPER-.25))
    records=[r for r in records if not (r["seed"]==56 and r["arm"]=="paper")]
    correlations=pd.DataFrame([dict(task="multiclass",model="exlightnet",n=12,spearman_rho=.2,ci_low=-.2,ci_high=.5)])
    table=_hypothesis(records,correlations,AMENDMENT)
    assert set(table.endpoint_role)=={"PRIMARY","SECONDARY"}
    assert set(table.n_pairs)=={14}
    assert set(table.missing_seeds)=={"56"}
    assert set(table.endpoint_decision)==set(table.overall_decision)=={"PENDING"}
    assert table.mean_difference_ci_low.isna().all()
    assert set(table.b1_n)=={12}
    assert set(table.b1_spearman_rho)=={.2}
    assert set(table.b1_ci_low)=={-.2}
    assert set(table.b1_ci_high)=={.5}
