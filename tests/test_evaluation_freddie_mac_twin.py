"""Freddie Mac 2007 semi-synthetic twin: real covariates, simulated default with planted effects.

The real 2007 file is a registered download that is never committed (data/freddie_mac is gitignored),
so tests that need it skip without it; the missing-file error and the config/registry checks always run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import statsmodels.api as sm

from rootcause.evaluation import scms
from rootcause.evaluation.scm import SCM
from rootcause.evaluation.scms import (
    FREDDIE_FIRST_TIME_LOGIT,
    FREDDIE_LEVER_LOGITS,
    FREDDIE_N_ROWS,
    FREDDIE_SCORE_LOGIT,
    FREDDIE_SEED,
    SCM_REGISTRY,
    freddie_mac_semi_synthetic_scm,
)
from rootcause.pipeline import runner

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_semi_synthetic_freddie_mac_data as generator  # noqa: E402

DOMAIN = "freddie_mac"
REAL_PATH = REPO_ROOT / "data" / "freddie_mac" / "loans_2007.csv"
TWIN_PATH = REPO_ROOT / "data" / "freddie_mac" / "loans_2007_semi_synthetic.csv"
needs_data = pytest.mark.skipif(not REAL_PATH.exists(), reason="data/freddie_mac/loans_2007.csv is not present (registered download)")
needs_twin = pytest.mark.skipif(not (REAL_PATH.exists() and TWIN_PATH.exists()), reason="the Freddie Mac 2007 files are not present")


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    return pd.read_csv(REAL_PATH)


# --- always run ---------------------------------------------------------------------------


def test_missing_real_file_is_a_clear_error_not_a_crash_deep_in_pandas(monkeypatch, tmp_path):
    monkeypatch.setattr(scms, "FREDDIE_DATA_PATH", tmp_path / "loans_2007.csv")
    scms._real_freddie_loans.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="prepare_freddie_mac"):
            freddie_mac_semi_synthetic_scm()
    finally:
        scms._real_freddie_loans.cache_clear()


def test_twin_is_registered_and_the_config_lists_it(config_loader):
    assert SCM_REGISTRY[(DOMAIN, "semi_synthetic")] is freddie_mac_semi_synthetic_scm
    assert (DOMAIN, "real_2007") not in SCM_REGISTRY  # real data has no truth
    cfg = config_loader.get_domain(DOMAIN).extra
    assert "semi_synthetic" in runner.available_datasets(cfg)
    assert cfg["ingestion"]["default_dataset"] == "real_2007"  # the twin never becomes the default
    assert str(runner.resolve_data_path(cfg, "semi_synthetic")).endswith("loans_2007_semi_synthetic.csv")


def test_planted_effects_are_documented_constants_of_the_expected_sign():
    assert set(FREDDIE_LEVER_LOGITS) == {"interest_rate", "ltv", "dti"}  # exactly the config's levers
    assert all(v > 0 for v in FREDDIE_LEVER_LOGITS.values())
    assert FREDDIE_SCORE_LOGIT < 0 and FREDDIE_FIRST_TIME_LOGIT < 0
    assert (FREDDIE_N_ROWS, FREDDIE_SEED) == (31_780, 42)


# --- need the real 2007 file --------------------------------------------------------------


@needs_data
def test_covariates_are_real_loans_and_only_the_outcome_is_simulated(real_df):
    scm = freddie_mac_semi_synthetic_scm()
    assert scm.graph_known is False and "row" not in scm.observed
    assert {child for _, child in scm.edges()} == {"default"}  # only the outcome's parents are modelled
    big = scm.sample(200_000, 1)
    assert list(big.columns) == [c for c in real_df.columns if c not in ("loan_id", "loan_sequence")]
    for column in ("credit_score", "ltv", "dti", "interest_rate", "first_time_buyer"):
        assert big[column].mean() == pytest.approx(real_df[column].mean(), rel=0.01), column
    corr = lambda d: d["interest_rate"].corr(d["credit_score"])  # noqa: E731
    assert corr(big) == pytest.approx(corr(real_df), abs=0.02)  # the real correlation between rate and score survives
    sample = scm.sample(3000, 3)
    covariates = [c for c in sample.columns if c != "default"]
    merged = sample[covariates].merge(real_df[covariates].drop_duplicates(), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()  # no covariate value was invented


@needs_data
def test_simulated_default_rate_matches_the_real_file(real_df):
    big = freddie_mac_semi_synthetic_scm().sample(400_000, 2)
    assert big["default"].mean() == pytest.approx(real_df["default"].mean(), abs=0.005)


@needs_data
def test_planted_lever_effects_are_what_a_logistic_fit_recovers():
    big = freddie_mac_semi_synthetic_scm().sample(300_000, 4)
    design = pd.DataFrame({c: big[c] for c in ("credit_score", "first_time_buyer", *FREDDIE_LEVER_LOGITS)}).astype(float)
    design["investment"] = (big["occupancy"] == "investment").astype(float)
    design["cash_out"] = (big["purpose"] == "refi_cash_out").astype(float)
    fit = sm.Logit(big["default"], sm.add_constant(design)).fit(disp=0)
    for lever, planted in FREDDIE_LEVER_LOGITS.items():
        assert fit.params[lever] == pytest.approx(planted, abs=0.2 * planted), lever
    assert fit.params["credit_score"] == pytest.approx(FREDDIE_SCORE_LOGIT, abs=0.001)


@needs_data
def test_true_effects_are_positive_and_do_leaves_covariates_alone():
    scm = freddie_mac_semi_synthetic_scm()
    effects = {lever: scm.true_effect(lever, "default", 1.0, n=400_000) for lever in FREDDIE_LEVER_LOGITS}
    assert all(e > 0 for e in effects.values())
    assert effects["interest_rate"] > effects["dti"] > effects["ltv"]  # per unit, as the planted logits imply
    assert effects["interest_rate"] == pytest.approx(0.9 * 0.13, rel=0.25)  # logit coef x about p(1-p)

    base, moved = scm.sample(20_000, 5), scm.sample(20_000, 5, shift={"interest_rate": 1.0})
    other = base.columns.drop(["interest_rate", "default"])
    pd.testing.assert_frame_equal(base[other], moved[other])  # a lever shift changes only itself and the outcome
    assert moved["default"].mean() > base["default"].mean()


# --- need the generated twin file ---------------------------------------------------------


@needs_twin
def test_local_twin_file_is_exactly_what_the_scm_generates(real_df):
    twin = pd.read_csv(TWIN_PATH)
    generated = generator.generate()
    pd.testing.assert_frame_equal(generated, twin)
    assert len(twin) == len(real_df) and "loan_sequence" not in twin.columns
    assert twin["loan_id"].is_unique
