"""freddie_mac domain: the prepare script's outcome and censoring rules, the config's consistency
with the encoded columns, and end-to-end runs.

The raw Freddie Mac files are a registered download that is never committed, so the rules are
tested on a small fake raw pair in the real file layout (always runs). Tests that pin counts in
the real prepared files skip when data/freddie_mac/loans_<year>.csv is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rootcause.evaluation import harness
from rootcause.evaluation.scms import SCM_REGISTRY
from rootcause.pipeline import preprocessing, runner

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_freddie_mac as prepare  # noqa: E402

DOMAIN = "freddie_mac"
LEVERS = ("interest_rate", "ltv", "dti")
YEARS = ("2007", "2008", "2010", "2011")
# (loans, defaults, loans with DTI not available) in the prepared files, from the sample vintages
REAL_COUNTS = {"2007": (31780, 4936, 678), "2008": (25531, 3013, 542),
               "2010": (30466, 671, 10335), "2011": (33210, 538, 12174)}
NEW_YEARS = ("2016", "2019", "2022")
# (loans, 90+ day defaults, relief-adjusted defaults, DTI not available) in the prepared files
NEW_COUNTS = {"2016": (40139, 550, 276, 2124), "2019": (19654, 2473, 194, 24), "2022": (42230, 1485, 707, 3)}


# --- a fake raw pair in the real layout ---------------------------------------------------


def orig_row(seq: str, **over: str) -> list[str]:
    """One origination row: 31 pipe-file columns as strings, keyed like prepare.ORIG."""
    base = {"credit_score": "750", "first_payment": "200702", "first_time": "N", "mi_pct": "0", "units": "1",
            "occupancy": "P", "dti": "35", "upb": "200000", "ltv": "80", "rate": "6.5", "channel": "R",
            "property_type": "SF", "loan_seq": seq, "purpose": "P", "term_months": "360", "borrowers": "2"}
    base.update(over)
    row = [""] * prepare.ORIG_COLUMNS
    for key, position in prepare.ORIG.items():
        row[position] = base[key]
    return row


def perf_rows(seq: str, ages: range, status: str = "00", zero_balance: str = "", status_at: dict | None = None):
    """Performance rows for one loan; `zero_balance` is set on the last row."""
    rows = []
    for age in ages:
        st = (status_at or {}).get(age, status)
        zb = zero_balance if age == ages[-1] else ""
        rows.append({0: seq, 3: st, 4: str(age), 8: zb})
    return rows


def frames(orig_rows: list[list[str]], perf: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(orig_rows, dtype=str), pd.DataFrame(perf, dtype=str)


def prepared(orig_rows, perf):
    return prepare.prepare(*frames(orig_rows, perf))


def synthetic_raw(n: int = 900, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`n` loans with every level of every categorical present, a default rate that rises with the
    rate, and a share of loans that prepay before month 36."""
    rng = np.random.default_rng(seed)
    orig, perf = [], []
    channels, purposes = list("RBCT"), list("PCN")
    props, occs = ["SF", "PU", "CO", "MH", "CP"], list("PIS")
    def pick(levels: list, i: int):
        """Every level shows up in the first rows; after that, independent random draws."""
        return levels[i] if i < len(levels) else levels[int(rng.integers(len(levels)))]

    for i in range(n):
        seq = f"F07Q{pick([1, 2, 3, 4], i)}{i:07d}"
        rate = round(float(rng.uniform(4.5, 8.5)), 3)
        relief = rng.random() < 0.2
        orig.append(orig_row(
            seq, rate=str(rate), credit_score=str(int(rng.integers(600, 820))), ltv=str(int(rng.integers(40, 100))),
            dti="999" if relief else str(int(rng.integers(15, 50))), upb=str(int(rng.integers(50, 500)) * 1000),
            channel=pick(channels, i), purpose=pick(purposes, i), property_type=pick(props, i),
            occupancy=pick(occs, i), first_time="Y" if rng.random() < 0.15 else "N",
            mi_pct=str(int(rng.integers(0, 3)) * 12),
            units=str(1 + int(rng.random() < 0.2)), borrowers=str(1 + int(rng.random() < 0.5)), term_months=str(int(rng.choice([180, 240, 360])))))
        p_default = 1 / (1 + np.exp(-((rate - 8.0) * 1.2 - 1.5)))
        fate = rng.random()
        if fate < p_default:
            perf.extend(perf_rows(seq, range(1, 30), status_at={int(rng.integers(8, 30)): "03"}))
        elif fate < p_default + 0.35 * (1 - p_default):
            perf.extend(perf_rows(seq, range(1, int(rng.integers(6, 30))), zero_balance="01"))
        else:
            perf.extend(perf_rows(seq, range(1, 41)))
    return frames(orig, perf)


@pytest.fixture(scope="module")
def fake_loans() -> pd.DataFrame:
    loans, _ = prepare.prepare(*synthetic_raw())
    return loans


@pytest.fixture(scope="module")
def fm_config(config_loader):
    return config_loader.get_domain(DOMAIN).extra


def real_path(year: str) -> Path:
    return REPO_ROOT / "data" / "freddie_mac" / f"loans_{year}.csv"


def new_path(year: str, adjusted: bool = False) -> Path:
    suffix = "_relief_adjusted" if adjusted else ""
    return REPO_ROOT / "data" / "freddie_mac" / f"loans_{year}{suffix}.csv"


needs_new = pytest.mark.skipif(
    not all(new_path(y, a).exists() for y in NEW_YEARS for a in (False, True)),
    reason="Freddie Mac 2016/2019/2022 loans files not prepared (registered download)")
needs_real = pytest.mark.skipif(not all(real_path(y).exists() for y in YEARS),
                                reason="Freddie Mac loans_<year>.csv not prepared (registered download)")


# --- the outcome and censoring rules ------------------------------------------------------


def test_default_means_90_days_late_reo_or_a_charge_off_within_36_months():
    a, b, c, d, e = ("F07Q10000001", "F07Q10000002", "F07Q10000003", "F07Q10000004", "F07Q10000005")
    orig = [orig_row(s) for s in (a, b, c, d, e)]
    perf = (
        perf_rows(a, range(1, 41), status_at={20: "03"})                    # 90 days late at 20: default
        + perf_rows(b, range(1, 41), status_at={10: "01", 11: "02"})        # 30-89 days only: fine
        + perf_rows(c, range(1, 41), status_at={30: "RA"})                  # REO acquisition: default
        + perf_rows(d, range(1, 41), status_at={37: "05"})                  # 90+ but AFTER month 36: fine
        + perf_rows(e, range(1, 30), zero_balance="03")                     # short sale/charge-off: default
    )
    loans, report = prepared(orig, perf)
    assert dict(zip(loans["loan_sequence"], loans["default"])) == {a: 1, b: 0, c: 1, d: 0, e: 1}
    assert report["n_default"] == 3 and report["n_loans"] == 5


def test_default_at_exactly_month_36_counts_and_month_37_does_not():
    orig = [orig_row("F07Q10000001"), orig_row("F07Q10000002")]
    perf = perf_rows("F07Q10000001", range(1, 50), status_at={36: "03"}) + perf_rows(
        "F07Q10000002", range(1, 50), status_at={37: "03"})
    loans, _ = prepared(orig, perf)
    assert loans.set_index("loan_sequence")["default"].to_dict() == {"F07Q10000001": 1, "F07Q10000002": 0}


def test_loans_that_left_before_month_36_without_defaulting_are_excluded_not_counted_as_good():
    orig = [orig_row(f"F07Q10000{i:03d}") for i in range(1, 6)]
    perf = (
        perf_rows("F07Q10000001", range(1, 41))                                   # stayed: good
        + perf_rows("F07Q10000002", range(1, 30), zero_balance="01")             # prepaid at 29: excluded
        + perf_rows("F07Q10000003", range(1, 37), zero_balance="01")             # prepaid at exactly 36: observed, good
        + perf_rows("F07Q10000004", range(1, 30), zero_balance="16")             # loan sale at 29: excluded
        + perf_rows("F07Q10000005", range(1, 20), status_at={12: "04"}, zero_balance="01")  # defaulted, then left: kept
    )
    loans, report = prepared(orig, perf)
    assert sorted(loans["loan_sequence"]) == ["F07Q10000001", "F07Q10000003", "F07Q10000005"]
    assert loans.set_index("loan_sequence")["default"].to_dict() == {
        "F07Q10000001": 0, "F07Q10000003": 0, "F07Q10000005": 1}
    assert report["excluded_left_before_horizon"] == 2
    assert report["excluded_by_zero_balance"] == {"01": 1, "16": 1}


def test_a_loan_still_active_but_watched_for_under_36_months_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="still active"):
        prepared([orig_row("F07Q10000001")], perf_rows("F07Q10000001", range(1, 20)))
    # ...but a loan that already defaulted has a known outcome however short its record
    loans, _ = prepared([orig_row("F07Q10000001")], perf_rows("F07Q10000001", range(1, 20), status_at={15: "03"}))
    assert loans["default"].tolist() == [1]


def test_dti_not_available_is_flagged_and_filled_with_the_median_of_the_rest():
    seqs = [f"F07Q10000{i:03d}" for i in range(1, 6)]
    dtis = ["30", "40", "999", "50", "999"]
    orig = [orig_row(s, dti=d) for s, d in zip(seqs, dtis)]
    perf = [row for s in seqs for row in perf_rows(s, range(1, 41))]
    loans, report = prepared(orig, perf)
    assert loans["dti_missing"].tolist() == [0, 0, 1, 0, 1] and report["n_dti_missing"] == 2
    assert loans["dti"].tolist() == [30, 40, 40, 50, 40]  # median of 30, 40, 50


def test_rows_with_a_not_available_code_are_dropped_and_counted():
    seqs = [f"F07Q10000{i:03d}" for i in range(1, 6)]
    orig = [orig_row(seqs[0]), orig_row(seqs[1], credit_score="9999"), orig_row(seqs[2], ltv="999"),
            orig_row(seqs[3], first_time="9"), orig_row(seqs[4], borrowers="99")]
    perf = [row for s in seqs for row in perf_rows(s, range(1, 41))]
    loans, report = prepared(orig, perf)
    assert loans["loan_sequence"].tolist() == [seqs[0]] and loans["loan_id"].tolist() == [1]
    assert report["dropped_unusable_value"] == {
        "credit_score": 1, "ltv": 1, "two_plus_borrowers": 1, "first_time_buyer": 1}


def test_columns_are_decoded_and_scaled():
    orig = [orig_row("F07Q30000001", first_time="Y", channel="T", purpose="C", property_type="CO", occupancy="I",
                     upb="250000", term_months="180", borrowers="1", mi_pct="25", units="2")]
    loans, _ = prepared(orig, perf_rows("F07Q30000001", range(1, 41)))
    row = loans.iloc[0]
    assert list(loans.columns) == prepare.COLUMNS
    assert (row["orig_quarter"], row["channel"], row["purpose"], row["property_type"], row["occupancy"]) == (
        "q3", "tpo_unspecified", "refi_cash_out", "condo", "investment")
    assert (row["orig_upb_100k"], row["term_years"], row["two_plus_borrowers"], row["first_time_buyer"]) == (2.5, 15.0, 0, 1)
    assert (row["mi_pct"], row["n_units"], row["interest_rate"]) == (25, 2, 6.5)


def test_prepare_rejects_a_layout_or_value_it_does_not_understand():
    o, p = [orig_row("F07Q10000001")], perf_rows("F07Q10000001", range(1, 41))
    with pytest.raises(ValueError, match="expected 31 origination columns"):
        prepare.prepare(pd.DataFrame([r + ["x"] for r in o], dtype=str), pd.DataFrame(p, dtype=str))
    with pytest.raises(ValueError, match="channel.*Z"):
        prepared([orig_row("F07Q10000001", channel="Z")], p)
    with pytest.raises(ValueError, match="status values.*XX"):
        prepared(o, perf_rows("F07Q10000001", range(1, 41), status_at={5: "XX"}))
    with pytest.raises(ValueError, match="not unique"):
        prepared(o + o, p)
    with pytest.raises(ValueError, match="not in the origination file"):
        prepared(o, p + perf_rows("F07Q10000099", range(1, 41)))
    with pytest.raises(ValueError, match="must carry columns"):
        prepare.prepare(pd.DataFrame(o, dtype=str), pd.DataFrame(p, dtype=str)[[0, 3]])


def test_a_defect_row_after_the_termination_does_not_move_the_termination_age():
    """Zero balance 96 (defect) can come with later rows; the loan left at the ZERO-BALANCE row."""
    orig = [orig_row("F07Q10000001")]
    perf = perf_rows("F07Q10000001", range(1, 10), zero_balance="96") + perf_rows("F07Q10000001", range(10, 45))
    loans, report = prepared(orig, perf)
    assert loans.empty and report["excluded_by_zero_balance"] == {"96": 1}


# --- config -------------------------------------------------------------------------------


def test_config_has_one_real_dataset_per_vintage_and_a_semi_synthetic_twin(config_loader, fm_config):
    assert config_loader.get_domain(DOMAIN).is_runnable
    assert runner.available_datasets(fm_config) == [
        *(f"real_{y}" for y in YEARS),
        *(f"real_{y}{suffix}" for y in NEW_YEARS for suffix in ("", "_relief_adjusted")),
        "semi_synthetic"]
    assert runner.resolve_dataset(fm_config) == "real_2007"  # the twin never becomes the default
    # the real vintages have no ground truth; only the semi-synthetic twin does
    assert [key for key in SCM_REGISTRY if key[0] == DOMAIN] == [(DOMAIN, "semi_synthetic")]


def test_config_adjusts_each_lever_for_every_other_encoded_column(fm_config, fake_loans):
    features = preprocessing.preprocess(fake_loans, fm_config).feature_columns
    treatments = fm_config["effect_estimation"]["treatments"]
    assert [t["name"] for t in treatments] == list(LEVERS)
    for treatment in treatments:
        assert sorted(treatment["confounders"]) == sorted(set(features) - {treatment["name"]}), treatment["name"]
        assert len(treatment["confounders"]) == len(set(treatment["confounders"]))


def test_config_discovery_variables_and_priors_are_consistent(fm_config, fake_loans):
    features = set(preprocessing.preprocess(fake_loans, fm_config).feature_columns)
    discovery = fm_config["causal_discovery"]
    assert set(discovery["variables"]) <= features | {"default"}
    assert not discovery.get("required_edges")  # nothing is asserted on observational data
    for a, b in discovery["forbidden_edges"]:
        assert a in discovery["variables"] and b in discovery["variables"]
    for variable in set(discovery["variables"]) - {"default"}:
        assert ["default", variable] in discovery["forbidden_edges"]  # the outcome is last
    assert ["interest_rate", "credit_score"] in discovery["forbidden_edges"]


def test_config_levers_candidates_and_sensitive_attribute_line_up(fm_config, fake_loans):
    treatments = {t["name"] for t in fm_config["effect_estimation"]["treatments"]}
    candidates = fm_config["interventions"]["candidates"]
    assert {c["target_variable"] for c in candidates} == treatments
    assert fm_config["counterfactuals"]["treatment"] in treatments
    assert fm_config["interventions"]["sensitive_attribute"] == "first_time_buyer"
    assert set(fake_loans["first_time_buyer"]) == {0, 1}
    assert all(c["cost"] > 0 for c in candidates)


def test_config_encoding_covers_every_prepared_column(fm_config, fake_loans):
    """Every prepared column is a feature, the id or the outcome, or the traceability column
    `loan_sequence`: none silently unused."""
    fs = fm_config["feature_store"]
    used = {fs["entity_id_column"], fm_config["ingestion"]["outcome_column"], *fs["feature_columns"]}
    used |= {c["name"] for c in fs["categorical_columns"]}
    assert used | {"loan_sequence"} == set(fake_loans.columns)


def test_declared_levels_match_the_levels_prepare_can_produce(fm_config):
    produced = {"orig_quarter": {"q1", "q2", "q3", "q4"}, "occupancy": set(prepare.OCCUPANCY.values()),
                "channel": set(prepare.CHANNEL.values()), "purpose": set(prepare.PURPOSE.values()),
                "property_type": set(prepare.PROPERTY.values())}
    declared = {c["name"]: set(c["levels"]) for c in fm_config["feature_store"]["categorical_columns"]}
    assert declared == produced


def test_loader_accepts_real_slices_and_rejects_other_names(tmp_path):
    import yaml

    from rootcause.utils.config_loader import ConfigLoader, REAL_SLICE

    assert REAL_SLICE.fullmatch("real_2007") and REAL_SLICE.fullmatch("real_ex_relief_2010")
    assert not any(REAL_SLICE.fullmatch(k) for k in ("real_", "real", "Real_2007", "real-2007", "fake_2007", "real_2007 "))

    def load(kinds):
        raw = yaml.safe_load((REPO_ROOT / "rootcause" / "configs" / "freddie_mac.yaml").read_text(encoding="utf-8"))
        raw["ingestion"]["datasets"] = {k: "x.csv" for k in kinds}
        raw["ingestion"]["default_dataset"] = kinds[0]
        (tmp_path / "d.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
        return ConfigLoader(tmp_path)

    assert load(["real_2007", "real_2010"]).get_domain(DOMAIN)
    with pytest.raises(ValueError, match=r"unknown dataset kind\(s\) \['real-2007'\]"):
        load(["real_2007", "real-2007"])


# --- the fake data, through the real pipeline ---------------------------------------------


def test_fake_vintage_runs_end_to_end_with_every_level_present(fm_config, fake_loans, isolated_feast_root, tmp_path):
    assert fake_loans["default"].between(0, 1).all() and 0.05 < fake_loans["default"].mean() < 0.6
    assert set(fake_loans["channel"]) == set(prepare.CHANNEL.values())
    path = tmp_path / "loans_fake.csv"
    fake_loans.to_csv(path, index=False)
    cfg = {
        **fm_config,
        "ingestion": {**fm_config["ingestion"], "datasets": {"real_2007": str(path)}, "default_dataset": "real_2007"},
        "effect_estimation": {**fm_config["effect_estimation"], "refutation_simulations": 5},
    }

    real, *rest = harness.evaluate_dataset(DOMAIN, cfg, "real_2007")

    assert rest == [] and real.truth is None and not real.scored_against_truth
    (run,) = real.runs
    assert run.error is None, run.error
    assert run.n_rows == len(fake_loans) and set(run.estimated_ates) == set(LEVERS)
    assert run.estimated_ates["interest_rate"] > 0  # the fake defaults rise with the rate
    assert run.top_choice in {c["id"] for c in fm_config["interventions"]["candidates"]}


# --- the real prepared files (skipped without the registered download) ---------------------


@needs_real
@pytest.mark.parametrize("year", YEARS)
def test_prepared_real_vintage_has_the_documented_counts(year):
    loans = pd.read_csv(real_path(year))
    n, defaults, dti_missing = REAL_COUNTS[year]
    assert list(loans.columns) == prepare.COLUMNS
    assert (len(loans), int(loans["default"].sum()), int(loans["dti_missing"].sum())) == (n, defaults, dti_missing)
    assert loans["loan_id"].tolist() == list(range(1, n + 1)) and loans["loan_sequence"].is_unique
    assert not loans.isna().any().any()


@needs_real
@pytest.mark.parametrize("year", YEARS)
def test_real_vintage_levels_are_declared_and_the_crisis_gap_is_there(year, fm_config):
    loans = pd.read_csv(real_path(year))
    for cfg in fm_config["feature_store"]["categorical_columns"]:
        assert set(loans[cfg["name"]].astype(str)) <= set(map(str, cfg["levels"])), cfg["name"]
    rate = loans["default"].mean()
    assert (rate > 0.10) if year in ("2007", "2008") else (rate < 0.03)


@needs_real
def test_2010_11_relief_refinances_have_no_dti_and_2007_08_mostly_do():
    share = {y: pd.read_csv(real_path(y))["dti_missing"].mean() for y in YEARS}
    assert share["2010"] > 0.25 and share["2011"] > 0.25
    assert share["2007"] < 0.05 and share["2008"] < 0.05


# --- the relief-adjusted outcome and the recent vintages ----------------------------------

RELIEF_COLUMNS = {"deferral": 24, "disaster": 28, "assistance_plan": 29}


def relief_perf(seq, ages, status_at=None, zero_balance="", flags=None):
    """perf_rows plus the three optional relief columns; `flags` maps age -> {column name: value}."""
    rows = perf_rows(seq, ages, status_at=status_at, zero_balance=zero_balance)
    for row in rows:
        for position in RELIEF_COLUMNS.values():
            row[position] = ""
        for column, value in (flags or {}).get(int(row[4]), {}).items():
            row[RELIEF_COLUMNS[column]] = value
    return rows


def relief_case():
    """Seven loans that each hit 90+ days at age 12 (G never does), differing in relief and what followed."""
    plan = {a: {"assistance_plan": "F"} for a in range(10, 15)}
    seqs = {n: f"F19Q10000{i:03d}" for i, n in enumerate("ABCDEFG", start=1)}
    perf = (
        relief_perf(seqs["A"], range(1, 41), status_at={12: "03"}, flags=plan)  # forbearance, cured
        + relief_perf(seqs["B"], range(1, 31), status_at={a: "03" for a in range(12, 31)}, zero_balance="03", flags=plan)  # ...then liquidated
        + relief_perf(seqs["C"], range(1, 41), status_at={12: "03"})  # no relief
        + relief_perf(seqs["D"], range(1, 41), status_at={12: "03"}, flags={20: {"assistance_plan": "F"}})  # flag AFTER
        + relief_perf(seqs["E"], range(1, 41), status_at={12: "03"}, flags={12: {"disaster": "Y"}})
        + relief_perf(seqs["F"], range(1, 41), status_at={12: "03"}, flags={11: {"deferral": "C"}})
        + relief_perf(seqs["G"], range(1, 41), flags=plan)  # relief but never delinquent
    )
    return seqs, [orig_row(s) for s in seqs.values()], perf


def test_relief_adjusted_outcome_drops_only_relief_flagged_delinquencies_that_never_liquidated():
    seqs, orig, perf = relief_case()
    serious, rep_s = prepare.prepare(*frames(orig, perf))
    adjusted, rep_a = prepare.prepare(*frames(orig, perf), outcome="relief_adjusted")
    by_loan = lambda df: dict(zip(df["loan_sequence"], df["default"]))  # noqa: E731
    assert by_loan(serious) == {seqs["A"]: 1, seqs["B"]: 1, seqs["C"]: 1, seqs["D"]: 1, seqs["E"]: 1, seqs["F"]: 1,
                                seqs["G"]: 0}
    # A (forbearance, cured), E (disaster) and F (COVID deferral) are not defaults any more; B was liquidated,
    # C had no relief and D's relief came after the delinquency began
    assert by_loan(adjusted) == {seqs["A"]: 0, seqs["B"]: 1, seqs["C"]: 1, seqs["D"]: 1, seqs["E"]: 0, seqs["F"]: 0,
                                 seqs["G"]: 0}
    assert list(serious["loan_sequence"]) == list(adjusted["loan_sequence"])  # same loans, only the label differs
    for report in (rep_s, rep_a):
        assert report["n_default_serious_delinquency"] == 6 and report["n_default_relief_adjusted"] == 3
        assert report["serious_share_relief_flagged"] == pytest.approx(4 / 6)  # A, B, E, F
        assert report["serious_share_liquidated"] == pytest.approx(1 / 6)
        assert report["serious_share_cured"] == pytest.approx(5 / 6)  # everyone but B went back to current
    assert (rep_s["outcome"], rep_a["outcome"]) == ("serious_delinquency", "relief_adjusted")
    assert rep_s["n_default"] == 6 and rep_a["n_default"] == 3


def test_without_the_relief_columns_both_outcomes_agree_and_nothing_is_flagged():
    orig = [orig_row("F07Q10000001"), orig_row("F07Q10000002")]
    perf = perf_rows("F07Q10000001", range(1, 41), status_at={12: "03"}) + perf_rows("F07Q10000002", range(1, 41))
    a, ra = prepared(orig, perf)
    b, rb = prepare.prepare(*frames(orig, perf), outcome="relief_adjusted")
    assert a["default"].tolist() == b["default"].tolist() == [1, 0]
    assert ra["serious_share_relief_flagged"] == 0.0


def test_a_flag_on_an_unknown_deferral_or_plan_code_is_not_relief():
    seq = "F19Q10000001"
    perf = relief_perf(seq, range(1, 41), status_at={12: "03"},
                       flags={12: {"deferral": "X", "assistance_plan": "N"}})
    loans, _ = prepare.prepare(*frames([orig_row(seq)], perf), outcome="relief_adjusted")
    assert loans["default"].tolist() == [1]


def test_unobserved_loans_can_be_dropped_and_counted_instead_of_raising():
    young, defaulted = "F22Q10000001", "F22Q10000002"
    orig = [orig_row(young), orig_row(defaulted), orig_row("F22Q10000003")]
    perf = (perf_rows(young, range(1, 20)) + perf_rows(defaulted, range(1, 20), status_at={15: "03"})
            + perf_rows("F22Q10000003", range(1, 41)))
    with pytest.raises(ValueError, match="still active"):
        prepare.prepare(*frames(orig, perf))
    loans, report = prepare.prepare(*frames(orig, perf), unobserved="drop")
    # the already-defaulted loan has a known outcome and stays; the young non-defaulter is unknown
    assert sorted(loans["loan_sequence"]) == [defaulted, "F22Q10000003"]
    assert report["excluded_still_unobserved"] == 1 and report["excluded_left_before_horizon"] == 0


def test_prepare_rejects_an_unknown_outcome_or_unobserved_mode():
    o, p = [orig_row("F07Q10000001")], perf_rows("F07Q10000001", range(1, 41))
    with pytest.raises(ValueError, match="outcome must be one of"):
        prepare.prepare(*frames(o, p), outcome="loss")
    with pytest.raises(ValueError, match="unobserved must be"):
        prepare.prepare(*frames(o, p), unobserved="guess")


@needs_new
@pytest.mark.parametrize("year", NEW_YEARS)
def test_recent_vintages_have_the_documented_counts_under_both_outcomes(year):
    n, serious, adjusted, dti_missing = NEW_COUNTS[year]
    a, b = pd.read_csv(new_path(year)), pd.read_csv(new_path(year, adjusted=True))
    assert list(a.columns) == list(b.columns) == prepare.COLUMNS
    assert (len(a), int(a["default"].sum()), int(a["dti_missing"].sum())) == (n, serious, dti_missing)
    assert int(b["default"].sum()) == adjusted and len(b) == n
    assert a["loan_sequence"].tolist() == b["loan_sequence"].tolist()
    assert (b["default"] <= a["default"]).all()  # the adjusted outcome only removes defaults
    assert not a.isna().any().any() and not b.isna().any().any()


@needs_new
def test_2019_is_a_pandemic_stress_case_the_90_day_outcome_is_not_credit_default():
    serious, adjusted = (pd.read_csv(new_path("2019", a))["default"].mean() for a in (False, True))
    assert serious > 0.10 and adjusted < 0.02  # ~12.6% vs ~1.0%: the gap is the forbearance
    # the other recent vintages sit on the ordinary side of the same rates
    assert pd.read_csv(new_path("2016"))["default"].mean() < 0.03
