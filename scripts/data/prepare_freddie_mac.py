"""Build data/freddie_mac/loans_<year>.csv (one row per loan) from a Freddie Mac Single-Family
Loan-Level Dataset SAMPLE vintage.

Source: Freddie Mac's Single-Family Loan-Level Dataset, Standard sample (a simple random sample of
50,000 fixed-rate loans per origination year), downloaded after registering at Freddie Mac's
Clarity / SFLLD data-download page. The raw files are NOT redistributed here (data/freddie_mac/ is
gitignored): put `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt` under
data/freddie_mac/raw/sample_YYYY/ and run

    python scripts/data/prepare_freddie_mac.py            # every vintage found under raw/
    python scripts/data/prepare_freddie_mac.py 2007 2010  # just these
    python scripts/data/prepare_freddie_mac.py --relief-adjusted 2019   # the relief-adjusted outcome (below)

Both raw files are pipe-delimited with NO header. The files this script was written against have 31
origination and 35 performance columns (the January 2026 user guide lists 32 origination columns;
servicer name is not in the sample origination file). Columns 1-23 of the origination file and
columns 1-13 of the performance file match the guide, and those are the only ones used.

Choices, all visible in the output and in the returned report:

  * Outcome `default` = 1 if the loan was ever 90+ days delinquent (status 03 or higher), in REO
    acquisition (RA), or terminated as a short sale/charge-off (zero balance 03) or REO disposition
    (09), at loan age <= HORIZON months (36). Delinquency status is a zero-padded string ("00",
    "01", ..., "RA"), not the 0/1/2 of the guide.
  * CENSORING: a loan that left the data before month 36 without defaulting (mostly prepaid or
    refinanced, zero balance 01) is EXCLUDED, not counted as good: its outcome is unknown, and
    counting it as good would understate risk. This conditions the sample on surviving to month 36,
    so default rates are "among loans that stayed", not lifetime default probabilities. The report
    gives the number excluded per zero-balance code.
  * TWO OUTCOMES. `serious_delinquency` (the default above) counts every 90+ day delinquency. In
    2020-21 most of those were pandemic forbearance, not credit failures: in the 2019 sample 92% of
    the loans that went 90+ days past due carried a relief flag at or before that month, 97% later
    returned to current and 0.5% were ever liquidated (2007: 47% liquidated). `relief_adjusted`
    drops from the defaults every loan that (a) had a relief flag at or before its first 90+ day
    month (disaster flag Y, payment deferral P/C, or a borrower assistance plan F/R/T) AND (b) was
    never liquidated (zero balance 03/09 or REO) at any later age. Those loans stay in the sample
    as non-defaults. It is a heuristic, not a verified loss definition: relief was not randomly
    assigned and some relieved loans were genuinely troubled. The report gives both counts.
  * HORIZON is loan AGE, which Freddie Mac does not advance for every calendar month a delinquent
    loan misses: a few recent-vintage loans are still active at the data cutoff with age < 36.
    Those that have not yet defaulted have an unknown outcome and are dropped and counted
    (`excluded_still_unobserved`); a loan that already defaulted keeps its known outcome.
  * dti_missing: original DTI is "not available" for every relief-refinance (HARP) loan, about 29% of
    the 2010 and 31% of the 2011 samples (and about 2% of 2007-08). DTI is filled with the
    vintage median of the observed values and `dti_missing` = 1 marks those loans. In 2010-11 the
    flag is therefore also a relief-refinance indicator, so a linear DTI slope is identified from
    the loans that report it.
  * Rows dropped for an unusable value: credit score 9999, LTV 999, first-time-homebuyer 9,
    number of borrowers 99, MI % 999, units 99, and unknown occupancy, channel or property type.
    The counts are in the report.
  * Not used: state, MSA, postal code, seller, first payment/maturity dates, CLTV, prepayment
    penalty, interest-only and the other columns that are constant or nearly so in these vintages.

`interest_rate`, `ltv` and `dti` are the levers the pipeline studies; the rest are covariates known
when the loan was made. There is no race, sex or age in this dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "freddie_mac"
RAW_DIR = DATA_DIR / "raw"

HORIZON = 36  # months of loan age over which default is observed
ORIG_COLUMNS = 31

# 0-based positions in the raw files (see the module docstring).
ORIG = {"credit_score": 0, "first_payment": 1, "first_time": 2, "mi_pct": 5, "units": 6,
        "occupancy": 7, "dti": 9, "upb": 10, "ltv": 11, "rate": 12, "channel": 13,
        "property_type": 17, "loan_seq": 19, "purpose": 20, "term_months": 21, "borrowers": 22}
PERF = {"loan_seq": 0, "status": 3, "age": 4, "zero_balance": 8}
# optional columns: only the relief_adjusted outcome and the report read them; absent = no relief
RELIEF = {"deferral": 24, "disaster": 28, "assistance_plan": 29}
RELIEF_DEFERRAL = {"P", "C"}  # payment deferral, COVID-19 payment deferral
RELIEF_ASSISTANCE_PLAN = {"F", "R", "T"}  # forbearance, repayment plan, trial period plan
OUTCOMES = ("serious_delinquency", "relief_adjusted")

OCCUPANCY = {"P": "primary", "I": "investment", "S": "second_home"}
CHANNEL = {"R": "retail", "B": "broker", "C": "correspondent", "T": "tpo_unspecified"}
PURPOSE = {"P": "purchase", "C": "refi_cash_out", "N": "refi_no_cash_out"}
PROPERTY = {"SF": "single_family", "PU": "pud", "CO": "condo", "MH": "manufactured", "CP": "coop"}
FIRST_TIME = {"Y": 1, "N": 0}

DEFAULT_ZERO_BALANCE = {"03", "09"}  # short sale / charge-off, REO disposition
OUTCOME = "default"
COLUMNS = [
    "loan_id", "loan_sequence", "orig_quarter", "credit_score", "ltv", "dti", "dti_missing", "interest_rate",
    "orig_upb_100k", "term_years", "mi_pct", "n_units", "two_plus_borrowers", "first_time_buyer",
    "occupancy", "channel", "purpose", "property_type", OUTCOME,
]


def read_orig(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", header=None, dtype=str, keep_default_na=False)


def read_perf(path: Path) -> pd.DataFrame:
    """Only the performance columns the outcome needs (the file is ~3.5M rows)."""
    return pd.read_csv(path, sep="|", header=None, dtype=str, keep_default_na=False,
                       usecols=sorted({*PERF.values(), *RELIEF.values()}), names=None)


def _relief(perf: pd.DataFrame) -> np.ndarray:
    """Row-level: a disaster flag, a payment deferral or a forbearance-type assistance plan."""
    def column(key: str) -> np.ndarray:
        position = RELIEF[key]
        return perf[position].to_numpy() if position in perf.columns else np.full(len(perf), "", dtype=object)

    return ((column("disaster") == "Y") | np.isin(column("deferral"), list(RELIEF_DEFERRAL))
            | np.isin(column("assistance_plan"), list(RELIEF_ASSISTANCE_PLAN)))


def _default_by_loan(perf: pd.DataFrame, horizon: int, unobserved: str = "error") -> pd.DataFrame:
    """Per loan: defaulted within `horizon` months (and the relief-adjusted variant), and whether its
    outcome is observable. `unobserved`: a loan still active with < `horizon` months of age and no
    default has an unknown outcome -> "error" raises, "drop" marks it not observable."""
    p = pd.DataFrame({
        "loan_sequence": perf[PERF["loan_seq"]].to_numpy(),
        "status": perf[PERF["status"]].to_numpy(),
        "age": pd.to_numeric(perf[PERF["age"]], errors="raise").to_numpy(),
        "zero_balance": perf[PERF["zero_balance"]].to_numpy(),
        "relief": _relief(perf),
    })
    numeric = pd.to_numeric(p["status"], errors="coerce")
    unknown = sorted(set(p.loc[numeric.isna() & (p["status"] != "RA"), "status"]))
    if unknown:
        raise ValueError(f"delinquency status values {unknown} are not decoded")
    bad = (p["status"] == "RA") | (numeric >= 3) | p["zero_balance"].isin(DEFAULT_ZERO_BALANCE)
    first_bad = p.loc[bad].groupby("loan_sequence")["age"].min()
    ended = p.loc[p["zero_balance"] != ""].groupby("loan_sequence").agg(
        end_age=("age", "min"), zero_balance=("zero_balance", "first"))
    out = pd.DataFrame({"last_age": p.groupby("loan_sequence")["age"].max()})
    out["first_bad_age"] = first_bad
    out = out.join(ended)
    out["default"] = (out["first_bad_age"] <= horizon).astype(int)

    # what happened around and after the first 90+ day month (only meaningful for loans that have one)
    p = p.join(out["first_bad_age"], on="loan_sequence")
    def loans_where(mask: pd.Series) -> set:
        return set(p.loc[mask, "loan_sequence"])

    relief_at_or_before = loans_where(p["relief"] & (p["age"] <= p["first_bad_age"]))
    cured_later = loans_where((p["age"] > p["first_bad_age"]) & (p["status"] == "00"))
    liquidated = loans_where((p["status"] == "RA") | p["zero_balance"].isin(DEFAULT_ZERO_BALANCE))
    out["relief_before"] = out.index.isin(relief_at_or_before)
    out["cured"] = out.index.isin(cured_later)
    out["liquidated"] = out.index.isin(liquidated)
    out["default_relief_adjusted"] = (out["default"].astype(bool) & ~(out["relief_before"] & ~out["liquidated"])).astype(int)

    left_early = out["end_age"] < horizon  # terminated before month `horizon`
    # data cutoff hit before month `horizon` with no default yet: the outcome is unknown (a
    # defaulted loan's outcome is known)
    still_unseen = out["end_age"].isna() & (out["last_age"] < horizon) & (out["default"] == 0)
    if still_unseen.any() and unobserved == "error":
        raise ValueError(f"{int(still_unseen.sum())} loans are still active but observed for < {horizon} months")
    out["unobserved"] = still_unseen
    out["observable"] = ((out["default"] == 1) | ~left_early) & ~still_unseen
    return out


def _decode(series: pd.Series, mapping: dict, name: str, missing: tuple = ()) -> pd.Series:
    unknown = sorted(set(series) - set(mapping) - set(missing))
    if unknown:
        raise ValueError(f"{name}: values {unknown} are not decoded")
    return series.map(mapping)


def prepare(orig: pd.DataFrame, perf: pd.DataFrame, horizon: int = HORIZON, outcome: str = "serious_delinquency",
            unobserved: str = "error") -> tuple[pd.DataFrame, dict]:
    """(loans, report): one row per observable loan, and the counts behind every drop. `outcome` is
    one of OUTCOMES (see the module docstring); `unobserved` is passed to `_default_by_loan`."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    if unobserved not in ("error", "drop"):
        raise ValueError(f"unobserved must be 'error' or 'drop', got {unobserved!r}")
    if orig.shape[1] != ORIG_COLUMNS:
        raise ValueError(f"expected {ORIG_COLUMNS} origination columns, got {orig.shape[1]}")
    if not set(PERF.values()) <= set(perf.columns):
        raise ValueError(f"performance data must carry columns {sorted(PERF.values())}, got {list(perf.columns)}")
    seq = orig[ORIG["loan_seq"]]
    if not seq.is_unique:
        raise ValueError("loan sequence numbers are not unique")
    if not set(perf[PERF["loan_seq"]]) <= set(seq):
        raise ValueError("performance file has loans that are not in the origination file")

    df = pd.DataFrame({"loan_sequence": seq.to_numpy()})
    df["orig_quarter"] = "q" + df["loan_sequence"].str[4]
    num = lambda key: pd.to_numeric(orig[ORIG[key]], errors="raise").to_numpy()  # noqa: E731
    df["credit_score"] = num("credit_score")
    df["ltv"] = num("ltv")
    dti = pd.Series(num("dti"))
    df["dti"] = dti
    df["interest_rate"] = num("rate")
    df["orig_upb_100k"] = num("upb") / 100_000.0
    df["term_years"] = num("term_months") / 12.0
    df["mi_pct"] = num("mi_pct")
    df["n_units"] = num("units")
    borrowers = pd.Series(num("borrowers"))
    df["two_plus_borrowers"] = np.where(borrowers == 99, np.nan, (borrowers >= 2).astype(float))
    df["first_time_buyer"] = _decode(orig[ORIG["first_time"]], FIRST_TIME, "first time homebuyer", ("9",)).to_numpy()
    df["occupancy"] = _decode(orig[ORIG["occupancy"]], OCCUPANCY, "occupancy").to_numpy()
    df["channel"] = _decode(orig[ORIG["channel"]], CHANNEL, "channel").to_numpy()
    df["purpose"] = _decode(orig[ORIG["purpose"]], PURPOSE, "loan purpose").to_numpy()
    df["property_type"] = _decode(orig[ORIG["property_type"]], PROPERTY, "property type").to_numpy()

    # "Not available" codes -> NaN. DTI is handled separately below.
    df.loc[df["credit_score"] == 9999, "credit_score"] = np.nan
    df.loc[df["ltv"] == 999, "ltv"] = np.nan
    df.loc[df["mi_pct"] == 999, "mi_pct"] = np.nan
    df.loc[df["n_units"] == 99, "n_units"] = np.nan
    df["dti_missing"] = (df["dti"] == 999).astype(int)
    df.loc[df["dti"] == 999, "dti"] = np.nan

    per_loan = _default_by_loan(perf, horizon, unobserved)
    df = df.join(per_loan[["default", "default_relief_adjusted", "observable", "unobserved", "relief_before", "cured",
                           "liquidated", "zero_balance"]], on="loan_sequence")
    if df["default"].isna().any():
        raise ValueError("some origination loans have no performance rows")

    report: dict = {"horizon_months": horizon, "outcome": outcome, "n_raw": len(df)}
    unseen = df["unobserved"].astype(bool)
    left = df[~df["observable"].astype(bool) & ~unseen]
    report["excluded_still_unobserved"] = int(unseen.sum())
    report["excluded_left_before_horizon"] = len(left)
    report["excluded_by_zero_balance"] = {str(k): int(v) for k, v in left["zero_balance"].value_counts().items()}
    df = df[df["observable"].astype(bool)].drop(columns=["observable", "unobserved", "zero_balance"])

    required = ["credit_score", "ltv", "mi_pct", "n_units", "two_plus_borrowers", "first_time_buyer"]
    report["dropped_unusable_value"] = {c: int(df[c].isna().sum()) for c in required if df[c].isna().any()}
    df = df.dropna(subset=required)
    report["n_dti_missing"] = int(df["dti_missing"].sum())
    df["dti"] = df["dti"].fillna(df["dti"].median())

    serious = df["default"].astype(int)
    adjusted = df["default_relief_adjusted"].astype(int)
    flagged, cured, liquidated = (df[c].astype(bool) & (serious == 1) for c in ("relief_before", "cured", "liquidated"))
    n_serious = int(serious.sum())
    share = lambda mask: float(mask.sum() / n_serious) if n_serious else 0.0  # noqa: E731
    report.update(n_default_serious_delinquency=n_serious, n_default_relief_adjusted=int(adjusted.sum()),
                  serious_share_relief_flagged=share(flagged), serious_share_cured=share(cured),
                  serious_share_liquidated=share(liquidated))
    df["default"] = serious if outcome == "serious_delinquency" else adjusted
    df = df.drop(columns=["default_relief_adjusted", "relief_before", "cured", "liquidated"])
    df["first_time_buyer"] = df["first_time_buyer"].astype(int)
    df["two_plus_borrowers"] = df["two_plus_borrowers"].astype(int)
    for column in ("credit_score", "ltv", "n_units", "mi_pct"):
        df[column] = df[column].astype(int)
    df.insert(0, "loan_id", np.arange(1, len(df) + 1))
    out = df[COLUMNS].reset_index(drop=True)
    report["n_loans"] = len(out)
    report["n_default"] = int(out[OUTCOME].sum())
    report["default_rate"] = float(out[OUTCOME].mean())
    return out, report


def vintages() -> list[str]:
    return sorted(p.name.removeprefix("sample_") for p in RAW_DIR.glob("sample_*") if p.is_dir())


def build(year: str, outcome: str = "serious_delinquency") -> dict:
    folder = RAW_DIR / f"sample_{year}"
    orig_path, perf_path = folder / f"sample_orig_{year}.txt", folder / f"sample_perf_{year}.txt"
    for path in (orig_path, perf_path):
        if not path.exists():
            sys.exit(f"{path} not found. See the module docstring for where to get and put the raw files.")
    loans, report = prepare(read_orig(orig_path), read_perf(perf_path), outcome=outcome, unobserved="drop")
    suffix = "" if outcome == "serious_delinquency" else f"_{outcome}"
    loans.to_csv(DATA_DIR / f"loans_{year}{suffix}.csv", index=False)
    return report


def main(argv: list[str]) -> None:
    outcome = "serious_delinquency"
    if "--relief-adjusted" in argv:
        outcome = "relief_adjusted"
        argv = [a for a in argv if a != "--relief-adjusted"]
    years = argv or vintages()
    if not years:
        sys.exit(f"no vintages under {RAW_DIR}")
    for year in years:
        r = build(year, outcome)
        print(f"{year} [{outcome}]: {r['n_raw']} sampled -> {r['excluded_left_before_horizon']} left before month "
              f"{r['horizon_months']} without defaulting {r['excluded_by_zero_balance']}, "
              f"{r['excluded_still_unobserved']} still active with an unknown outcome, "
              f"{sum(r['dropped_unusable_value'].values())} dropped for an unusable value "
              f"{r['dropped_unusable_value']} -> {r['n_loans']} loans, {r['n_default']} defaults "
              f"({r['default_rate']:.2%}), {r['n_dti_missing']} with DTI not available")
        print(f"    of the {r['n_default_serious_delinquency']} serious delinquencies: "
              f"{r['serious_share_relief_flagged']:.1%} relief-flagged at or before the first 90+ day month, "
              f"{r['serious_share_cured']:.1%} later current again, {r['serious_share_liquidated']:.1%} ever "
              f"liquidated/REO; relief-adjusted defaults = {r['n_default_relief_adjusted']}")


if __name__ == "__main__":
    main(sys.argv[1:])
