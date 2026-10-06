"""Build data/carclaims/claims.csv from the raw `carclaims` vehicle-insurance file.

Source: the `carclaims` sample shipped with Angoss KnowledgeSEEKER (claims from 1994-96 at an
unnamed US insurer), used in many fraud-detection studies (e.g. Phua et al. 2004). Copy used:
https://raw.githubusercontent.com/Rashmi-77/Vehicle-Insurance-Fraud-Detection/main/carclaims.csv
(also on Kaggle as "Vehicle Claim Fraud Detection"), kept in raw/ as carclaims.csv.
15,420 claims, 923 (6.0%) with FraudFound = Yes; observational, and the outcome is fraud that
was FOUND among claims that were filed.

The raw file uses free-text bins ("more than 30", "2 to 4") and column names with ':' and '-'.
This script decodes them into snake_case levels and makes these choices, all visible in the
output:

  * Dropped as not plausibly causing both a lever and fraud: the six calendar columns (month,
    week and weekday of the accident and of the claim; one row has 0 for the claim month and
    weekday), RepNumber (the claims handler, a 16-level id) and PolicyNumber (becomes claim_id).
  * Dropped as unreliable: `Age` is 0 in 320 rows (exactly the "16 to 17" bracket) and does not
    agree with the policyholder age bracket (the "26 to 30" bracket holds ages 21-25), so only
    the bracket is kept, as `policyholder_age_group`.
  * Dropped as redundant and inconsistent: `PolicyType` is vehicle category + base policy, but
    all 4,987 "Sedan - Liability" rows have VehicleCategory Sport. `vehicle_category` and
    `base_policy` are kept as given.
  * make: the twelve makes with fewer than 300 claims (704 claims together) are grouped as
    `other`; the source's misspellings (Accura, Nisson, Porche, Mecedes) are corrected.
  * deductible_100usd = Deductible / 100 (300-700 dollars becomes 3-7), so one unit is $100.
  * fraud_found = 1 for FraudFound = Yes (the outcome).

    python scripts/data/prepare_carclaims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "carclaims"

RAW_COLUMNS = [
    "Month", "WeekOfMonth", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed",
    "MonthClaimed", "WeekOfMonthClaimed", "Sex", "MaritalStatus", "Age", "Fault", "PolicyType",
    "VehicleCategory", "VehiclePrice", "PolicyNumber", "RepNumber", "Deductible", "DriverRating",
    "Days:Policy-Accident", "Days:Policy-Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfSuppliments", "AddressChange-Claim", "NumberOfCars", "Year", "BasePolicy", "FraudFound",
]

MAKES_KEPT = {"Pontiac": "pontiac", "Toyota": "toyota", "Honda": "honda", "Mazda": "mazda",
              "Chevrolet": "chevrolet", "Accura": "acura", "Ford": "ford"}
MAKES_GROUPED = ("VW", "Dodge", "Saab", "Mercury", "Saturn", "Nisson", "BMW", "Jaguar", "Porche",
                 "Mecedes", "Ferrari", "Lexus")

DAYS_BINS = {"none": "none", "1 to 7": "1_to_7", "8 to 15": "8_to_15", "15 to 30": "15_to_30",
             "more than 30": "gt_30"}

# raw column -> (output column, {raw level: output level})
DECODE = {
    "Make": ("make", {**MAKES_KEPT, **{m: "other" for m in MAKES_GROUPED}}),
    "MaritalStatus": ("marital_status", {"Married": "married", "Single": "single",
                                         "Divorced": "divorced", "Widow": "widow"}),
    "BasePolicy": ("base_policy", {"Collision": "collision", "All Perils": "all_perils",
                                   "Liability": "liability"}),
    "VehicleCategory": ("vehicle_category", {"Sedan": "sedan", "Sport": "sport", "Utility": "utility"}),
    "VehiclePrice": ("vehicle_price", {
        "less than 20,000": "lt_20k", "20,000 to 29,000": "20k_to_29k", "30,000 to 39,000": "30k_to_39k",
        "40,000 to 59,000": "40k_to_59k", "60,000 to 69,000": "60k_to_69k", "more than 69,000": "gt_69k",
    }),
    "AgeOfVehicle": ("vehicle_age", {
        "new": "new", "2 years": "2y", "3 years": "3y", "4 years": "4y", "5 years": "5y",
        "6 years": "6y", "7 years": "7y", "more than 7": "gt_7y",
    }),
    "AgeOfPolicyHolder": ("policyholder_age_group", {
        "16 to 17": "16_to_17", "18 to 20": "18_to_20", "21 to 25": "21_to_25", "26 to 30": "26_to_30",
        "31 to 35": "31_to_35", "36 to 40": "36_to_40", "41 to 50": "41_to_50", "51 to 65": "51_to_65",
        "over 65": "over_65",
    }),
    "Days:Policy-Accident": ("days_policy_to_accident", DAYS_BINS),
    "Days:Policy-Claim": ("days_policy_to_claim", DAYS_BINS),
    "PastNumberOfClaims": ("past_claims", {"none": "none", "1": "1", "2 to 4": "2_to_4", "more than 4": "gt_4"}),
    "NumberOfSuppliments": ("n_supplements", {"none": "none", "1 to 2": "1_to_2", "3 to 5": "3_to_5",
                                              "more than 5": "gt_5"}),
    "AddressChange-Claim": ("address_change", {
        "under 6 months": "lt_6m", "1 year": "1y", "2 to 3 years": "2_to_3y", "4 to 8 years": "4_to_8y",
        "no change": "no_change",
    }),
    "NumberOfCars": ("n_cars", {"1 vehicle": "1", "2 vehicles": "2", "3 to 4": "3_to_4", "5 to 8": "5_to_8",
                                "more than 8": "gt_8"}),
}

# raw column -> (output column, the raw value that is 1)
BINARY = {
    "Sex": ("female", "Female"),
    "Fault": ("fault_policy_holder", "Policy Holder"),
    "AccidentArea": ("accident_rural", "Rural"),
    "WitnessPresent": ("witness_present", "Yes"),
    "PoliceReportFiled": ("police_report_filed", "Yes"),
    "AgentType": ("agent_internal", "Internal"),
    "FraudFound": ("fraud_found", "Yes"),
}
BINARY_OTHER = {"Sex": "Male", "Fault": "Third Party", "AccidentArea": "Urban", "WitnessPresent": "No",
                "PoliceReportFiled": "No", "AgentType": "External", "FraudFound": "No"}

OUTCOME = "fraud_found"
COLUMNS = [
    "claim_id", "year", "policyholder_age_group", "female", "marital_status", "make", "vehicle_category",
    "vehicle_price", "vehicle_age", "n_cars", "base_policy", "deductible_100usd", "driver_rating",
    "agent_internal", "past_claims", "days_policy_to_accident", "days_policy_to_claim", "address_change",
    "fault_policy_holder", "accident_rural", "witness_present", "police_report_filed", "n_supplements",
    OUTCOME,
]


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"expected columns {RAW_COLUMNS}, got {list(raw.columns)}")
    df = pd.DataFrame(index=raw.index)
    df["claim_id"] = raw["PolicyNumber"].astype(int)
    if not df["claim_id"].is_unique:
        raise ValueError("PolicyNumber is not unique, so it cannot be the claim id")

    for column, (name, mapping) in DECODE.items():
        unknown = sorted(set(raw[column].astype(str)) - set(mapping))
        if unknown:
            raise ValueError(f"{column}: values {unknown} are not decoded")
        df[name] = raw[column].astype(str).map(mapping)
    for column, (name, positive) in BINARY.items():
        unknown = sorted(set(raw[column]) - {positive, BINARY_OTHER[column]})
        if unknown:
            raise ValueError(f"{column}: values {unknown} are not decoded")
        df[name] = (raw[column] == positive).astype(int)

    df["deductible_100usd"] = raw["Deductible"] / 100.0
    df["driver_rating"] = raw["DriverRating"].astype(int)
    df["year"] = raw["Year"].astype(int)
    return df[COLUMNS]


def main() -> None:
    source = DATA_DIR / "raw" / "carclaims.csv"
    if not source.exists():
        sys.exit(
            f"{source} not found. Download it from "
            "https://raw.githubusercontent.com/Rashmi-77/Vehicle-Insurance-Fraud-Detection/main/carclaims.csv"
        )
    raw = pd.read_csv(source)
    out = prepare(raw)
    path = DATA_DIR / "claims.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows ({int(out[OUTCOME].sum())} fraud found, {out[OUTCOME].mean():.1%}) to {path}")


if __name__ == "__main__":
    main()
