# carclaims data

| File | What it is |
|---|---|
| `raw/carclaims.csv` | The raw file, unmodified (SHA-256 `a9ff2d9b…f6403bf6ace2d6`), from https://raw.githubusercontent.com/Rashmi-77/Vehicle-Insurance-Fraud-Detection/main/carclaims.csv (also on Kaggle as "Vehicle Claim Fraud Detection"). 15,420 vehicle-insurance claims from 1994-96, 33 columns, no missing values, `FraudFound` = Yes on 923 (5.99%). The `carclaims` sample distributed with Angoss KnowledgeSEEKER, used in many fraud-detection papers (e.g. Phua et al. 2004). |
| `claims.csv` | Derived by `scripts/data/prepare_carclaims.py`: bins decoded into snake_case levels, columns renamed, an id (`claim_id` = the source's PolicyNumber, 1-15,420), and the outcome `fraud_found`. The dataset the pipeline reads. |

There is **no synthetic or semi-synthetic twin** of this dataset, and the pipeline's numbers on it are not validated against anything. See "What this data can and cannot support".

## What this data can and cannot support

- **Provenance.** It is a vendor sample from an unnamed insurer. Nothing in it can be checked against the insurer, so "real" rests on the file's long use in the literature. I found no license statement; check before redistributing.
- **Observational, no ground truth.** Nobody was randomized to a deductible, a police report or a sales channel. The pipeline's estimates are associations adjusted for the measured columns; whatever drove a policyholder's choices (or an insurer's rules) and also drives fraud is not blocked, and the permutation placebo cannot see it.
- **The outcome is fraud that was found.** A lever can move it by changing the amount of fraud or the chance of detecting it (a police report is evidence an investigator can use). The data cannot separate the two.
- **Selection.** Only filed claims appear. Fraud that never becomes a claim, and claims that were never investigated, are absent.
- **Weak levers.** 428 claims have a police report (2.8%), 241 an internal agent (1.6%) and 582 a deductible other than $400 (3.8%). The estimates for these are noisy and the report says so through the bootstrap.
- **Vintage.** Claims from 1994-96.

## Choices made in `prepare_carclaims.py`

- **Dropped, no plausible role as a cause of both a lever and fraud:** the calendar columns (`Month`, `WeekOfMonth`, `DayOfWeek` and their `...Claimed` counterparts, one row of which is 0 for month and weekday), and `RepNumber` (claims handler, 16 levels).
- **`Age` dropped, only `AgeOfPolicyHolder` kept.** `Age` is 0 in exactly the 320 rows in the "16 to 17" bracket, and the two columns disagree elsewhere (the "26 to 30" bracket holds ages 21 to 25). The bracket is kept as an ordinal `policyholder_age_group`.
- **`PolicyType` dropped.** It is `VehicleCategory` + `BasePolicy`, but not consistently: all 4,987 "Sedan - Liability" rows have VehicleCategory Sport. `vehicle_category` and `base_policy` are kept as given, so the source's own inconsistency is visible rather than hidden.
- **`Make`:** the twelve makes with fewer than 300 claims (704 together) are grouped as `other`; the source's misspellings (Accura, Nisson, Porche, Mecedes) are corrected.
- **`deductible_100usd`** = `Deductible` / 100, so one unit of an effect estimate is $100. 96% of claims have a $400 deductible.
- **Binned counts and durations are ordinal** (one step per bin), which treats unequal-width bins as equally spaced. Only ordering is used.

## Facts worth knowing before reading the results

- Fraud found by `Fault`: 7.9% when the policyholder is at fault against 0.9% when a third party is. `Fault` is not something an insurer can change, so it is a covariate, not a lever.
- Fraud found by `BasePolicy`: liability 0.7%, collision 7.3%, all perils 10.2%. Female policyholders hold more liability policies (39% against 31% for male), which explains part of why fraud is found less often for them (4.3% against 6.3%, a Stage 6 ratio of 0.69).
- `DriverRating` is flat across 1-4 (5.6-6.2%); it carries no fraud signal.

**Cite:** Angoss KnowledgeSEEKER `carclaims` sample; Phua, C., Alahakoon, D., Lee, V. (2004), *Minority report in fraud detection*, SIGKDD Explorations 6(1).
