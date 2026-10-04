# Freddie Mac loan data

The only file here that is committed. Everything else in this folder (`raw/` and `loans_<year>*.csv`) is **gitignored**: the data are a registered download, so neither the raw files nor any loan-level file derived from them is committed.

**Terms.** Source: Freddie Mac's Single-Family Loan-Level Dataset, used under its "Terms and Conditions for Single-Family Loan-Level Dataset" (Effective November 2025) and Freddie Mac's general website terms (updated 2026-09-11). Those terms allow analysis for personal or internal purposes and allow academic or research results and related derived products to be made public for noncommercial purposes, provided they cannot be used to derive or recreate any part of the dataset or to identify an individual. They prohibit giving the dataset or derived products to third parties otherwise. What this repository publishes is aggregate (evaluation tables, effect estimates, graph figures, code); it contains no loan-level rows and no real loan identifiers. This is a noncommercial research and learning project and is not affiliated with or endorsed by Freddie Mac.

## Where the data come from

Freddie Mac's **Single-Family Loan-Level Dataset**, Standard sample: a simple random sample of 50,000 loans from each origination year, fixed-rate mortgages that Freddie Mac bought or that back its securities. Free, after registering at Freddie Mac's Clarity data-download page. The general user guide (January 2026) is at https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf. Vintages used: **2007, 2008, 2010, 2011, 2016, 2019, 2022** (a 2009 sample is a mixed crisis/HARP year and was skipped; 2010-11 are the calm, refinance-heavy years; 2016 is a calm baseline; 2019 is the pandemic stress case; 2022 is the newest year whose loans can all be observed for 36 months, because the data end 2026-03-31). 2023 onward cannot be used with a 36-month outcome. The samples carry the July 2026 (Release 47) layout: 31 origination and 35 performance columns, with the columns the script reads at the same positions as before.

```
data/freddie_mac/raw/sample_2007/sample_orig_2007.txt   origination, pipe-delimited, no header
data/freddie_mac/raw/sample_2007/sample_perf_2007.txt   monthly performance (3.0M rows)
...same for the other vintages
```

Then build the pipeline's input, one row per loan:

```
.venv/Scripts/python.exe scripts/prepare_freddie_mac.py          # every vintage under raw/ (about 20 s)
```

## Where the files differ from the user guide

Checked first-hand on the downloaded samples (the prepare script relies on these):

- The performance file is `sample_perf_YYYY.txt` (the guide says `sample_svcg_`).
- The origination file has **31** columns, not 32: servicer name is not in it (it is column 34 of the 35-column performance file). Columns 1-23 match the guide.
- Delinquency status is a zero-padded string (`00`, `01`, ..., `RA`), not `0`, `1`, `2`.
- `MI cancellation` (column 31) is the constant `9999`, so it is unused.

## What the prepare script decides

- **Outcome `default`** = ever 90+ days delinquent (status 03 or worse), REO acquisition (`RA`), or terminated by short sale/charge-off (zero balance 03) or REO disposition (09), at loan age of 36 months or less.
- **Loans that left before month 36 without defaulting are excluded, not counted as good.** In these vintages that is 33-49% of the sample (nearly all prepaid or refinanced). Counting them as good would understate risk; excluding them means the default rates describe loans that *stayed* for 36 months, not a lifetime probability, and the survivors are not a random subset (a borrower who could refinance is a better credit).
- **DTI not available** for every relief-refinance (HARP) loan: about 34% of the kept 2010 loans and 37% of 2011, and about 2% of 2007-08. It is filled with the vintage median and flagged in `dti_missing`, so in 2010-11 that flag is also a relief-refinance indicator. The DTI effect is identified from the loans that report it.
- **Rows dropped for an unusable value** (credit score 9999, LTV 999, first-time-homebuyer 9, number of borrowers 99): 19-52 rows per vintage.
- **Two outcomes (`--relief-adjusted`).** The 90+ day outcome above is called **serious delinquency**. It stays the primary outcome, but it is not credit default in 2020-21: in the 2019 sample 92% of the loans that went 90+ days late carried a relief flag at or before that month (disaster, payment deferral P/C or a forbearance/repayment/trial assistance plan F/R/T), 97% were later current again and 0.5% were ever liquidated or REO (2007: 47% liquidated). The **relief-adjusted** outcome (`scripts/prepare_freddie_mac.py --relief-adjusted 2019` -> `loans_2019_relief_adjusted.csv`) removes from the defaults the loans that were relief-flagged at or before their first 90+ day month and never liquidated; they stay in the sample as non-defaults. It is a heuristic: relief was not randomly assigned and some relieved loans were genuinely troubled. The two files hold the same loans and differ only in `default`.
- **Loan age is not calendar time.** Freddie Mac does not advance loan age for every month a delinquent loan misses, so a few recent-vintage loans are still active at the cutoff with age under 36. Those that had not defaulted yet have an unknown outcome and are dropped and counted (24 in the 2022 sample); a loan that already defaulted keeps its known outcome.
- **Not used:** state, MSA, postal code, seller, servicer, dates, CLTV, prepayment penalty (constant in three of four vintages), interest-only (constant).

## What this data can and cannot support

- **Observational, no ground truth.** Lenders set the rate, LTV and terms from the borrower's risk, so a higher rate goes with riskier loans through things this file does not record (points, pricing adjustments, underwriter judgement, income stability). The pipeline's estimates are associations adjusted for what is measured; the placebo test cannot see the rest.
- **The population is loans Freddie Mac acquired**, mostly fully documented fixed-rate loans, not all US mortgages.
- **The vintages are different populations**, so they are never pooled: serious delinquency among kept loans is 15.5% in 2007, 11.8% in 2008, 2.2% in 2010, 1.6% in 2011, 1.4% in 2016, 12.6% in 2019 (relief-adjusted: 1.0%) and 3.5% in 2022 (relief-adjusted: 1.7%).
- **No race, sex or age.** The "sensitive attribute" in Stage 6 is first-time-homebuyer status, which is not a protected class; the fairness flag says only that default rates differ between first-time and repeat buyers.
- **Rate is not a policy lever in the same sense as a fee.** Lowering a rate changes the borrower's payment and may change who applies; the illustrative costs in the config are placeholders, not from the data.
