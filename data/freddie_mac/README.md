# Freddie Mac loan data

The only file here that is committed. Everything else in this folder (`raw/` and `loans_<year>.csv`) is **gitignored**: the data are a registered download and I have not verified Freddie Mac's redistribution terms, so nothing derived from them is committed either. Read the terms on the download page before sharing any of it.

## Where the data come from

Freddie Mac's **Single-Family Loan-Level Dataset**, Standard sample: a simple random sample of 50,000 loans from each origination year, fixed-rate mortgages that Freddie Mac bought or that back its securities. Free, after registering at Freddie Mac's Clarity data-download page. The general user guide (January 2026) is at https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf. Vintages used: **2007, 2008, 2010, 2011** (a 2009 sample is a mixed crisis/HARP year and was skipped; 2010-11 are the calm, refinance-heavy years).

```
data/freddie_mac/raw/sample_2007/sample_orig_2007.txt   origination, pipe-delimited, no header
data/freddie_mac/raw/sample_2007/sample_perf_2007.txt   monthly performance (3.0M rows)
...same for 2008, 2010, 2011
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
- **Not used:** state, MSA, postal code, seller, servicer, dates, CLTV, prepayment penalty (constant in three of four vintages), interest-only (constant).

## What this data can and cannot support

- **Observational, no ground truth.** Lenders set the rate, LTV and terms from the borrower's risk, so a higher rate goes with riskier loans through things this file does not record (points, pricing adjustments, underwriter judgement, income stability). The pipeline's estimates are associations adjusted for what is measured; the placebo test cannot see the rest.
- **The population is loans Freddie Mac acquired**, mostly fully documented fixed-rate loans, not all US mortgages.
- **The vintages are different populations**, so they are never pooled: 2007 has 15.5% serious delinquency among kept loans, 2008 11.8%, 2010 2.2%, 2011 1.6%.
- **No race, sex or age.** The "sensitive attribute" in Stage 6 is first-time-homebuyer status, which is not a protected class; the fairness flag says only that default rates differ between first-time and repeat buyers.
- **Rate is not a policy lever in the same sense as a fee.** Lowering a rate changes the borrower's payment and may change who applies; the illustrative costs in the config are placeholders, not from the data.
