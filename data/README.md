# Data

No data files are committed to this repository. All sources are public.

## 1. Bauer–Swanson monetary policy surprises

30-minute asset-price responses around FOMC announcements and the high-frequency
monetary-policy surprise `MPS`.

- Source: https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/
- File used: `monetary-policy-surprises-data.xlsx`
- Place in: `data/`

Variables used: `MPS`, `MPS_ORTH`, `TNOTE02`, `TNOTE05`, `TNOTE10`, `SP500`,
`SP500_emini`, and the pre-meeting macro state variables `SP500_3M`, `SLOPE_3M`,
`BCOM_3M`, `NFP_12M`.

## 2. U.S. Monetary Policy Event-Study Database (USMPD)

High-frequency futures, Treasury, TIPS and equity responses, reported separately for
statement, press-conference and minutes windows. Used to construct the
Gürkaynak–Sack–Swanson target and path factors.

- Source: https://sffed.us/usmpd
- File used: `USMPD.xlsx`
- Place in: `data/`

Sheets used: `Statements` (239 scheduled meetings, 1994–2023, complete coverage of
`MP1`, `FF1`–`FF6`, `ED1`–`ED8`, `UST2Y`, `UST5Y`, `UST10Y`) and `Press Conferences`
(93 events from 2011, same variables in the press-conference window).

Cite as: Acosta, M., A. Ajello, M. Bauer, F. Loria and S. Miranda-Agrippino (2025),
"Financial Market Effects of FOMC Communication: Evidence from a New Event-Study
Database," FRBSF Working Paper 2025-30.

## 3. FOMC statement corpus

Scraped directly from the Federal Reserve website by the pipeline; no manual download
required. Statements are written to `data/fomc_statements/` as one text file per meeting.

- 1994–2020: historical materials archive, one page per year
- 2021–2023: press-release pages

The raw corpus contains 224 statements. 209 correspond to scheduled meetings and enter
the analysis; the remaining 15 are intermeeting emergency statements (for example
January and September 2001, October 2008, March 2020) and are excluded to keep the
high-frequency window comparable.
