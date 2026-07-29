# Do FOMC Statements Move Asset Prices Beyond the Rate Decision?

**Evidence from five text representations of Federal Reserve communication, 1994–2023.**

This repository contains the full empirical pipeline for a study of whether the *text* of
FOMC post-meeting statements moves asset prices beyond the interest-rate decision itself.
Statement content is measured five different ways — from a hand-built dictionary to
contextual transformer embeddings — and each is tested against the same high-frequency
benchmark.

Based on a Master's seminar paper at the Finance Center Münster, University of Münster
(graded 1.0). Currently being extended toward a working paper.

---

## Research question

Central banks move markets not only through the policy rate but through the language
of their communication. On 28 January 2004 the FOMC left rates unchanged, yet two- and
five-year Treasury yields moved 20–25 basis points in the following thirty minutes.
The trigger was a change of wording.

This raises a measurement question: **if statement language moves prices, can text
analysis detect it?** And if a text measure finds nothing, is that because there is no
signal — or because the measure is too coarse?

To separate those explanations, the same econometric test is applied to five text
representations of increasing expressiveness, holding the identification, sample, and
outcome variables fixed.

---

## Headline results

**1. The monetary-policy surprise dominates.** Regressing 30-minute asset-price responses
on the Bauer–Swanson high-frequency surprise gives R² of 0.813, 0.638, and 0.482 at the
two-, five-, and ten-year Treasury maturities.

**2. No text measure adds explanatory power beyond it.** Across all five representations
and nine asset-price outcomes, incremental R² never exceeds 0.013 for the interpretable
measures, and joint significance is never attained in the specification consistent with
high-frequency identification.

| Text representation | Type | Result |
|---|---|---|
| Bag-of-words hawkishness index | Lexical count | Null |
| FinBERT | Transformer, financial news | Null |
| CentralBankRoBERTa | Transformer, central-bank text | Null |
| LDA topic intensities | Topic model | Null |
| RoBERTa contextual embeddings | Contextual representation | Null (one marginal exception) |

**3. Domain matching is not the explanation.** A natural objection to a FinBERT null is
that a model trained on financial news misreads central-bank prose. CentralBankRoBERTa,
trained specifically on central-bank communication, returns the same null.

**4. Two apparent positives were diagnosed as artifacts.** Both are documented in the
code rather than discarded:

- Topic *levels* appeared to explain S&P 500 responses (p = 0.005). The effect vanishes
  in the identification-consistent *changes* specification (p = 0.936) and loses
  significance once a post-2008 regime indicator is added (p = 0.092). Topic levels were
  proxying for the policy regime, not for announcement-window content.
- An initial attempt at the Gürkaynak–Sack–Swanson factor decomposition produced a
  degenerate rotation, because the underlying funds-futures series began only in 2010 and
  carried too little variation at the zero lower bound. The diagnostics that reveal this
  are retained in the code.

**5. GSS target/path decomposition, replicated from raw futures data.** Using the SF Fed's
U.S. Monetary Policy Event-Study Database, target and path factors are constructed by
principal components with the GSS identifying restriction (the path factor has zero
loading on the current-month contract). The replication reproduces the published pattern:

| Treasury maturity | R², target factor only | R², adding path factor |
|---|---|---|
| 2-year | 0.114 | 0.839 |
| 5-year | 0.063 | 0.650 |
| 10-year | 0.008 | 0.388 |

This enables the sharpest form of the research question: GSS identify path news from
asset-price covariation and note that it cannot reveal *which* statement content produces
it. The text measures are tested directly against that factor.

---

## Method

**Identification.** High-frequency event study on 239 scheduled FOMC meetings, 1994–2023,
using 30-minute windows bracketing each announcement. All regressions use HC3
heteroskedasticity-robust standard errors.

**Text measures enter as first differences.** A 30-minute announcement return can respond
only to information that is new in the window; the pre-meeting level of a text measure is
already public. The change relative to the previous meeting is the statement surprise, in
direct analogy to the monetary-policy surprise being an innovation rather than a level.
This distinction turns out to matter empirically (see result 4).

**Multi-dimensional measures are tested jointly.** LDA topics and embedding components
enter as blocks and are assessed with robust Wald tests, since individual coefficients are
not separately interpretable.

**Narrative shocks.** Following Hansen, McMahon and Tong (2019), each text measure is
residualised against four pre-meeting macroeconomic state variables to isolate the
component of statement content orthogonal to the observable economic state.

---

## Repository structure

```
├── notebooks/
│   └── main_analysis.ipynb      Full pipeline, end to end
├── src/
│   ├── text_measures.py         BoW, FinBERT, CentralBankRoBERTa, LDA, embeddings
│   ├── gss_factors.py           Target/path factor construction and diagnostics
│   └── regressions.py           HC3 regressions, joint Wald tests, incremental R²
├── data/
│   └── README.md                Download instructions (data not committed)
└── results/                     Output tables
```

---

## Reproducing

```bash
git clone https://github.com/<username>/fomc-statements-asset-prices.git
cd fomc-statements-asset-prices
pip install -r requirements.txt
```

Then follow `data/README.md` to download the two public datasets, and run the notebook
top to bottom. The LDA and regression components need only scikit-learn and statsmodels;
the transformer measures require `transformers` and `torch` and will download model
weights on first use.

Note that one model used in the paper (`gtfintechlab/FOMC-RoBERTa`, a hawkish/dovish
classifier) is an access-gated repository on Hugging Face. The pipeline skips any measure
whose inputs are unavailable, so the analysis runs without it.

---

## Data

Neither dataset is committed to this repository. Both are public:

- **Bauer & Swanson (2023)** monetary-policy surprises and 30-minute asset-price responses.
- **U.S. Monetary Policy Event-Study Database** (Acosta, Ajello, Bauer, Loria and
  Miranda-Agrippino, 2025), SF Fed — high-frequency futures, Treasury, TIPS and equity
  responses for statement, press-conference and minutes windows.

FOMC statements are scraped from the Federal Reserve website by the pipeline itself.

---

## Key references

- Gürkaynak, Sack and Swanson (2005), *International Journal of Central Banking* — target
  and path factor decomposition.
- Hansen, McMahon and Tong (2019), *Journal of Monetary Economics* — LDA topic modelling
  of central-bank communication and the narrative-shock methodology.
- Bauer and Swanson (2023), *NBER Macroeconomics Annual* — high-frequency surprise series.
- Pfeifer and Marohl (2023), *Journal of Finance and Data Science* — CentralBankRoBERTa.
- Blei, Ng and Jordan (2003), *JMLR* — Latent Dirichlet Allocation.

---

## Status

The seminar paper is complete and graded. The extension documented here — the LDA and
transformer measures, and the GSS factor replication — is ongoing work toward a working
paper. Code is provided as-is and will be updated.
