# Main results

Sample: 239 scheduled FOMC meetings, February 1994 – December 2023.
209 meetings have a matched post-meeting statement; regressions in first
differences use 208. All standard errors are HC3 robust.

## 1. Baseline: the monetary-policy surprise

| Outcome | Coefficient | HC3 s.e. | R² | N |
|---|---|---|---|---|
| 2Y Treasury | 0.831 | 0.053 | 0.813 | 239 |
| 5Y Treasury | 0.744 | 0.057 | 0.638 | 239 |
| 10Y Treasury | 0.520 | 0.049 | 0.482 | 239 |
| S&P 500 | −4.728 | 0.682 | 0.244 | 239 |
| 10Y − 2Y slope | −0.311 | 0.045 | 0.339 | 239 |

## 2. Incremental explanatory power of five text representations

Increase in R² over the surprise-only baseline. Stars denote joint significance
of the text coefficients (HC3 robust Wald test).

| Measure | 2Y | 5Y | 10Y | S&P 500 | 10Y−2Y |
|---|---|---|---|---|---|
| BoW dictionary | +0.002 | +0.003 | +0.007 | +0.002 | +0.002 |
| FinBERT | +0.000 | +0.000 | +0.000 | +0.010 | +0.000 |
| CentralBankRoBERTa | +0.005* | +0.004 | +0.007 | +0.005 | +0.000 |
| LDA topics | +0.004 | +0.006 | +0.011 | +0.006 | +0.036 |
| RoBERTa embeddings | +0.003 | +0.007 | +0.009 | +0.069*** | +0.023** |

The embedding row uses component *levels* and is diagnosed below.

## 3. Diagnosis of the two apparent positives

**Topic levels and the S&P 500.** Significant in levels (p = 0.005); absent in the
identification-consistent changes specification (p = 0.936); survives excluding the
2008 and 2020 crisis windows (p = 0.012); loses significance once a post-2008
regime indicator is added (p = 0.092). Topic levels track the policy regime rather
than announcement-window content.

**Embedding levels and equities.** Same pattern. The first principal component
correlates −0.719 with the calendar year, so embedding levels substantially encode
the era in which a statement was written.

**One marginal survivor.** Embedding *changes* remain associated with the 10Y−2Y
slope (p = 0.009), concentrated in a single component (p = 0.001) that separates
the 2007–08 crisis statements from routine ones. The association weakens to
p = 0.095 outside the zero lower bound and p = 0.077 excluding acute-crisis
statements, and does not appear in the other two slope measures. Reported as
suggestive rather than conclusive.

## 4. GSS target and path factor replication

Constructed from the U.S. Monetary Policy Event-Study Database by principal
components with the Gürkaynak–Sack–Swanson identifying restriction.

Diagnostics: corr(target, path) = 0.000; corr(path, current-month contract) = 0.000
(the restriction); corr(target, current-month contract) = 0.978; corr(path,
four-quarter contract) = −0.921.

| Treasury maturity | R², target only | R², adding path |
|---|---|---|
| 2Y | 0.114 | 0.839 |
| 5Y | 0.063 | 0.650 |
| 10Y | 0.008 | 0.388 |

The target factor is insignificant at ten and thirty years while the path factor is
significant at every maturity, reproducing the pattern in GSS (2005).
