"""
Regression utilities for the FOMC statement event study.

All specifications are estimated by OLS with HC3 heteroskedasticity-robust
standard errors. Multi-dimensional text measures (LDA topic intensities,
embedding principal components) are assessed jointly with robust Wald tests,
since their individual coefficients are not separately interpretable.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_hc3(data, y, x_cols):
    """OLS with HC3 robust standard errors on the non-missing subsample."""
    sub = data[[y] + x_cols].dropna()
    return sm.OLS(sub[y], sm.add_constant(sub[x_cols])).fit(cov_type="HC3")


def joint_p(model, var_names):
    """
    HC3 robust joint Wald p-value for H0: all coefficients in `var_names` are zero.

    Used wherever a text measure enters as a block (LDA topics, embedding
    components), where testing coefficients one at a time would both lose power
    and invite multiple-testing problems.
    """
    idx = {name: i for i, name in enumerate(model.params.index)}
    R = np.zeros((len(var_names), len(model.params)))
    for r, v in enumerate(var_names):
        R[r, idx[v]] = 1.0
    return float(model.wald_test(R, scalar=True).pvalue)


def incremental_r2(data, y, feat_cols, control="MPS"):
    """
    Incremental explanatory power of a text measure over a control-only baseline.

    Both models are fit on the *same* non-missing subsample, so the difference in
    R-squared is comparable. Returns (full_model, delta_r2).

    This is the natural summary when comparing text representations of different
    dimensionality: a dictionary score has one coefficient, an embedding block has
    five, and delta R-squared with a joint test puts them on a common footing.
    """
    full = run_hc3(data, y, [control] + feat_cols)
    sub = data[[y, control] + feat_cols].dropna()
    base = sm.OLS(sub[y], sm.add_constant(sub[[control]])).fit()
    return full, full.rsquared - base.rsquared


def residualize(data, y, controls):
    """
    Narrative-shock first stage, following Hansen, McMahon and Tong (2019).

    Regresses a text measure on pre-meeting macroeconomic state variables and
    returns the residual, which is the component of statement content that is not
    predictable from publicly observable macro conditions.
    """
    sub = data[[y] + controls].dropna()
    fit = sm.OLS(sub[y], sm.add_constant(sub[controls])).fit()
    resid = pd.Series(np.nan, index=data.index)
    resid.loc[sub.index] = fit.resid
    return resid, fit


def stars(p):
    """Significance markers at the 1, 5 and 10 percent levels."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def comparison_table(data, methods, outcomes, control="MPS"):
    """
    Build the headline comparison: text representations x asset-price outcomes,
    reporting incremental R-squared over the control-only baseline with joint
    significance stars.

    `methods` maps a display name to the list of columns that measure represents,
    e.g. {"BoW dictionary": ["d_BoW"], "LDA topics": ["d_topic_0", ...]}.
    Methods whose columns are absent from `data` are skipped, so the table still
    builds when a measure could not be computed.
    """
    rows = []
    for name, cols in methods.items():
        if not all(c in data.columns for c in cols):
            continue
        row = {"method": name}
        for y in outcomes:
            full, d = incremental_r2(data, y, cols, control)
            row[y] = f"{d:+.3f}{stars(joint_p(full, cols))}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")
