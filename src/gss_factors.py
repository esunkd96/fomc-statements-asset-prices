"""
Gurkaynak, Sack and Swanson (2005) target and path factor decomposition.

FOMC announcements move a whole strip of interest-rate futures at once. GSS
show this variation is well summarised by two orthogonal factors:

    target factor : the surprise in the current policy rate
    path factor   : news about the expected future path, identified by the
                    restriction that it has zero loading on the current-month
                    contract

The path factor is the object of interest for a study of communication, because
it isolates announcement news that is *not* the rate decision. GSS note that,
being identified from asset-price covariation alone, it cannot reveal which
statement content produces it. That is precisely the question the text measures
in this project are asked to answer.

Constructed here from the SF Fed's U.S. Monetary Policy Event-Study Database
(Acosta et al. 2025), which covers all 239 scheduled meetings from 1994.

WHY THE DIAGNOSTICS MATTER
--------------------------
An earlier attempt used a futures panel whose current-month series began only in
2010 and carried little variation at the zero lower bound. The rotation then
assigned the dominant principal component to the *path* factor rather than the
target factor, and the resulting "path" factor correlated -0.99 with the
aggregate surprise: the decomposition had silently collapsed. `diagnostics()`
below checks for exactly that failure, and should be run before the factors are
used in any regression.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# GSS use the current-month and three-month-ahead federal funds futures together
# with the two-, three- and four-quarter-ahead eurodollar contracts.
GSS_CONTRACTS = ["MP1", "FF4", "ED2", "ED3", "ED4"]


def load_usmpd_statements(path, start="1994-01-01", end="2023-12-31"):
    """Scheduled-meeting statement-window surprises from USMPD.xlsx."""
    d = pd.read_excel(path, sheet_name="Statements")
    d = d[(d["Unscheduled"] == 0) & (d["Date"] >= start) & (d["Date"] <= end)]
    return d.reset_index(drop=True)


def build_factors(df, contracts=GSS_CONTRACTS):
    """
    Extract target and path factors from a panel of futures surprises.

    Standardises the contracts, takes the first two principal components, and
    rotates them so that the second factor has no loading on the first contract
    (the current-month rate). Both factors are returned standardised.

    Returns (factors_df, info) where info carries the loadings and the share of
    variance explained, both of which are reported in the paper.
    """
    sub = df[["Date"] + contracts].dropna().reset_index(drop=True)

    X = sub[contracts].values
    Xs = (X - X.mean(0)) / X.std(0)

    pca = PCA(n_components=len(contracts), random_state=42).fit(Xs)
    F = pca.transform(Xs)[:, :2]
    F = (F - F.mean(0)) / F.std(0)

    # loadings of the two components on the current-month contract
    b = np.linalg.lstsq(F, Xs[:, 0], rcond=None)[0]

    # rotation: target spans the current-rate direction, path is its complement
    target = b[0] * F[:, 0] + b[1] * F[:, 1]
    path = -b[1] * F[:, 0] + b[0] * F[:, 1]

    sub["TARGET"] = (target - target.mean()) / target.std()
    sub["PATH"] = (path - path.mean()) / path.std()

    info = {
        "n": len(sub),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "loadings_on_current_contract": b,
        "standardised_contracts": Xs,
    }
    return sub[["Date", "TARGET", "PATH"] + contracts], info


def diagnostics(factors, info, contracts=GSS_CONTRACTS, verbose=True):
    """
    Check that the decomposition is well identified.

    Healthy output looks like:
      corr(TARGET, PATH)              ~  0      orthogonality holds by construction
      corr(PATH, current contract)    ~  0      the GSS identifying restriction
      corr(TARGET, current contract)  high      target really is the rate surprise
      corr(PATH, far contract)        high      path really is longer-horizon news

    A degenerate rotation shows up as a *low* correlation between the target
    factor and the current-month contract, with the far-horizon variation loading
    on the path factor instead.
    """
    Xs = info["standardised_contracts"]
    t, p = factors["TARGET"].values, factors["PATH"].values

    out = {
        "corr_target_path": np.corrcoef(t, p)[0, 1],
        "corr_path_current": np.corrcoef(p, Xs[:, 0])[0, 1],
        "corr_target_current": np.corrcoef(t, Xs[:, 0])[0, 1],
        "corr_path_far": np.corrcoef(p, Xs[:, -1])[0, 1],
        "loadings": info["loadings_on_current_contract"],
        "explained_variance": info["explained_variance_ratio"][:2],
    }

    if verbose:
        print(f"n = {info['n']}")
        print(f"explained variance (first two): {np.round(out['explained_variance'], 3)}")
        print(f"loadings on {contracts[0]}: {np.round(out['loadings'], 4)}")
        print(f"corr(TARGET, PATH)      = {out['corr_target_path']:+.6f}   (should be ~0)")
        print(f"corr(PATH, {contracts[0]})        = {out['corr_path_current']:+.6f}   (restriction, ~0)")
        print(f"corr(TARGET, {contracts[0]})      = {out['corr_target_current']:+.3f}   (should be high)")
        print(f"corr(PATH, {contracts[-1]})        = {out['corr_path_far']:+.3f}   (should be high)")

        if abs(out["corr_target_current"]) < 0.5:
            print("\nWARNING: target factor is weakly related to the current-month "
                  "contract. The rotation may have collapsed; check whether that "
                  "contract has adequate variation in this sample.")

    return out


def validate(factors, outcomes_df, outcomes, verbose=True):
    """
    Replicate the GSS validation: adding the path factor should sharply raise the
    explanatory power for longer-maturity yields, and the target factor should
    lose significance as maturity increases.

    GSS (2005, p. 83) report R-squared rising from 0.41, 0.19 and 0.08 to 0.94,
    0.80 and 0.74 at the two-, five- and ten-year maturities.
    """
    import statsmodels.api as sm

    m = factors.merge(outcomes_df, on="Date", how="left")
    rows = []
    for y in outcomes:
        s = m[[y, "TARGET", "PATH"]].dropna()
        both = sm.OLS(s[y], sm.add_constant(s[["TARGET", "PATH"]])).fit(cov_type="HC3")
        tgt = sm.OLS(s[y], sm.add_constant(s[["TARGET"]])).fit()
        rows.append({
            "outcome": y,
            "target": round(both.params["TARGET"], 4),
            "p_target": round(both.pvalues["TARGET"], 3),
            "path": round(both.params["PATH"], 4),
            "p_path": round(both.pvalues["PATH"], 3),
            "R2_target_only": round(tgt.rsquared, 3),
            "R2_both": round(both.rsquared, 3),
            "N": int(both.nobs),
        })
    table = pd.DataFrame(rows)
    if verbose:
        print(table.to_string(index=False))
    return table
