#!/usr/bin/env python3
"""
Shared Actigraphic Risk Component (SHARC) association with known
health descriptors.

Computes, for the paper:
  (1) Bootstrap 95% CIs for the two headline Spearman correlations in the Results
      sentence: rho(SHARC, future-disease burden) and rho(SHARC, BMI).
  (2) A grouped association table for the Extended Data forest plot: raw Spearman
      rho(SHARC, x) + 95% CI for each descriptor, plus a BMI-adjusted partial Spearman
      (activity + sleep rows) as a robustness column.

Deterministic: seed=42, 10,000 percentile bootstrap resamples (subject-level).
Sanity gate: reproduces rho(SHARC,burden)=0.22236241 and rho(SHARC,bmi)=0.48250145.

Run:
  python3 -m ukb_disease.paper_additions.sharc.sharc_descriptor_assoc
(or run the file directly)
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from ukb_disease.paths import UKB_ROOT

# config
SEED = 42
N_BOOT = 10000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sharc_descriptors")
os.makedirs(OUT, exist_ok=True)

SHARC_CSV = os.path.join(HERE, "sharc_per_subject.csv")
CIRC = f"{UKB_ROOT}/circadian_covariates_v3/all.csv"
INTEN = f"{UKB_ROOT}/enmo_intensity_disjoint_split/all.csv"
UKBACT = f"{UKB_ROOT}/paper_additions/caches/ukb_summary_activity/ukb_summary_activity.parquet"

PUB_BURDEN = 0.22236241324342687   # reference rho(SHARC, burden)
PUB_BMI = 0.48250145326622923      # reference rho(SHARC, bmi)

# helpers
def _finite_pair(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def fast_spearman(x, y):
    """Spearman rho = Pearson on (tie-corrected) ranks; recomputed within each resample."""
    rx = rankdata(x); ry = rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return (rx @ ry) / d if d > 0 else np.nan

def spearman_ci(x, y, n_boot=N_BOOT, seed=SEED):
    """Raw Spearman rho(x,y) + percentile bootstrap 95% CI (subject resample).
    Point estimate uses scipy.spearmanr (canonical); bootstrap uses fast_spearman
    (identical estimator: Pearson-on-ranks with tie correction)."""
    x, y = _finite_pair(x, y)
    n = len(x)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = fast_spearman(x[idx], y[idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(rho=float(rho), ci_lo=float(lo), ci_hi=float(hi),
                n=int(n), n_boot=int(len(boot)))

def _resid_on(a, b):
    """Residual of a after linear regression on b (with intercept)."""
    A = np.vstack([np.ones_like(b), b]).T
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    return a - A @ coef

def partial_spearman(x, y, z):
    """Partial Spearman corr(x,y | z): rank-transform, residualise on rank(z), Pearson."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex, ey = _resid_on(rx, rz), _resid_on(ry, rz)
    return float(np.corrcoef(ex, ey)[0, 1])

def partial_spearman_ci(x, y, z, n_boot=N_BOOT, seed=SEED):
    """BMI-adjusted partial Spearman rho(x,y|z) + percentile bootstrap 95% CI."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    rho = partial_spearman(x, y, z)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot[b] = partial_spearman(x[idx], y[idx], z[idx])
        except Exception:
            boot[b] = np.nan
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(rho=float(rho), ci_lo=float(lo), ci_hi=float(hi),
                n=int(n), n_boot=int(len(boot)))

# load + join
df = pd.read_csv(SHARC_CSV)
assert list(df.columns)[:9] == ["eid","SHARC","SHARC_pct","SHARC_decile","burden","age","sex","bmi","died6"]
n0 = len(df)

circ = pd.read_csv(CIRC, usecols=["eid","enmo_mean","enmo_M10","enmo_RA","pwake_mean","pwake_RA"])
inten = pd.read_csv(INTEN, usecols=["eid","enmo_frac_mvpa","enmo_frac_sed"])
ukb = pd.read_parquet(UKBACT, columns=["eid","90012-0.0"]).rename(columns={"90012-0.0":"ukb_accel"})

for extra in (circ, inten, ukb):
    df = df.merge(extra.drop_duplicates("eid"), on="eid", how="left")
assert len(df) == n0, f"join changed row count {n0}->{len(df)}"

# sanity gate
g_burden = spearmanr(df["SHARC"], df["burden"]).statistic
mbmi = np.isfinite(df["SHARC"]) & np.isfinite(df["bmi"])
g_bmi = spearmanr(df.loc[mbmi,"SHARC"], df.loc[mbmi,"bmi"]).statistic
print(f"[sanity] rho(SHARC,burden) = {g_burden:.11f}  (published {PUB_BURDEN:.11f})")
print(f"[sanity] rho(SHARC,bmi)    = {g_bmi:.11f}  (published {PUB_BMI:.11f}, n_bmi={int(mbmi.sum())})")
assert abs(g_burden - PUB_BURDEN) < 1e-6, "burden Spearman does not match published value"
assert abs(g_bmi - PUB_BMI) < 1e-6, "BMI Spearman does not match published value"
assert int(mbmi.sum()) == 5240, f"BMI n mismatch: {int(mbmi.sum())}"
print("[sanity] PASS. SHARC reproduces both reference Spearman values.\n")

# descriptor table
# (group, key, label, adjust_bmi)
DESCR = [
    ("Demographics",               "age",            "Age",                              False),
    ("Demographics",               "sex",            "Sex (male vs female)",             False),
    ("Demographics",               "bmi",            "Body-mass index",                  False),
    ("Activity",                   "enmo_mean",      "Mean acceleration (ENMO)",         True),
    ("Activity",                   "enmo_M10",       "M10 (most-active 10 h)",           True),
    ("Activity",                   "enmo_RA",        "Rest-activity amplitude",          True),
    ("Activity",                   "enmo_frac_mvpa", "MVPA fraction",                    True),
    ("Activity",                   "enmo_frac_sed",  "Sedentary fraction",               True),
    ("Activity",                   "ukb_accel",      "UKB overall acceleration",         True),
    ("Sleep (actigraphy-derived)", "pwake_mean",     "Wake fraction",                    True),
    ("Sleep (actigraphy-derived)", "pwake_RA",       "Sleep-wake rhythm",                True),
    ("Outcomes (reference)",       "burden",         "Future disease burden",            False),
    ("Outcomes (reference)",       "died6",          "6-year mortality",                 False),
]

rows = []
sharc = df["SHARC"].to_numpy(float)
bmi = df["bmi"].to_numpy(float)
for group, key, label, adj in DESCR:
    x = df[key].to_numpy(float)
    raw = spearman_ci(sharc, x)
    rec = dict(group=group, key=key, label=label,
               n=raw["n"], rho=raw["rho"], ci_lo=raw["ci_lo"], ci_hi=raw["ci_hi"])
    if adj:
        pa = partial_spearman_ci(sharc, x, bmi)
        rec.update(n_bmiadj=pa["n"], rho_bmiadj=pa["rho"],
                   ci_lo_bmiadj=pa["ci_lo"], ci_hi_bmiadj=pa["ci_hi"])
    else:
        rec.update(n_bmiadj=np.nan, rho_bmiadj=np.nan,
                   ci_lo_bmiadj=np.nan, ci_hi_bmiadj=np.nan)
    rows.append(rec)
    adjs = (f"   | BMI-adj {pa['rho']:+.3f} [{pa['ci_lo']:+.3f},{pa['ci_hi']:+.3f}] n={pa['n']}"
            if adj else "")
    print(f"{group:28s} {label:30s} rho={raw['rho']:+.3f} "
          f"[{raw['ci_lo']:+.3f},{raw['ci_hi']:+.3f}] n={raw['n']}{adjs}")

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "assoc_table.csv"), index=False)

# headline CIs
headline = {
    "burden": spearman_ci(sharc, df["burden"].to_numpy(float)),
    "bmi":    spearman_ci(sharc, bmi),
}
with open(os.path.join(OUT, "headline_ci.json"), "w") as f:
    json.dump({"seed": SEED, "n_boot": N_BOOT, **headline,
               "published": {"burden": PUB_BURDEN, "bmi": PUB_BMI}}, f, indent=2)

print("\n=== HEADLINE (for the Results sentence) ===")
for k in ("burden","bmi"):
    h = headline[k]
    print(f"  {k:7s} rho={h['rho']:.4f}  95% CI [{h['ci_lo']:.3f}, {h['ci_hi']:.3f}]  n={h['n']}")
print(f"\nWrote: {OUT}/assoc_table.csv , {OUT}/headline_ci.json")
