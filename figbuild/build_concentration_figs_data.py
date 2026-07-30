"""Compute source data for the risk-concentration and horizon Extended Data figures.

Needs sklearn. Read-only on the seed-averaged hazard matrix; writes CSVs to
paper_additions/concentration_figs/. Everything uses the wrist-embedding model at
the paper's 6-year horizon and top-5% operating point.

Outputs:
  concentration_ppv.csv    top-N diseases by PPV@top-5% (6y) + bootstrap CI + enrichment
  concentration_auprc.csv  top-N diseases by AUPRC (6y) + bootstrap CI + lift
  horizon_sweep.csv        mean AUPRC / PPV@top-5% / prevalence across 1-8y horizons
  pr_at_horizon.csv        per-disease precision & recall & F1 at top-5% (6y)
  six_disease_horizon.csv  P/R/F1 at top-5% across horizons for six headline diseases
"""
import os, sys, json
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_tmp"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib as F
from pathlib import Path

UKB_ROOT = os.environ.get("UKB_ROOT", "data")
SEED_AVG = Path(f"{UKB_ROOT}/paper_additions/runs/embedding_seed_avg")
OUT = Path(f"{UKB_ROOT}/paper_additions/concentration_figs")
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 6.0
FRAC = 0.05
N_TOP = 20
HORIZONS = list(range(1, 9))
# Six headline diseases for the per-disease horizon panel (neuro flagship + well-powered cardiometabolic/mortality)
SIX = [("time_to_death", "All-cause mortality"), ("NS_324.11", "Parkinson's disease"),
       ("CV_424", "Heart failure"), ("CV_401.1", "Essential hypertension"),
       ("EM_202.2", "Type 2 diabetes"), ("CV_416.2", "Atrial fibrillation")]

H = np.load(SEED_AVG / "test_hazards.npy")
E = np.load(SEED_AVG / "test_is_event.npy")
T = np.load(SEED_AVG / "test_event_times.npy")
M = np.load(SEED_AVG / "test_eval_mask.npy").astype(bool)
phecodes = json.load(open(SEED_AVG / "config.json"))["phecodes"]
pm = F.phecode_map()

def name_cat(ph):
    if ph == "time_to_death":
        return "All-cause mortality", "Mortality"
    if ph in pm.index:
        return pm.loc[ph, "PhecodeString"], pm.loc[ph, "PhecodeCategory"]
    return ph, "Other"

def label_at(j, m, h):
    """Positive = incident event within h years (mortality = death within h years)."""
    return ((E[m, j] == 1) & (T[m, j] <= h)).astype(int)

def topk_ppv(scores, y, frac=FRAC):
    n = len(y); k = max(1, int(np.ceil(frac * n)))
    order = np.argpartition(-scores, k - 1)[:k]
    tp = int(y[order].sum())
    return tp / k, tp / max(1, int(y.sum())), k

def ppv_boot(scores, y, frac=FRAC, n_boot=500, seed=42):
    n = len(y); k = max(1, int(np.ceil(frac * n)))
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, n, size=(n_boot, n))
    bs = scores[bi]; by = y[bi]
    part = np.argpartition(-bs, k - 1, axis=1)[:, :k]
    boots = np.take_along_axis(by, part, axis=1).mean(axis=1)
    return tuple(np.percentile(boots, [2.5, 97.5]))

def auprc_boot(scores, y, n_boot=300, seed=13):
    rng = np.random.default_rng(seed); n = len(y); boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() >= 2:
            boots.append(average_precision_score(y[idx], scores[idx]))
    return tuple(np.nanpercentile(boots, [2.5, 97.5]))

# --------------------------------------------------------------------------- #
# 1. Per-disease PPV@5% + AUPRC at 6y (qualifying diseases), then top-N + boot
# --------------------------------------------------------------------------- #
rows = []
for j, ph in enumerate(phecodes):
    m = M[:, j]; n_eval = int(m.sum())
    if n_eval < 200:
        continue
    n_pos_total = int(E[m, j].sum())
    if n_pos_total < 50:                    # matches the reference inclusion floor
        continue
    s = H[m, j].astype(float)
    y = label_at(j, m, HORIZON)
    n_pos = int(y.sum())
    if n_pos < 10:
        continue
    prev = float(y.mean())
    ppv, recall, _ = topk_ppv(s, y)
    auprc = float(average_precision_score(y, s))
    nm, cat = name_cat(ph)
    rows.append(dict(phecode=ph, name=nm, category=cat, p_idx=j, n_eval=n_eval,
                     n_pos=n_pos, prevalence=prev, ppv=ppv, recall=recall, auprc=auprc,
                     enrichment=ppv / prev, lift=auprc / prev))
base = pd.DataFrame(rows)
print(f"qualifying diseases (n_eval>=200, >=50 total events, >=10 by {int(HORIZON)}y): {len(base)}")

def top_boot(metric, boot_fn):
    top = base.nlargest(N_TOP, metric).copy()
    los, his = [], []
    for _, r in top.iterrows():
        j = int(r.p_idx); m = M[:, j]
        lo, hi = boot_fn(H[m, j].astype(float), label_at(j, m, HORIZON))
        los.append(float(lo)); his.append(float(hi))
    top[f"{metric}_lo"] = los; top[f"{metric}_hi"] = his
    return top

top_boot("ppv", ppv_boot).to_csv(OUT / "concentration_ppv.csv", index=False)
top_boot("auprc", auprc_boot).to_csv(OUT / "concentration_auprc.csv", index=False)

# --------------------------------------------------------------------------- #
# 2. Horizon sweep (mean AUPRC / PPV@5% / prevalence over phecodes >=20 pos)
# --------------------------------------------------------------------------- #
hz_rows = []
for h in HORIZONS:
    aps, ppvs, prevs = [], [], []
    for j in range(len(phecodes)):
        m = M[:, j]
        if m.sum() < 100:
            continue
        y = label_at(j, m, h)
        if y.sum() < 20:
            continue
        s = H[m, j].astype(float)
        aps.append(average_precision_score(y, s))
        ppvs.append(topk_ppv(s, y)[0]); prevs.append(float(y.mean()))
    hz_rows.append(dict(horizon=h, n_phecodes=len(aps), mean_auprc=float(np.mean(aps)),
                        mean_ppv_top5=float(np.mean(ppvs)), mean_prevalence=float(np.mean(prevs))))
pd.DataFrame(hz_rows).to_csv(OUT / "horizon_sweep.csv", index=False)

# --------------------------------------------------------------------------- #
# 3. PR at top-5% (6y), per disease (>=30 positives so P/R are stable)
# --------------------------------------------------------------------------- #
pr_rows = []
for j, ph in enumerate(phecodes):
    m = M[:, j]; n_eval = int(m.sum())
    if n_eval < 200:
        continue
    y = label_at(j, m, HORIZON); n_pos = int(y.sum())
    if n_pos < 30:
        continue
    s = H[m, j].astype(float)
    ppv, recall, _ = topk_ppv(s, y)
    f1 = 2 * ppv * recall / (ppv + recall) if (ppv + recall) > 0 else 0.0
    nm, cat = name_cat(ph)
    pr_rows.append(dict(phecode=ph, name=nm, category=cat, n_pos=n_pos,
                        precision=ppv, recall=recall, f1=f1))
pd.DataFrame(pr_rows).to_csv(OUT / "pr_at_horizon.csv", index=False)
print(f"PR-scatter diseases (>=30 pos by {int(HORIZON)}y): {len(pr_rows)}")

# --------------------------------------------------------------------------- #
# 4. Six-disease P/R/F1 across horizons
# --------------------------------------------------------------------------- #
six_rows = []
for ph, label in SIX:
    j = phecodes.index(ph); m = M[:, j]; s = H[m, j].astype(float)
    for h in HORIZONS:
        y = label_at(j, m, h); npos = int(y.sum())
        if npos < 5:
            six_rows.append(dict(phecode=ph, label=label, horizon=h, n_pos=npos,
                                 precision=np.nan, recall=np.nan, f1=np.nan)); continue
        ppv, recall, _ = topk_ppv(s, y)
        f1 = 2 * ppv * recall / (ppv + recall) if (ppv + recall) > 0 else 0.0
        six_rows.append(dict(phecode=ph, label=label, horizon=h, n_pos=npos,
                             precision=ppv, recall=recall, f1=f1))
pd.DataFrame(six_rows).to_csv(OUT / "six_disease_horizon.csv", index=False)

# --------------------------------------------------------------------------- #
# spot-checks
# --------------------------------------------------------------------------- #
pp = pd.read_csv(OUT / "concentration_ppv.csv")
au = pd.read_csv(OUT / "concentration_auprc.csv")
hz = pd.read_csv(OUT / "horizon_sweep.csv")
print("\ntop-5 by PPV@5%:")
print(pp.head(5)[["name", "n_pos", "prevalence", "ppv", "enrichment", "ppv_lo", "ppv_hi"]].to_string(index=False))
print("\ntop-5 by AUPRC:")
print(au.head(5)[["name", "n_pos", "prevalence", "auprc", "lift", "auprc_lo", "auprc_hi"]].to_string(index=False))
print("\nhorizon sweep:")
print(hz.to_string(index=False))
print("DONE")
