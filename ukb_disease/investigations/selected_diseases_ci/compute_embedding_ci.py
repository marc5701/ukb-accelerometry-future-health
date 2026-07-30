"""embedding per-disease concordance plus a 1,000-resample subject bootstrap CI for the
highlighted diseases. Point estimate matches per_disease_c.json (same harness); the CI
is the embedding model's own bootstrap so the highlighted-disease figure and the full per-disease
table stay internally consistent. Run: python3 compute_embedding_ci.py
"""
import sys, json, os
import numpy as np
from ukb_disease.benchmark.concordance_benchmark import load_model, harrell_counts
from ukb_disease.benchmark.per_disease_c import phecode_names
from ukb_disease.paths import UKB_ROOT

RUNS = f"{UKB_ROOT}/runs_cox"
HERE = os.path.dirname(os.path.abspath(__file__))
NBOOT = 1000

# label -> phecode (canonical), for the highlighted diseases
WANT = [
    ("Parkinson's", "NS_324.11"), ("Alzheimer's", "NS_328.11"), ("Dementias", "NS_328.1"),
    ("Heart failure", "CV_428.2"), ("Aortic stenosis", None), ("Ischemic heart", None),
    ("Obesity", "EM_236.1"), ("Type 2 diabetes", None), ("COPD", None), ("Sleep apnea", "NS_333.1"),
    ("Atrial fibrillation", None), ("Repeated falls", None), ("Abnormality of gait", None),
    ("Chronic kidney", None), ("Major depressive", None), ("All-cause mortality", "time_to_death"),
]

# resolve missing phecodes by label substring from the phecode map
import pandas as pd
pm = pd.read_csv(f"{UKB_ROOT}/metadata/Phecode_mapX_filtered.csv")
pm = pm.groupby("Phecode").first()[["PhecodeString"]]
def find_ph(sub):
    m = pm[pm.PhecodeString.str.contains(sub, case=False, na=False, regex=False)]
    return m.index[0] if len(m) else None

mean_h, _bs, base, seeds = load_model(RUNS, "embedding_disjoint_split")
T, E, M = base["T"], base["E"], base["M"]
names = phecode_names(RUNS, "embedding_disjoint_split", T.shape[1])
col = {n: j for j, n in enumerate(names)}

out = {}
rng = np.random.default_rng(0)
for label, ph in WANT:
    if ph is None:
        ph = find_ph(label)
    if ph is None or ph not in col:
        print(f"  MISS {label} ({ph})"); continue
    j = col[ph]; m = M[:, j].astype(bool)
    h, t, e = mean_h[m, j], T[m, j], E[m, j]
    conc, comp = harrell_counts(h, t, e)
    C = conc / comp
    idx = np.arange(len(h)); cs = []
    for _ in range(NBOOT):
        bi = rng.choice(idx, len(idx), replace=True)
        cc, cp = harrell_counts(h[bi], t[bi], e[bi])
        if cp > 0:
            cs.append(cc / cp)
    lo, hi = np.percentile(cs, [2.5, 97.5])
    out[ph] = dict(label=label, phecode=ph, C=round(float(C), 4),
                   CI_lo=round(float(lo), 4), CI_hi=round(float(hi), 4),
                   n_event=int(e.sum()), n_eval=int(m.sum()))
    print(f"  {label:22s} {ph:12s} C={C:.4f} [{lo:.4f},{hi:.4f}] n={int(e.sum())}")

json.dump(dict(nboot=NBOOT, seeds=list(seeds), diseases=out),
          open(os.path.join(HERE, "embedding_ci.json"), "w"), indent=2)
print("saved embedding_ci.json")
