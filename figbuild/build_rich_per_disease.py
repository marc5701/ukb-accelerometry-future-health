"""Assemble the per-disease rich metrics table + against-chance figure source data.

Needs sklearn. Read-only on all sources; writes only two CSVs under
paper_additions/rich_per_disease/. Everything is derived from the frozen seed-averaged
hazard matrix and the existing per-disease result files, with no modeling rerun.

Per disease (over eval-masked subjects m): score=test_hazards[m,j], y=test_is_event[m,j],
prevalence=y.mean().
  incident C            <- benchmark/output/disjoint_split_v1/per_disease_c.json (canonical)
  demographics C        <- investigations/clinical_baseline/output/per_disease_3arm.csv (age+sex+BMI)
  lead time (yr)        =  mean(test_event_times[m & y==1, j])   (wear -> diagnosis)
  AUPRC / fold          =  average_precision_score(y,score) ; fold = AUPRC / prevalence
  PPV@top-5% / fold     =  precision in the top-5% highest-risk ; fold = PPV / prevalence
  recall@5%, F1@5%      =  at the same top-5% budget
  prevalent screening   <- screening_prevalent_v1/same_dist_cv (wrist logistic_emb + demo AUROC/AUPRC)
Row kept if n_event>=10 (hard cutoff). (in_selected_diseases is retained as metadata only.)
"""
import os, sys, json
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_tmp"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib as F  # phecode_map, path helpers, canonical roots

UKB = F.UKB
SEED_AVG = UKB / "paper_additions" / "runs" / "embedding_seed_avg"
OUTDIR = UKB / "paper_additions" / "rich_per_disease"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOP_FRAC = 0.05  # top-5% screening budget

# --------------------------------------------------------------------------- #
# Load the frozen seed-averaged matrix (the exact object behind the headline C)
# --------------------------------------------------------------------------- #
H = np.load(SEED_AVG / "test_hazards.npy")          # (N, P) log-hazard, higher = earlier event
E = np.load(SEED_AVG / "test_is_event.npy")         # (N, P) 6-yr incident event indicator
T = np.load(SEED_AVG / "test_event_times.npy")      # (N, P) years wear -> event/censor
M = np.load(SEED_AVG / "test_eval_mask.npy").astype(bool)   # (N, P) evaluable (subject,disease)
phecodes = json.load(open(SEED_AVG / "config.json"))["phecodes"]
N, P = H.shape
assert len(phecodes) == P, (len(phecodes), P)

# --------------------------------------------------------------------------- #
# Canonical incident C + demographics C + prevalent screening + selected_diseases
# --------------------------------------------------------------------------- #
pdc = json.load(open(F.BENCH / "per_disease_c.json"))["embedding_disjoint_split"]["per_disease"]

clin = pd.read_csv(UKB / "investigations" / "clinical_baseline" / "output" / "per_disease_3arm.csv")
clin = clin.set_index("phecode")

scr_path = F.oak_path("mdige/results/ukb_disease/screening_prevalent_v1/same_dist_cv/screening_results_primary.csv")
scr = pd.read_csv(scr_path)
scr_wrist = scr[scr.arm == "logistic_emb"].set_index("phecode")
scr_demo = scr[scr.arm == "demo"].set_index("phecode")

wl = json.load(open(UKB / "benchmark" / "selected_diseases_frozen.json"))
protected = set()
for tier_codes in wl["tiers"].values():
    protected.update(tier_codes)
print(f"selected_diseases protected phecodes: {len(protected)}")

pm = F.phecode_map()

# --------------------------------------------------------------------------- #
# Per-disease compute
# --------------------------------------------------------------------------- #
def topk_metrics(y, s, prevalence):
    """PPV/recall/F1 at the top-5% highest-risk budget (rank-based, tie-robust)."""
    n = len(s)
    k = max(1, int(round(TOP_FRAC * n)))
    order = np.argsort(-s, kind="mergesort")   # stable; highest risk first
    flagged = order[:k]
    tp = int(y[flagged].sum())
    ppv = tp / k
    npos = int(y.sum())
    recall = tp / npos if npos else np.nan
    f1 = (2 * ppv * recall / (ppv + recall)) if (ppv + recall) > 0 else 0.0
    return ppv, recall, f1, (ppv / prevalence if prevalence > 0 else np.nan), k

rows = []
for j, ph in enumerate(phecodes):
    m = M[:, j]
    y = E[m, j].astype(int)
    s = H[m, j].astype(float)
    t = T[m, j].astype(float)
    n_eval = int(m.sum())
    n_event = int(y.sum())
    prevalence = float(y.mean()) if n_eval else np.nan

    r = dict(phecode=ph, n_eval=n_eval, n_event=n_event, prevalence=prevalence)

    # names / category (paper-consistent mapping, with mortality override)
    if ph == "time_to_death":
        r["name"], r["category"] = "All-cause mortality", "Mortality"
    elif ph in pm.index:
        r["name"], r["category"] = pm.loc[ph, "PhecodeString"], pm.loc[ph, "PhecodeCategory"]
    else:
        r["name"], r["category"] = ph, "Other"

    # incident discrimination + operating-point metrics
    if n_event >= 1 and n_eval > 0:
        r["auprc"] = float(average_precision_score(y, s))
        r["auprc_fold"] = r["auprc"] / prevalence if prevalence > 0 else np.nan
        r["auroc6y"] = float(roc_auc_score(y, s)) if 0 < n_event < n_eval else np.nan
        ppv, rec, f1, ppv_fold, k = topk_metrics(y, s, prevalence)
        r.update(ppv5=ppv, recall5=rec, f1_5=f1, ppv5_fold=ppv_fold, top5_k=k)
        ev = (E[:, j] == 1) & m
        r["lead_yr"] = float(T[ev, j].mean()) if ev.any() else np.nan
    else:
        for kk in ("auprc", "auprc_fold", "auroc6y", "ppv5", "recall5", "f1_5", "ppv5_fold", "lead_yr"):
            r[kk] = np.nan

    # canonical incident C (from the reviewed harness JSON)
    r["c_model"] = pdc[ph]["c"] if ph in pdc else np.nan
    # guardrail: matrix n_event must match the JSON's
    if ph in pdc:
        assert pdc[ph]["n_event"] == n_event, (ph, pdc[ph]["n_event"], n_event)

    # demographics (age+sex+BMI) C + guardrail that c_embedding == canonical C
    if ph in clin.index:
        r["c_demo"] = float(clin.loc[ph, "c_clinical"])
        r["_c_emb_3arm"] = float(clin.loc[ph, "c_embedding"])
    else:
        r["c_demo"] = np.nan
        r["_c_emb_3arm"] = np.nan
    r["dc"] = (r["c_model"] - r["c_demo"]) if pd.notna(r["c_model"]) and pd.notna(r["c_demo"]) else np.nan

    # prevalent screening (same-distribution CV; wrist logistic_emb + demo)
    if ph in scr_wrist.index:
        r["prev_n"] = int(scr_wrist.loc[ph, "n_test_pos"])
        r["prev_auroc"] = float(scr_wrist.loc[ph, "auroc"])
        r["prev_auprc"] = float(scr_wrist.loc[ph, "auprc"])
        r["prev_prevalence"] = float(scr_wrist.loc[ph, "prevalence_pop"])
        r["prev_auprc_fold"] = (r["prev_auprc"] / r["prev_prevalence"]) if r["prev_prevalence"] > 0 else np.nan
        r["prev_auroc_demo"] = float(scr_demo.loc[ph, "auroc"]) if ph in scr_demo.index else np.nan
    else:
        for kk in ("prev_n", "prev_auroc", "prev_auprc", "prev_prevalence", "prev_auprc_fold", "prev_auroc_demo"):
            r[kk] = np.nan

    r["in_selected_diseases"] = ph in protected
    r["is_mortality"] = (ph == "time_to_death")
    r["keep"] = (n_event >= 10)   # hard cutoff (selected-disease list does not override)
    rows.append(r)

df = pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# Guardrails + spot-checks
# --------------------------------------------------------------------------- #
shared = df[df._c_emb_3arm.notna() & df.c_model.notna()]
demb = (shared.c_model - shared._c_emb_3arm).abs()
print(f"GUARD c_model == 3arm c_embedding: max|Δ|={demb.max():.6f} over {len(shared)} (expect ~0)")
assert demb.max() < 1e-4

def show(ph):
    r = df[df.phecode == ph].iloc[0]
    print(f"  {ph:>14} {r['name'][:26]:<26} n_ev={int(r.n_event):>4} C={r.c_model:.3f} "
          f"Cdemo={r.c_demo:.3f} dC={r.dc:+.3f} AUPRC={r.auprc:.3f}({r.auprc_fold:.0f}x) "
          f"F1@5={r.f1_5:.3f} PPV@5={r.ppv5:.3f}({r.ppv5_fold:.0f}x) lead={r.lead_yr:.2f}y "
          f"prevAUROC={r.prev_auroc:.3f} n_prev={r.prev_n if pd.notna(r.prev_n) else '--'}")

print("SPOT CHECKS:")
for ph in ["NS_324.11", "NS_328.11", "NS_328.1", "time_to_death"]:
    show(ph)
pdrow = df[df.phecode == "NS_324.11"].iloc[0]
assert abs(pdrow.auprc - 0.4067) < 2e-3, pdrow.auprc          # vs investigations/pd_auprc/results.json
assert abs(pdrow.c_model - 0.909) < 2e-3, pdrow.c_model
assert abs(pdrow.prev_auroc - 0.9169) < 2e-3, pdrow.prev_auroc

kept = df[df.keep].copy().sort_values("c_model", ascending=False)
assert (kept.n_event >= 10).all(), "hard n_event>=10 cutoff violated"
print(f"\nFULL panel: {len(df)}  |  KEPT (hard n_event>=10): {len(kept)}")
print(f"  kept mean incident C = {kept.c_model.mean():.4f} ; mean dC over demographics = {kept.dc.mean():+.4f}")
print(f"  (info) selected-disease list diseases retained by the hard cut: {kept.in_selected_diseases.sum()}/43 "
      f"(min selected-disease list n_event = {df[df.in_selected_diseases].n_event.min()})")
n_prev_avail = kept.prev_auroc.notna().sum()
print(f"  kept rows with prevalent screening cell: {n_prev_avail}/{len(kept)}")

# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
cols = ["phecode", "name", "category", "is_mortality", "in_selected_diseases", "keep",
        "n_eval", "n_event", "prevalence", "lead_yr",
        "c_model", "c_demo", "dc",
        "auprc", "auprc_fold", "auroc6y", "ppv5", "recall5", "f1_5", "ppv5_fold", "top5_k",
        "prev_n", "prev_auroc", "prev_auroc_demo", "prev_auprc", "prev_prevalence", "prev_auprc_fold"]
df[cols].to_csv(OUTDIR / "rich_per_disease_full.csv", index=False)
kept[cols].to_csv(OUTDIR / "rich_per_disease.csv", index=False)
print(f"\nwrote {OUTDIR/'rich_per_disease.csv'} ({len(kept)} rows)")
print(f"wrote {OUTDIR/'rich_per_disease_full.csv'} ({len(df)} rows)")
print("DONE")
