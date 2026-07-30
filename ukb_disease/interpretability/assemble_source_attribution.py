"""Assemble the SOURCE_ATTRIBUTION day/night by modality heatmap with decision_rule and cell_correlation.

Run once the Tier-2 (dn_strat), Tier-3 REMOVE_AND_RETRAIN (dn_remove_and_retrain_<cell>), and Tier-2 ablation
outputs exist. Uses saved per-subject hazards for the REMOVE_AND_RETRAIN dC (no extra forwards).
Produces:
  * day/night by {AR,HA} heatmap (per-disease and global) with decision_rule decision overlay
  * C-parity check (stratified model vs reference mean-C)
  * cell_correlation diagnostics (4x4 cell-correlation, marginal-minus-conditional leakage)
  * per-disease profiles and per-subject local explanations
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.interpretability.ladder import attribution_common as ic
from ukb_disease.interpretability.validate import analysis as A
from ukb_disease.interpretability.validate import decision_rule as BK
from ukb_disease.interpretability.validate import cell_correlation as EN
from ukb_disease.interpretability import build_figures as BF
from ukb_disease.interpretability import source_attribution_profiles as W3

DN_CELLS = ["AR_day", "HA_day", "AR_night", "HA_night"]
WATCH = ["NS_324.11", "NS_328.11", "NS_328.1", "NS_333.1", "EM_202.2", "EM_236.1",
         "CV_416.2", "RE_474", "MB_286.2", "time_to_death"]


def remove_and_retrain_dc(strat_prefix: str, remove_and_retrain_prefix: str, cells: list[str], seeds: list[int]):
    """dC_remove_and_retrain[cell, disease] = mean_seed( C(dn_strat) - C(dn_remove_and_retrain_cell) ), from saved hazards."""
    P = None
    per_cell = {}
    c_strat_seeds = []
    for s in seeds:
        sd = ic.load_saved_test(f"{ic.RUNS_ROOT}/{strat_prefix}_seed{s}")
        res = {"hazards": sd["hazards"], "event_times": sd["event_times"],
               "is_event": sd["is_event"], "eval_mask": sd["eval_mask"]}
        c_strat_seeds.append(ic.per_disease_c(res))
    c_strat = np.nanmean(c_strat_seeds, axis=0)
    P = c_strat.shape[0]
    for cell in cells:
        # per-seed dC (C_strat - C_remove_and_retrain_cell), then average over seeds
        per_seed = []
        for si, s in enumerate(seeds):
            rd = f"{ic.RUNS_ROOT}/{remove_and_retrain_prefix}_{cell}_seed{s}"
            try:
                sd = ic.load_saved_test(rd)
            except Exception:
                continue
            cr = ic.per_disease_c({"hazards": sd["hazards"], "event_times": sd["event_times"],
                                   "is_event": sd["is_event"], "eval_mask": sd["eval_mask"]})
            per_seed.append(c_strat_seeds[si] - cr)
        if len(per_seed) < len(seeds):
            print(f"[source_attribution] WARN REMOVE_AND_RETRAIN cell {cell}: only {len(per_seed)}/{len(seeds)} seeds found")
        per_cell[cell] = np.nanmean(per_seed, axis=0) if per_seed else np.full(P, np.nan)
    M = np.vstack([per_cell[c] for c in cells])
    return cells, M, c_strat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strat-prefix", default="dn_strat")
    ap.add_argument("--remove_and_retrain-prefix", default="dn_remove_and_retrain")
    ap.add_argument("--cache", default=f"{UKB_ROOT}/day_lmoment_day_night_disjoint_split")
    ap.add_argument("--seeds", type=int, nargs="+", default=ic.SEEDS)
    ap.add_argument("--out-root", default=f"{UKB_ROOT}/interpretability/source_attribution")
    args = ap.parse_args()

    # Tier-2 ablation tables (conditional/marginal/joint)
    npz = A.load_ablation_seeds(args.strat_prefix)
    tab_cond = A.dc_table(npz, mode="cond")
    tab_marg = A.dc_table(npz, mode="marg")
    ph = tab_cond["phecodes"]
    print(f"[source_attribution] Tier-2 cells={tab_cond['cell_names']} global cond dC="
          f"{ {c: round(float(np.nanmean(tab_cond['dc_mean'][i])),4) for i,c in enumerate(tab_cond['cell_names'])} }")

    # Tier-3 REMOVE_AND_RETRAIN dC
    cells_r, M_remove_and_retrain, c_strat = remove_and_retrain_dc(args.strat_prefix, args.remove_and_retrain_prefix, DN_CELLS, args.seeds)
    print(f"[source_attribution] C-parity: stratified mean-C={np.nanmean(c_strat):.4f} vs reference 0.6880")
    print(f"[source_attribution] REMOVE_AND_RETRAIN global dC={ {c: round(float(np.nanmean(M_remove_and_retrain[i])),4) for i,c in enumerate(cells_r)} }")

    # decision_rule: tier-2 conditional + REMOVE_AND_RETRAIN (+ marginal for leakage)
    # align cells
    ci = {c: i for i, c in enumerate(tab_cond["cell_names"])}
    cond_M = np.vstack([tab_cond["dc_mean"][ci[c]] for c in DN_CELLS])
    cond_q = np.vstack([tab_cond["q"][ci[c]] for c in DN_CELLS])
    marg_M = np.vstack([tab_marg["dc_mean"][ci[c]] for c in DN_CELLS])
    # REMOVE_AND_RETRAIN q: one-sided over seeds is thin, so use sign plus conditional q as the arbiter.
    # Significance comes from the well-powered conditional tier (4 seeds x 5 perms); REMOVE_AND_RETRAIN
    # provides the sign and magnitude agreement gate. This yields decisions identical to
    # borrowing cond_q for REMOVE_AND_RETRAIN, so the arbiter label stays consistent.
    bankres = BK.bank({"tier2_cond": cond_M, "remove_and_retrain": M_remove_and_retrain},
                      {"tier2_cond": cond_q})
    print(f"[source_attribution] decision_rule: confirmed={bankres['n_confirmed']} suggestive={bankres['n_suggestive']} "
          f"unfaithful={bankres['n_unfaithful']}")

    # cell_correlation: cell-correlation + leakage
    _, cmeans = EN.per_subject_cell_means(args.cache, "test",
                                          {c: ic.day_night_cells(2)[c] for c in DN_CELLS})
    names, R = EN.cell_correlation_matrix(cmeans)
    leak = marg_M - cond_M
    print(f"[source_attribution] cell-corr (day-AR vs night-AR)={R[0,2]:.3f} (AR-day vs HA-day)={R[0,1]:.3f}")
    print(f"[source_attribution] mean leakage (marg-cond) per cell="
          f"{ {c: round(float(np.nanmean(leak[i])),4) for i,c in enumerate(DN_CELLS)} }")

    # figures
    dn_tab = {"cell_names": DN_CELLS, "phecodes": ph, "dc_mean": cond_M,
              "dc_lo": np.vstack([tab_cond['dc_lo'][ci[c]] for c in DN_CELLS]),
              "dc_hi": np.vstack([tab_cond['dc_hi'][ci[c]] for c in DN_CELLS]),
              "q": cond_q, "significant": cond_q < 0.05}
    BF.fig_modality_heatmap(dn_tab, WATCH, "source_attribution_day_night_heatmap.png",
                            "SOURCE_ATTRIBUTION day/night x modality attribution (Tier-2 conditional)")
    nevt = {p: int(np.nansum(npz[0]['full_is_event'][:, k] > 0)) for k, p in enumerate(ph)}
    prof = W3.build_profiles(dn_tab, n_event=nevt, decision=bankres["decision"])
    out_dir = resolve_oak_path(args.out_root); os.makedirs(out_dir, exist_ok=True)
    prof.to_csv(os.path.join(out_dir, "day_night_profiles.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "source_attribution_summary.npz"),
                        cells=np.array(DN_CELLS), phecodes=np.array(ph),
                        dc_cond=cond_M, dc_marg=marg_M, dc_remove_and_retrain=M_remove_and_retrain, leakage=leak,
                        cell_corr=R, decision=bankres["decision"], c_strat=c_strat)
    print(f"[source_attribution] wrote {out_dir} + heatmap. C-parity, decision_rule, cell_correlation complete.")


if __name__ == "__main__":
    main()
