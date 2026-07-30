"""Hand-rolled survival metrics with IPCW (scikit-survival is unavailable here).

Everything here is plain numpy (plus optional lifelines only inside
`calibration_slope_citl`), so it runs identically anywhere. All estimators follow
standard references:

  * Kaplan-Meier step survival, with left-limit evaluation (needed for IPCW).
  * Breslow baseline cumulative hazard with the linear predictor as a FIXED
    coefficient-1 offset -> absolute risk R(t) = 1 - exp(-H0(t) * exp(lp)).
  * Cross-fit Breslow risk (event-stratified folds) so the absolute-risk map is
    never fit on the same rows it is evaluated on (no in-sample optimism).
  * IPCW Brier (Graf 1999), integrated Brier, and IPA / scaled Brier.
  * Uno cumulative/dynamic IPCW time-dependent AUROC, and an IPCW time-dependent
    AUPRC (average precision) with the no-skill floor returned alongside.
  * Vickers/Elkin time-to-event net benefit (KM within the flagged subgroup).

Conventions
-----------
time, event are per-subject arrays; event is 1 if the event of interest occurred
at `time`, else 0 (censored). `score`/`lp` are higher = more risk. `risk` is an
absolute probability in [0, 1] at a stated horizon.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "StepFn", "km_survival", "km_censoring", "km_incidence",
    "breslow_cumhaz", "risk_at_horizon", "crossfit_breslow_risk",
    "ipcw_brier", "integrated_brier", "ipa_scaled_brier",
    "ipcw_cd_auc", "ipcw_td_ap", "net_benefit_tte",
    "calibration_slope_citl", "subject_bootstrap_ci",
]


# --------------------------------------------------------------------------- #
# Step functions (right-continuous, with left-limit lookup)
# --------------------------------------------------------------------------- #
class StepFn:
    """Right-continuous step function defined by jump times and post-jump values.

    `times` ascending, `vals[k]` = value on [times[k], times[k+1]).
    `before` = value strictly before times[0] (default 1.0 for survival curves).
    """

    def __init__(self, times: np.ndarray, vals: np.ndarray, before: float = 1.0):
        self.t = np.asarray(times, float)
        self.v = np.asarray(vals, float)
        self.before = float(before)

    def at(self, x) -> np.ndarray:
        """Right-continuous value at x: value at the largest jump time <= x."""
        x = np.asarray(x, float)
        idx = np.searchsorted(self.t, x, side="right") - 1
        out = np.where(idx < 0, self.before, self.v[np.clip(idx, 0, len(self.v) - 1)])
        return out if out.shape else float(out)

    def at_minus(self, x) -> np.ndarray:
        """Left-limit value just before x: value at the largest jump time < x."""
        x = np.asarray(x, float)
        idx = np.searchsorted(self.t, x, side="left") - 1
        out = np.where(idx < 0, self.before, self.v[np.clip(idx, 0, len(self.v) - 1)])
        return out if out.shape else float(out)


def _km(time: np.ndarray, indicator: np.ndarray) -> StepFn:
    """Kaplan-Meier survival of the process whose events are `indicator==1`."""
    time = np.asarray(time, float)
    ind = np.asarray(indicator).astype(bool)
    order = np.argsort(time, kind="stable")
    t, d = time[order], ind[order]
    n = len(t)
    at_risk = n
    S = 1.0
    jt, jv = [], []
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d_u = int(d[i:j].sum())
        if d_u > 0 and at_risk > 0:
            S *= (1.0 - d_u / at_risk)
            jt.append(t[i])
            jv.append(S)
        at_risk -= (j - i)
        i = j
    if not jt:
        return StepFn(np.array([np.inf]), np.array([1.0]), before=1.0)
    return StepFn(np.array(jt), np.array(jv), before=1.0)


def km_survival(time, event) -> StepFn:
    """KM survival of the EVENT-of-interest distribution (event==1 jumps)."""
    return _km(time, np.asarray(event).astype(bool))


def km_censoring(time, event) -> StepFn:
    """KM survival of the CENSORING distribution G(t) (censored == event==0)."""
    return _km(time, ~np.asarray(event).astype(bool))


def km_incidence(time, event, t_star: float, ci: bool = True):
    """Observed cumulative incidence 1 - S(t*) with a Greenwood log-log 95% CI.

    Returns (incidence, lo, hi). lo/hi are NaN when ci=False or degenerate.
    """
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(bool)
    if len(time) == 0 or not ev.any():
        return 0.0, np.nan, np.nan
    order = np.argsort(time, kind="stable")
    t, d = time[order], ev[order]
    n = len(t)
    at_risk = n
    S = 1.0
    cum_var = 0.0  # sum d/(n(n-d)) for Greenwood
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        if t[i] > t_star:
            break
        d_u = int(d[i:j].sum())
        if d_u > 0 and at_risk > 0:
            S *= (1.0 - d_u / at_risk)
            if at_risk - d_u > 0:
                cum_var += d_u / (at_risk * (at_risk - d_u))
        at_risk -= (j - i)
        i = j
    inc = 1.0 - S
    if not ci or S <= 0 or S >= 1:
        return float(inc), np.nan, np.nan
    # log-log (cloglog) CI on S, then map to incidence.
    logS = np.log(S)
    se_loglog = np.sqrt(cum_var) / abs(logS)
    z = 1.959963984540054
    lo_S = S ** np.exp(z * se_loglog)
    hi_S = S ** np.exp(-z * se_loglog)
    return float(inc), float(1.0 - hi_S), float(1.0 - lo_S)


# --------------------------------------------------------------------------- #
# Breslow baseline + absolute risk
# --------------------------------------------------------------------------- #
def breslow_cumhaz(lp: np.ndarray, time: np.ndarray, event: np.ndarray) -> StepFn:
    """Breslow baseline cumulative hazard H0(t) with `lp` as a fixed offset.

    H0(t) = sum_{event times u <= t} d_u / sum_{j : T_j >= u} exp(lp_j).
    Returned as a right-continuous step function (H0=0 before the first event).
    """
    lp = np.asarray(lp, float)
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(bool)
    w = np.exp(lp - lp.max())  # subtract max for numerical stability (cancels in ratio? no -> see note)
    # NOTE: subtracting a constant c from lp scales every exp(lp) by exp(-c),
    # which divides the risk-set sum by exp(-c) too -> H0 unchanged. So risk
    # 1-exp(-H0*exp(lp)) must use the SAME shifted lp. We therefore return the
    # shift so callers can apply it consistently.
    order = np.argsort(time, kind="stable")
    t, e, ws = time[order], ev[order], w[order]
    suff = np.cumsum(ws[::-1])[::-1]  # suff[i] = sum over time>=t[i]
    n = len(t)
    jt, jv, H = [], [], 0.0
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d_u = int(e[i:j].sum())
        if d_u > 0 and suff[i] > 0:
            H += d_u / suff[i]
            jt.append(t[i])
            jv.append(H)
        i = j
    if not jt:
        fn = StepFn(np.array([np.inf]), np.array([0.0]), before=0.0)
    else:
        fn = StepFn(np.array(jt), np.array(jv), before=0.0)
    fn.lp_shift = float(lp.max())  # type: ignore[attr-defined]
    return fn


def risk_at_horizon(lp: np.ndarray, h0: StepFn, t_star: float) -> np.ndarray:
    """Absolute risk R(t*) = 1 - exp(-H0(t*) * exp(lp - lp_shift))."""
    lp = np.asarray(lp, float)
    shift = getattr(h0, "lp_shift", 0.0)
    H = h0.at(t_star)
    return 1.0 - np.exp(-H * np.exp(lp - shift))


def crossfit_breslow_risk(lp: np.ndarray, time: np.ndarray, event: np.ndarray,
                          t_star: float, fold: np.ndarray) -> np.ndarray:
    """Honest absolute risk at t*: per fold, fit Breslow H0 on the OTHER folds
    and map the held-out fold. Avoids fitting the baseline on rows it scores.
    Folds with no training events fall back to the marginal KM incidence.
    """
    lp = np.asarray(lp, float)
    time = np.asarray(time, float)
    event = np.asarray(event).astype(bool)
    risk = np.full(len(lp), np.nan)
    for f in np.unique(fold):
        te = fold == f
        tr = ~te
        if int(event[tr].sum()) < 1:
            inc, _, _ = km_incidence(time[tr], event[tr], t_star, ci=False)
            risk[te] = inc
            continue
        h0 = breslow_cumhaz(lp[tr], time[tr], event[tr])
        risk[te] = risk_at_horizon(lp[te], h0, t_star)
    return np.clip(risk, 1e-12, 1 - 1e-12)


# --------------------------------------------------------------------------- #
# IPCW Brier / IBS / IPA
# --------------------------------------------------------------------------- #
def _ipcw_weights(time, event, t_star, G: StepFn, g_floor=1e-3):
    """Graf IPCW weights at horizon t*; returns (w, D, usable_mask).

    D = 1{event by t*}. Weights: event-by-t* -> 1/G(T-); event-free-past-t* ->
    1/G(t*); censored-before-t* -> 0 (dropped). G clipped at g_floor.
    """
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(bool)
    D = ((time <= t_star) & ev).astype(float)
    w = np.zeros(len(time))
    case = (time <= t_star) & ev
    surv = time > t_star
    w[case] = 1.0 / np.clip(G.at_minus(time[case]), g_floor, None)
    w[surv] = 1.0 / max(float(G.at(t_star)), g_floor)
    usable = case | surv  # censored-before-t* get weight 0
    return w, D, usable


def ipcw_brier(risk, time, event, t_star, G: StepFn = None, g_floor=1e-3) -> float:
    """IPCW Brier score at horizon t* (Graf 1999). Lower is better."""
    risk = np.asarray(risk, float)
    if G is None:
        G = km_censoring(time, event)
    w, D, _ = _ipcw_weights(time, event, t_star, G, g_floor)
    n = len(risk)
    return float(np.sum(w * (D - risk) ** 2) / n)


def integrated_brier(risk_fn, time, event, t_star, grid=None, G: StepFn = None,
                     g_floor=1e-3) -> float:
    """Integrated Brier 1/t* * int_0^t* BS(u) du (trapezoid).

    risk_fn(u) -> per-subject absolute risk at horizon u. `grid` defaults to 20
    points up to t* truncated where G stays >= g_floor.
    """
    if G is None:
        G = km_censoring(time, event)
    if grid is None:
        grid = np.linspace(t_star / 20.0, t_star, 20)
    grid = np.asarray([u for u in grid if G.at(u) >= g_floor], float)
    if len(grid) < 2:
        return float("nan")
    bs = np.array([ipcw_brier(risk_fn(u), time, event, u, G, g_floor) for u in grid])
    return float(np.trapz(bs, grid) / (grid[-1] - grid[0]))


def ipa_scaled_brier(risk, time, event, t_star, G: StepFn = None, g_floor=1e-3) -> float:
    """Index of prediction accuracy: 1 - BS_model/BS_null (null = marginal KM)."""
    if G is None:
        G = km_censoring(time, event)
    inc, _, _ = km_incidence(time, event, t_star, ci=False)
    bs_model = ipcw_brier(risk, time, event, t_star, G, g_floor)
    bs_null = ipcw_brier(np.full(len(np.asarray(risk)), inc), time, event, t_star, G, g_floor)
    if bs_null <= 0:
        return float("nan")
    return float(1.0 - bs_model / bs_null)


# --------------------------------------------------------------------------- #
# Time-dependent discrimination (IPCW)
# --------------------------------------------------------------------------- #
def ipcw_cd_auc(time, event, score, horizon, G: StepFn = None, g_floor=1e-3) -> dict:
    """Uno cumulative-case / dynamic-control IPCW AUROC at one horizon.

    cases  : time<=t & event,           weight 1/G(T-)
    controls: time>t,                    weight 1 (event-free at t fully observed)
    censored-before-t: excluded.
    Returns {auc, n_case, n_ctrl}.
    """
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(bool)
    score = np.asarray(score, float)
    if G is None:
        G = km_censoring(time, event)
    case = (time <= horizon) & ev
    ctrl = time > horizon
    nc, nk = int(case.sum()), int(ctrl.sum())
    if nc == 0 or nk == 0:
        return {"auc": float("nan"), "n_case": nc, "n_ctrl": nk}
    wc = 1.0 / np.clip(G.at_minus(time[case]), g_floor, None)
    sc, sk = score[case], score[ctrl]
    diff = sc[:, None] - sk[None, :]
    comp = (diff > 0).astype(float) + 0.5 * (diff == 0)
    num = (wc[:, None] * comp).sum()
    den = wc.sum() * nk
    return {"auc": float(num / den), "n_case": nc, "n_ctrl": nk}


def ipcw_td_ap(time, event, score, horizon, G: StepFn = None, g_floor=1e-3) -> dict:
    """IPCW time-dependent average precision (cumulative cases at horizon).

    cases  : time<=t & event,  weight 1/G(T-);  controls: time>t, weight 1.
    AP via the step rule sum (recall_k - recall_{k-1}) * precision_k over
    descending thresholds. Returns {ap, floor, n_case, n_ctrl} where `floor` is
    the IPCW-weighted prevalence (no-skill AP).
    """
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(bool)
    score = np.asarray(score, float)
    if G is None:
        G = km_censoring(time, event)
    case = (time <= horizon) & ev
    ctrl = time > horizon
    nc, nk = int(case.sum()), int(ctrl.sum())
    if nc == 0 or nk == 0:
        return {"ap": float("nan"), "floor": float("nan"), "n_case": nc, "n_ctrl": nk}
    wc = 1.0 / np.clip(G.at_minus(time[case]), g_floor, None)
    P = wc.sum()                       # weighted positives
    s_case, s_ctrl = score[case], score[ctrl]
    # thresholds = unique scores descending; sweep accumulating weighted TP and FP.
    order_c = np.argsort(-s_case)
    order_k = np.argsort(-s_ctrl)
    sc_sorted, wc_sorted = s_case[order_c], wc[order_c]
    sk_sorted = s_ctrl[order_k]
    thresholds = np.unique(np.concatenate([s_case, s_ctrl]))[::-1]
    ci = ki = 0
    cumTP = 0.0
    cumFP = 0.0
    prev_recall = 0.0
    ap = 0.0
    for thr in thresholds:
        while ci < nc and sc_sorted[ci] >= thr:
            cumTP += wc_sorted[ci]
            ci += 1
        while ki < nk and sk_sorted[ki] >= thr:
            cumFP += 1.0
            ki += 1
        recall = cumTP / P
        denom = cumTP + cumFP
        precision = cumTP / denom if denom > 0 else 1.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    floor = P / (P + nk)
    return {"ap": float(ap), "floor": float(floor), "n_case": nc, "n_ctrl": nk}


# --------------------------------------------------------------------------- #
# Calibration slope + CITL
# --------------------------------------------------------------------------- #
def calibration_slope_citl(lp, time, event, t_star, risk=None) -> dict:
    """External calibration slope (Cox time~lp coefficient) + O:E ratio.

    slope = coef of lp in Cox(time,event) ~ lp (slope 1 = ideal). Uses lifelines.
    O:E = KM observed incidence at t* / mean predicted risk (CITL proxy; only
    meaningful if `risk` absolute risks are provided). Returns NaNs on failure.
    """
    out = {"slope": float("nan"), "oe_ratio": float("nan"),
           "obs_incidence": float("nan"), "mean_pred": float("nan")}
    lp = np.asarray(lp, float)
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(float)
    if ev.sum() < 5 or np.std(lp) == 0:
        return out
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
        z = (lp - lp.mean()) / (lp.std() + 1e-12)
        df = pd.DataFrame({"lp": z, "T": time, "E": ev})
        cph = CoxPHFitter(penalizer=0.0).fit(df, "T", "E")
        # de-standardize: slope on raw lp = beta_z / sd(lp)
        out["slope"] = float(cph.params_["lp"] / (lp.std() + 1e-12))
    except Exception:
        try:
            import pandas as pd
            from lifelines import CoxPHFitter
            z = (lp - lp.mean()) / (lp.std() + 1e-12)
            df = pd.DataFrame({"lp": z, "T": time, "E": ev})
            cph = CoxPHFitter(penalizer=1e-3).fit(df, "T", "E")
            out["slope"] = float(cph.params_["lp"] / (lp.std() + 1e-12))
        except Exception:
            pass
    if risk is not None:
        inc, _, _ = km_incidence(time, ev, t_star, ci=False)
        mp = float(np.mean(risk))
        out["obs_incidence"] = float(inc)
        out["mean_pred"] = mp
        out["oe_ratio"] = float(inc / mp) if mp > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Decision-curve net benefit (time-to-event, Vickers/Elkin)
# --------------------------------------------------------------------------- #
def net_benefit_tte(risk, time, event, t_star, thresholds) -> dict:
    """Vickers/Elkin time-to-event net benefit at horizon t* (KM-based).

    For each threshold p: flagged = risk>=p; f = mean(flagged);
    R+ = KM incidence at t* within flagged; R_all = overall KM incidence.
    NB_model = f*R+ - f*(1-R+)*p/(1-p);  NB_all = R_all - (1-R_all)*p/(1-p).
    Returns dict of arrays {threshold, nb_model, nb_all, nb_none, frac_flagged}.
    """
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    ev = np.asarray(event).astype(bool)
    R_all, _, _ = km_incidence(time, ev, t_star, ci=False)
    th = np.asarray(thresholds, float)
    nb_model = np.full(len(th), np.nan)
    frac = np.full(len(th), np.nan)
    for i, p in enumerate(th):
        flag = risk >= p
        f = float(flag.mean())
        frac[i] = f
        if f <= 0:
            nb_model[i] = 0.0
            continue
        Rp, _, _ = km_incidence(time[flag], ev[flag], t_star, ci=False)
        odds = p / (1.0 - p) if p < 1 else np.inf
        nb_model[i] = f * Rp - f * (1.0 - Rp) * odds
    nb_all = R_all - (1.0 - R_all) * (th / (1.0 - th))
    nb_all = np.where(th < 1, nb_all, -np.inf)
    return {"threshold": th, "nb_model": nb_model, "nb_all": nb_all,
            "nb_none": np.zeros(len(th)), "frac_flagged": frac, "R_all": float(R_all)}


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def subject_bootstrap_ci(stat_fn, n_subjects, n_boot=1000, seed=42, alpha=0.05):
    """Percentile CI of `stat_fn(idx)` over subject-resampled index arrays.

    stat_fn receives a bootstrap index array (with replacement) and returns a
    float (or NaN to skip). Returns (point, lo, hi) where point = stat_fn(all).
    """
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(n_subjects))
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_subjects, size=n_subjects)
        try:
            vals[b] = stat_fn(idx)
        except Exception:
            vals[b] = np.nan
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float(point), float("nan"), float("nan")
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return float(point), lo, hi
