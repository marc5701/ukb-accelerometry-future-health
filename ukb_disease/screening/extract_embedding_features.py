"""Extract frozen sequence-model (Embedding) features (penultimate representation plus
hazard) per subject.

The "stronger embedding" test for the prevalent-disease screener. Runs the frozen
deep Embedding transformer forward over all subjects (train+val+test) and writes, per
subject (mean over its recordings, matching pool_embeddings):
  - rep_seed{42,123,456,789} : 256-d penultimate CLS-pooled vector. Purely
    wrist-derived: in TransformerSeqHead.forward covariates fuse only at the
    hazard head, downstream of `pooled`, so the pooled representation carries no
    age/sex. The honest wrist-only feature.
  - hazard_mean   : 390-d Cox log-hazard averaged across the 4 seeds (calibrated
    logit space, then the standard 4-seed ensemble average).
  - pd_hazard_mean: hazard_mean[:, PD_IDX].
  - bucket        : train/val/test split bucket via hash_split, for leakage control.

LEAKAGE WARNING: Embedding's hazard head was trained on PD labels cohort-wide, and most
of the prevalent PD cases are in the train bucket (is_event=1, time~0 directly
optimized). The hazard is therefore a leaky upper bound; the representation is the
honest (but still disease-supervised) feature. Downstream eval reports both the
full set and the held-out (val+test) positive subset.

One forward pass per (seed, split) via a CombinedWrapper that returns
concat([hazard(390), pooled(256)]); reuses window_dataset.forward_subject_aggregated
for windowing/aggregation. A --check-hazards smoke validates the extracted test
hazards against the cached test_hazards.npy.

Run (GPU):
  python3 -u -m ukb_disease.screening.extract_embedding_features \
      --out screening_embedding_v1/embedding_features.npz --device cuda
Smoke (CPU, tiny):
  python3 -u -m ukb_disease.screening.extract_embedding_features \
      --splits test --seeds 42 --check-hazards --max-subj 40 --out embedding_smoke.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import CANONICAL_DAY_LMOMENT_ROOT, resolve_oak_path
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.baseline.cox_models import build_cox_model
from ukb_disease.baseline.window_dataset import WindowDataset, forward_subject_aggregated
from ukb_disease.baseline._split_hash import hash_split

RUNS_ROOT = f"{UKB_ROOT}/runs_cox"
PD_CODE = "NS_324.11"


class CombinedWrapper(torch.nn.Module):
    """Map (window, valid_mask, covariates) -> concat([hazard(390), pooled(256)]).

    Replicates TransformerSeqHead.forward up to `pooled` (the Embedding config branch:
    across_day_pool='cls', cov_bottleneck_dim=0, stream_split=0), then computes the
    real hazard head AND returns the pooled rep, so one forward yields both. Asserts
    the config so it fails loudly if the checkpoint arch ever differs.
    """

    def __init__(self, m: torch.nn.Module):
        super().__init__()
        assert getattr(m, "across_day_pool", "cls") == "cls", "expected cls pooling"
        assert getattr(m, "cov_bottleneck_dim", 0) == 0, "expected no cov bottleneck"
        assert getattr(m, "stream_split", 0) == 0, "expected no stream split"
        self.m = m

    @torch.no_grad()
    def forward(self, window, valid_mask, covariates):
        m = self.m
        B, N, _ = window.shape
        x = m.input_proj(m.input_norm(window))            # (B,N,d); no covariates
        cls = m.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, x], dim=1)                  # (B,1+N,d)
        pos_idx = torch.arange(1 + N, device=window.device)
        pos = m.pos_emb(pos_idx) if m.pe_type == "learned" else m.pos_emb_table[pos_idx]
        seq = seq + pos.unsqueeze(0)
        cls_real = torch.ones(B, 1, dtype=torch.bool, device=valid_mask.device)
        full_mask = torch.cat([cls_real, valid_mask], dim=1)
        out = m.encoder(seq, src_key_padding_mask=~full_mask)
        out = m.norm(out)
        pooled = out[:, 0, :]                             # (B,256) CLS; wrist-only
        cov_in = covariates[..., :m._head_n_covar]
        hazard = m.head(torch.cat([pooled, cov_in], dim=-1))   # (B,390)
        return torch.cat([hazard, pooled], dim=-1)        # (B,646)


def build_model_from_ckpt(seed_dir: str, device: str):
    cfg = json.load(open(os.path.join(seed_dir, "config.json")))
    a = cfg.get("args", cfg)
    ckpt = torch.load(os.path.join(seed_dir, "best.pt"), map_location=device, weights_only=False)
    phecodes = list(ckpt["phecodes"])
    n_phe = len(phecodes)
    model = build_cox_model(
        arch=a.get("arch", "transformer"),
        window_size=int(a.get("window_size", 3)),
        n_phecodes=n_phe,
        n_covar=int(cfg.get("n_covar", a.get("n_covar", 2))),
        in_dim=1536,
        dropout=float(a.get("dropout", 0.1)),
        n_layers=int(a.get("n_layers", 2)),
        pe_type=a.get("pe_type", "learned"),
        n_out_per_phecode=1,
        input_norm=bool(a.get("input_norm", True)),
        max_days=8,
        across_day_pool=a.get("across_day_pool", "cls"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)  # strict -> build must be exact
    model.eval()
    return model, phecodes


def minimal_outcomes(split_root: str, split: str, n_phe: int, eid_filter=None):
    """Dummy survival arrays covering every eid in the split index (only `hazards`
    from the forward is used downstream; event_times etc. are never read)."""
    idx = pd.read_parquet(resolve_oak_path(f"{split_root}/{split}/index.parquet"))
    eids = np.unique(idx["eid"].to_numpy(np.int64))
    if eid_filter is not None:
        eids = eids[np.isin(eids, np.asarray(list(eid_filter), np.int64))]
    n = len(eids)
    return {"eids": eids,
            "event_times": np.zeros((n, n_phe), np.float32),
            "is_event": np.zeros((n, n_phe), np.float32),
            "eval_mask": np.ones((n, n_phe), bool)}


def extract_split(model, split, day_root, df_cov, n_phe, device, batch_size,
                  eid_filter=None, eager=True):
    outc = minimal_outcomes(day_root, split, n_phe, eid_filter=eid_filter)
    ds = WindowDataset(split=split, window_size=3, outcomes=outc, covariates=df_cov,
                       day_mean_root=day_root, eager_load=eager, eid_filter=eid_filter)
    wrap = CombinedWrapper(model).to(device)
    res = forward_subject_aggregated(wrap, ds, device=device, batch_size=batch_size,
                                     out_dim=n_phe + 256)
    out = res["hazards"]                    # (n_rec, 646) per-recording (mean over windows)
    eids = res["eids"]
    haz = out[:, :n_phe]
    rep = out[:, n_phe:]
    return eids, haz, rep


def subj_aggregate(eids, arr):
    """Mean over recordings -> one row per subject (matches pool_embeddings)."""
    df = pd.DataFrame(arr)
    df["eid"] = eids
    g = df.groupby("eid").mean()
    return g.index.to_numpy(np.int64), g.to_numpy(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--run-glob", default="embedding_disjoint_split_seed*")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789])
    ap.add_argument("--day-means-root", default=CANONICAL_DAY_LMOMENT_ROOT)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--check-hazards", action="store_true",
                    help="validate extracted test hazards vs cached test_hazards.npy")
    ap.add_argument("--max-subj", type=int, default=0, help="smoke: cap subjects per split")
    args = ap.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"[extract] device={device} seeds={args.seeds} splits={args.splits}", flush=True)
    t0 = time.time()

    df_cov = load_covariates()  # raw [age, sex] indexed by eid
    print(f"[extract] covariates {df_cov.shape} cols={list(df_cov.columns)}", flush=True)

    per_seed_rep = {}     # seed -> (subj_eids, rep256)
    per_seed_haz = {}     # seed -> (subj_eids, haz390)
    phecodes_ref = None
    for seed in args.seeds:
        sd = os.path.join(resolve_oak_path(args.runs_root), f"embedding_disjoint_split_seed{seed}")
        model, phecodes = build_model_from_ckpt(sd, device)
        phecodes_ref = phecodes
        pd_idx = phecodes.index(PD_CODE)
        ts = time.time()
        rec_eids, rec_haz, rec_rep = [], [], []
        for split in args.splits:
            efilt = None
            if args.max_subj:
                idx = pd.read_parquet(resolve_oak_path(f"{args.day_mean_root}/{split}/index.parquet"))
                efilt = np.unique(idx["eid"].to_numpy(np.int64))[: args.max_subj]
            e, h, r = extract_split(model, split, args.day_mean_root, df_cov,
                                    len(phecodes), device, args.batch_size, eid_filter=efilt,
                                    eager=(args.max_subj == 0))
            rec_eids.append(e); rec_haz.append(h); rec_rep.append(r)
            print(f"  [seed{seed}/{split}] recordings={len(e)} ({time.time()-ts:.0f}s)", flush=True)
            if args.check_hazards and split == "test":
                cached = np.load(os.path.join(sd, "test_hazards.npy"))
                ceids = np.load(os.path.join(sd, "test_eids.npy")).astype(np.int64)
                se, sh = subj_aggregate(e, h)
                cmap = pd.Series(range(len(ceids)), index=ceids)
                common = np.intersect1d(se, ceids)
                ai = pd.Series(range(len(se)), index=se).loc[common].to_numpy()
                ci = cmap.loc[common].to_numpy()
                diff = np.abs(sh[ai] - cached[ci])
                print(f"  [check] test hazard vs cached: n={len(common)} "
                      f"max|d|={diff.max():.4f} mean|d|={diff.mean():.5f} "
                      f"corr_pd={np.corrcoef(sh[ai][:,pd_idx], cached[ci][:,pd_idx])[0,1]:.4f}", flush=True)
        rec_eids = np.concatenate(rec_eids)
        rec_haz = np.concatenate(rec_haz)
        rec_rep = np.concatenate(rec_rep)
        se, sh = subj_aggregate(rec_eids, rec_haz)
        _, sr = subj_aggregate(rec_eids, rec_rep)
        per_seed_haz[seed] = (se, sh)
        per_seed_rep[seed] = (se, sr)
        print(f"[seed{seed}] subjects={len(se)} ({time.time()-ts:.0f}s)", flush=True)

    # Align all seeds to the intersection of subject eids
    common = None
    for seed in args.seeds:
        e = per_seed_haz[seed][0]
        common = e if common is None else np.intersect1d(common, e)
    common = np.sort(common)
    print(f"[extract] common subjects across seeds: {len(common)}", flush=True)

    def reorder(eids, arr):
        loc = pd.Series(range(len(eids)), index=eids)
        return arr[loc.loc[common].to_numpy()]

    haz_stack = np.stack([reorder(*per_seed_haz[s]) for s in args.seeds])  # (S,n,390)
    hazard_mean = haz_stack.mean(0).astype(np.float32)
    pd_idx = phecodes_ref.index(PD_CODE)
    save = {"eids": common, "hazard_mean": hazard_mean,
            "pd_hazard_mean": hazard_mean[:, pd_idx].astype(np.float32),
            "phecodes": np.array(phecodes_ref), "pd_idx": pd_idx,
            "bucket": np.array([hash_split(int(e)) for e in common], dtype="U8"),
            "seeds": np.array(args.seeds)}
    for s in args.seeds:
        save[f"rep_seed{s}"] = reorder(*per_seed_rep[s]).astype(np.float32)

    out = resolve_oak_path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **save)
    print(f"[extract] wrote {out}: subjects={len(common)} rep_dim=256x{len(args.seeds)} "
          f"hazard_dim={hazard_mean.shape[1]} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
