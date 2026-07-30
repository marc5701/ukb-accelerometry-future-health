"""Tier-3 REMOVE_AND_RETRAIN: retrain the stratified model with one cell removed.

REMOVE_AND_RETRAIN (Remove-And-Retrain) is the OOD-robust faithfulness arbiter. Instead of
perturbing a frozen model's input (which can measure off-manifold brittleness),
we delete the cell's information from train+val+test and retrain from scratch, so
dC_remove_and_retrain = C_full_retrained - C_cellremoved_retrained is the cell's true learnable
value. Cheap here because the head is small (~2.1M params).

Non-invasive: monkeypatch WindowDataset to zero the cell's columns right after
load (a removed-and-constant feature the input projection learns to ignore), then
delegate to the canonical train_rolling_window_cox trainer unchanged. All train_rolling_window_cox flags pass
through verbatim after the two remove_and_retrain args are stripped.

Usage:
    python3 -m ukb_disease.interpretability.ladder.remove_and_retrain \
        --ablate-cell AR_night --n-bins 2 -- \
        --arch transformer --window-size 3 ... --day-means-root <day_night cache> \
        --run-name dn_remove_and_retrain_AR_night_seed42 ...
"""

from __future__ import annotations

import argparse
import sys

from ukb_disease.baseline import window_dataset as wd
from ukb_disease.interpretability.ladder import attribution_common as ic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--ablate-cell", required=True,
                    help="day/night cell name to remove (e.g. AR_night), or 'none' "
                         "for the full retrained baseline.")
    ap.add_argument("--n-bins", type=int, default=2)
    ap.add_argument("--help-remove_and_retrain", action="store_true")
    known, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]   # drop the argparse option/positional separator

    spec = None
    if known.ablate_cell != "none":
        cells = ic.day_night_cells(n_bins=known.n_bins)
        if known.ablate_cell not in cells:
            raise SystemExit(f"--ablate-cell {known.ablate_cell} not in {list(cells)}")
        spec = cells[known.ablate_cell]
        print(f"[retrain_tier-remove_and_retrain] removing cell {known.ablate_cell} -> spec {spec}")
    else:
        print("[retrain_tier-remove_and_retrain] ablate-cell=none -> full retrained baseline")

    _orig_init = wd.WindowDataset.__init__

    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        if spec is not None:
            tname, s, e = spec
            arr = self._ar if tname == "ar" else self._ha
            if arr is not None:
                arr[:, :, s:e] = 0.0  # remove the cell's information in EVERY split
    wd.WindowDataset.__init__ = _patched_init

    import ukb_disease.baseline.train_rolling_window_cox as rp
    sys.argv = ["train_rolling_window_cox"] + rest
    rp.main()


if __name__ == "__main__":
    main()
