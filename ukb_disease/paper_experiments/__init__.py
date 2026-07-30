"""Paper experiments (calibration, prodromal timing, decision-curve utility, proportional-hazards checks, regional transportability) for the wrist-actigraphy disease-prediction analysis.

Reuses the shared evaluation harnesses (crossfit_sweep, residual_partial,
head_to_head, screening_metrics, neuro_probe, io_utils) and writes results under
paper_experiments/.

Split = disjoint_split_v1; per-subject scoring; 6-year incident horizon.
scikit-survival is unavailable in this environment, so IPCW (censoring-weighted)
metrics are hand-implemented in `survmetrics`.
"""
