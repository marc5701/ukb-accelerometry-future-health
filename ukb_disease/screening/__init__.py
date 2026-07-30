"""Cross-entropy screening side-result for UKB disease prediction.

A complement to the CoxPH incident-prediction headline: train per-disease
binary screeners (cross-entropy) to detect *current/prevalent* disease, using
the experimental `screening_prevalent_v1` split (train on incident, test on
prevalent). See the plan + `SCREENING.md` for the full design.
"""
