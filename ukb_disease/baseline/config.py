"""Path and hyperparameter constants for the disease-prediction models.

Paths are expressed relative to ``UKB_ROOT`` (see ``ukb_disease.paths``) and
routed through ``resolve_oak_path`` at the point of filesystem access.
"""

from __future__ import annotations

from ukb_disease.paths import UKB_ROOT, resolve_oak_path  # noqa: F401  (re-export for callers)

# Per-recording embedding cache (extraction output).
CACHE_ROOT_V2 = f"{UKB_ROOT}/embeddings_v2"
# Per-patch embeddings for recordings added to the split after the main cache was
# built; same format, scanned as a fallback so the sequence model covers the full
# canonical set / 390-outcome panel.
CACHE_ROOT_V2_EXTRAS = f"{UKB_ROOT}/embeddings_extras"

# Subject-level mean vectors and aggregated parquets.
MEANS_ROOT = f"{UKB_ROOT}/means_v2"

# Per-day mean vectors, aggregated padded tensors, and model run directories.
DAY_MEAN_ROOT = f"{UKB_ROOT}/day_mean_v2"
RUNS_COX_ROOT = f"{UKB_ROOT}/runs_cox"

# Canonical aggregated tensors for the default split: the neuro outcomes are kept
# in train/val/test (so Parkinson's/Alzheimer's/dementia are evaluable) and the
# any-positive prevalence floor is 0.30%, giving a 390-phecode panel. The mean
# model uses CANONICAL_DAY_MEAN_ROOT; the L-moment model passes
# CANONICAL_DAY_LMOMENT_ROOT explicitly (with --input-norm).
CANONICAL_DAY_MEAN_ROOT = f"{UKB_ROOT}/day_mean_disjoint_split"
CANONICAL_DAY_LMOMENT_ROOT = f"{UKB_ROOT}/day_lmoment_disjoint_split"

# Sleep/wake-stratified per-day means.
DAY_MEAN_STRAT_ROOT = f"{UKB_ROOT}/day_mean_strat_v3"

# Per-day distributional summary ([mean || std] per dim).
DAY_DISTRIB_ROOT = f"{UKB_ROOT}/day_distrib_v3"

# Additional per-day pooling caches. Each mode writes its own root so the L-moment
# cache (day_lmoment_v3) and the mean||std cache (day_distrib_v3) are not
# clobbered:
#   meanstd_cov : [mean || std || vech(within-day cov_r)]  (cov off-diagonal)
#   l3_physio   : [l1||l2||l3] + resp-weighted [mean||std]  (AR-only extra)
#   l3_circ     : [l1||l2||l3 || Lomb-Scargle circadian amplitudes]
DAY_DISTRIB_MEANSTDCOV_ROOT = f"{UKB_ROOT}/day_distrib_meanstdcov_v3"
DAY_DISTRIB_LMOMENT_PHYSIO_ROOT = f"{UKB_ROOT}/day_distrib_lmoment_physio_v3"
DAY_DISTRIB_LMOMENT_CIRC_ROOT = f"{UKB_ROOT}/day_distrib_lmoment_circ_v3"
# PCA basis (train-only fit) consumed by the meanstd_cov pooling.
PCA_BASIS_ROOT = f"{UKB_ROOT}/pca_basis_v3"
# Informative-missingness per-subject wear-pattern descriptor CSVs.
MISSINGNESS_COV_ROOT = f"{UKB_ROOT}/missingness_covariates_v3"
# Circadian rest-activity-rhythm per-subject covariate CSVs (ENMO + p_wake
# families): IS / IV / RA / M10 / L5 / cosinor amplitude+acrophase across the full
# multi-day recording. Consumed by load_covariates(circadian_csv=...).
CIRCADIAN_COV_ROOT = f"{UKB_ROOT}/circadian_covariates_v3"

# Per-day pooling caches:
#   per-day soft-prediction embeddings (sleep/resp/activity probabilities),
#   per-day hard-prediction architecture stats (bouts/transitions/fractions),
#   and the two concatenated (composite), also appended onto the L-moment cache.
DAY_SOFTPRED_ROOT = f"{UKB_ROOT}/day_softpred_v1"
DAY_SOFTPRED_MEANSTD_ROOT = f"{UKB_ROOT}/day_softpred_meanstd_v1"
DAY_HARDFEAT_ROOT = f"{UKB_ROOT}/day_hardfeat_v1"
DAY_COMPOSITE_ROOT = f"{UKB_ROOT}/day_composite_v1"

# Patch-sequence head outputs.
PATCH_SEQUENCE_OUTPUT_ROOT = f"{UKB_ROOT}/patch_sequence_v1"

# Patch-sequence sweep cells. Tuple = (num_trans_layers, num_lstm_layers, lr, wd,
# grad_clip). Insertion order defines run order.
STAGE1_CELLS = {
    "S1.B":  (2, 0, 1e-4, 1e-3, 1.0),  # canonical baseline
    "S1.F":  (0, 0, 1e-4, 1e-3, 1.0),  # pool+linear control (no transformer)
    "S1.E":  (2, 1, 1e-4, 1e-3, 1.0),  # 2L + BiLSTM
    "S1.A":  (1, 0, 1e-4, 1e-3, 1.0),  # capacity-down
    "S1.D":  (2, 0, 5e-4, 1e-3, 1.0),  # higher LR
    "S1.C'": (2, 0, 1e-4, 1e-3, 0.5),  # strict grad clip
}

# Patch-sequence constants across all cells.
STAGE1_EMBED_DIM = 256
STAGE1_NUM_ATTN_HEADS = 4
STAGE1_NUM_POOL_HEADS = 6
STAGE1_MLP_RATIO = 2.0
STAGE1_MAX_SEQ_LEN = 2880   # AR patches per day
STAGE1_ENCODER_DROPOUT = 0.0
STAGE1_HEAD_DROPOUT = 0.4
STAGE1_BATCH_SIZE = 32      # single-GPU; DDP-4 uses 8 per GPU = 32 effective
STAGE1_EPOCHS = 50
STAGE1_PATIENCE = 10
STAGE1_WARMUP_FRAC = 0.10
STAGE1_SEED = 42

# Labels and covariates.
MORTALITY_PHECODES_CSV = f"{UKB_ROOT}/metadata/mortality_and_phecodes.csv"
ACC_DEM_CSV = f"{UKB_ROOT}/metadata/acc_dem.csv"
PHECODE_MAP_CSV = f"{UKB_ROOT}/metadata/Phecode_mapX_filtered.csv"

# Per-model patch lengths (must match extraction).
AR_PATCH_SECONDS = 30
HA_PATCH_SECONDS = 10

# Embedding dim (both encoders are 256-d; concat = 512).
AR_EMB_DIM = 256
HA_EMB_DIM = 256
CONCAT_EMB_DIM = AR_EMB_DIM + HA_EMB_DIM

# Prevalence filter. Default = 0.30% any-positive (incident + prevalent); on the
# participant-disjoint train this yields the 390-phecode panel and makes PD/AD/dementia
# (0.36-0.58% any-positive) evaluable. Override per run via --min-prevalence /
# --incident-only.
MIN_PREVALENCE = 0.003
SEVEN_DAY_YEARS = 7 / 365.25    # 7-day exclusion rule
HORIZON_YEARS = 6               # headline AUROC horizon
N_BOOTSTRAP = 1000              # 95% CI resamples
PHECODES_FIVE_HEADLINE = [
    # Resolved from the CSV column names at run time; see per_disease_eval.py.
]

__all__ = [
    "AR_EMB_DIM",
    "AR_PATCH_SECONDS",
    "ACC_DEM_CSV",
    "CACHE_ROOT_V2",
    "CACHE_ROOT_V2_EXTRAS",
    "CONCAT_EMB_DIM",
    "DAY_DISTRIB_ROOT",
    "DAY_DISTRIB_MEANSTDCOV_ROOT",
    "DAY_DISTRIB_LMOMENT_PHYSIO_ROOT",
    "DAY_DISTRIB_LMOMENT_CIRC_ROOT",
    "CANONICAL_DAY_MEAN_ROOT",
    "CANONICAL_DAY_LMOMENT_ROOT",
    "PCA_BASIS_ROOT",
    "MISSINGNESS_COV_ROOT",
    "CIRCADIAN_COV_ROOT",
    "DAY_MEAN_ROOT",
    "DAY_MEAN_STRAT_ROOT",
    "DAY_SOFTPRED_ROOT",
    "DAY_SOFTPRED_MEANSTD_ROOT",
    "DAY_HARDFEAT_ROOT",
    "DAY_COMPOSITE_ROOT",
    "HA_EMB_DIM",
    "HA_PATCH_SECONDS",
    "HORIZON_YEARS",
    "MEANS_ROOT",
    "MIN_PREVALENCE",
    "MORTALITY_PHECODES_CSV",
    "PATCH_SEQUENCE_OUTPUT_ROOT",
    "PHECODE_MAP_CSV",
    "N_BOOTSTRAP",
    "PHECODES_FIVE_HEADLINE",
    "RUNS_COX_ROOT",
    "SEVEN_DAY_YEARS",
    "STAGE1_BATCH_SIZE",
    "STAGE1_CELLS",
    "STAGE1_EMBED_DIM",
    "STAGE1_ENCODER_DROPOUT",
    "STAGE1_EPOCHS",
    "STAGE1_HEAD_DROPOUT",
    "STAGE1_MAX_SEQ_LEN",
    "STAGE1_MLP_RATIO",
    "STAGE1_NUM_ATTN_HEADS",
    "STAGE1_NUM_POOL_HEADS",
    "STAGE1_PATIENCE",
    "STAGE1_SEED",
    "STAGE1_WARMUP_FRAC",
    "resolve_oak_path",
]
