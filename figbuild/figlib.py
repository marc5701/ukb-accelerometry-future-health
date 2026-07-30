"""Shared figure library for the UKB disease-prediction paper.
Single source of truth for style, palette, loaders, and save routine.
Vector PDF (figure of record) goes to draft/figures/ ; QA PNG to figbuild/qa/.
Read-only on all source data.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_tmp"))
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ukb_disease.paths import UKB_ROOT

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Style/output paths derive relative to this file; data paths derive from UKB_ROOT.
_HERE = Path(os.path.abspath(__file__)).parent        # .../figbuild
_PAPER_ROOT = _HERE.parent                            # paper root

def oak_path(rel):
    """Resolve a data path under the configured data root."""
    return Path(UKB_ROOT) / rel

UKB = Path(UKB_ROOT)
BENCH = UKB / "benchmark" / "output" / "disjoint_split_v1"
SELECTED_DISEASES_CSV = UKB / "benchmark" / "selected_diseases_frozen.csv"
CLIN_DIR = UKB / "investigations" / "clinical_baseline" / "output"
AGEMECH_DIR = UKB / "investigations" / "age_mechanism" / "output"
RESID_DIR = UKB / "investigations" / "residual_partial" / "output"
CROSSFIT = UKB / "benchmark" / "output" / "crossfit_sweep_nm"
AGE_CAMPAIGN = UKB / "reports" / "age_campaign"
BMI_DIR = UKB / "bmi_loop"
INTERP_SOURCE_ATTRIBUTION = UKB / "interp_results" / "source_attribution"
PHECODE_MAP_CSV = UKB / "metadata" / "Phecode_mapX_filtered.csv"
# Prevalent-disease screening figure bundle.
SCREENING_BUNDLE = UKB / "screening_prevalent_v1" / "fig_bundle"

DRAFT_FIGS = _PAPER_ROOT / "draft" / "figures"
QA = _PAPER_ROOT / "figbuild" / "qa"
DRAFT_FIGS.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Style: Okabe-Ito colorblind-safe palette, embedded fonts
# --------------------------------------------------------------------------- #
OK = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F",
}
C_WRIST = OK["blue"]; C_AGESEX = OK["orange"]; C_EMBEDDING = OK["green"]
C_CLIN = OK["vermillion"]; C_POS = OK["green"]; C_NEG = OK["vermillion"]
C_NEUTRAL = OK["grey"]; C_AR = OK["purple"]; C_HA = OK["blue"]; C_HL = OK["orange"]

# Single source of truth for figure style: the Nature Communications mplstyle
# (Arimo sans-serif, 5 to 7 pt text, Type-42 embedded fonts, Okabe-Ito cycle, white
# background, >=1 pt lines). Applied once on import so every build script inherits one
# consistent style. figlib keeps only data: the palette dicts, the loaders, and save().
# To change style, edit figures/nature.mplstyle, not here.
NATURE_STYLE = str(_PAPER_ROOT / "figures" / "nature.mplstyle")
plt.style.use(NATURE_STYLE)

# Nature Communications column widths (inches). NatComms has only single (88 mm) and
# double (180 mm); there is no 1.5-column. W15 is kept as a deprecated alias (== W2) so
# existing imports keep working; new figures should use W1 or W2.
W1 = 88 / 25.4        # single column, 88 mm
W2 = 180 / 25.4       # double column, 180 mm
W15 = W2              # deprecated: 1.5-col maps to double; downsize a sparse figure to W1

# Single colourblind-safe disease-category palette: Paul Tol 'muted' for the main organ
# systems (the 9 mutually CB-distinct hues), plus distinct extras for rarer categories.
# The same category is the same colour in every figure, so figures share this dict rather
# than keeping their own palette. Grey for Other/Symptoms, black for Mortality. Alias keys
# (source-data spelling variants) map to the same colour.
CAT_COLORS = {
    "Cardiovascular": "#332288",          # indigo
    "Neurological": "#AA4499",            # purple
    "Endocrine/Metab": "#DDCC77",         # sand
    "Endocrine/metabolic": "#DDCC77",
    "Respiratory": "#88CCEE",             # cyan
    "Mental": "#44AA99",                   # teal
    "Mental disorders": "#44AA99",
    "Gastrointestinal": "#117733",        # green
    "Sense organs": "#CC6677",            # rose
    "Musculoskeletal": "#999933",         # olive
    "Muscloskeletal": "#999933",          # (data typo -> same colour)
    "Neoplasms": "#882255",               # wine
    "Dermatological": "#EE8866",          # light orange
    "Genitourinary": "#6699CC",           # mid blue
    "Blood/Immune": "#661100",            # dark brick
    "Infections": "#FFAABB",              # light pink
    "Symptoms": "#BBBBBB",                # grey
    "Other": "#DDDDDD",                   # pale grey
    "Mortality": "#000000",               # black
    "Pregnancy": "#AA4466",               # muted rose
    "Congenital": "#AAAA00",              # dark yellow
}

# Clean display names for category strings that arrive misspelled / variant in the source data.
CAT_DISPLAY = {
    "Muscloskeletal": "Musculoskeletal", "Muscoskeletal": "Musculoskeletal",
    "Endocrine/metabolic": "Endocrine/Metab", "Mental disorders": "Mental",
}

def cat_color(c):
    return CAT_COLORS.get(c, "#BBBBBB")

def cat_label(c):
    """Clean display name for a disease category (fixes source-data spelling variants)."""
    return CAT_DISPLAY.get(c, c)


def panel(ax, letter):
    """Bold lowercase panel letter (Nature style): a single letter at the top-left of the
    axes, 8 pt bold. Use this for panel identification; everything descriptive goes in the
    LaTeX caption, never in the artwork."""
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=8, pad=2)


def _is_panel_letter(text):
    t = text.strip()
    return len(t) == 1 and t.isalpha() and t.islower()


def _nature_lint(fig):
    """Block Nature figure-rule violations at save time: no figure suptitle, and no axes
    title other than a bare lowercase panel letter (a, b, c, ...). Titles, descriptions and
    takeaways belong in the LaTeX \\caption{}, not in the artwork. Raises ValueError listing
    every offender. Pass save(..., strict=False) only for a deliberate, documented exception."""
    bad = []
    supt = getattr(fig, "_suptitle", None)
    if supt is not None and supt.get_text().strip():
        bad.append(f"figure suptitle present: {supt.get_text().strip()!r}")
    for i, ax in enumerate(fig.get_axes()):
        for loc in ("center", "left", "right"):
            t = ax.get_title(loc=loc)
            if t.strip() and not _is_panel_letter(t):
                bad.append(f"axes[{i}] {loc}-title is not a bare panel letter: {t.strip()!r}")
    if bad:
        raise ValueError(
            "Nature figure rules violated (titles/descriptions go in the LaTeX caption, "
            "not in the artwork):\n  " + "\n  ".join(bad)
            + "\nFix: delete the title text; use figlib.panel(ax, 'a') for panel letters. "
              "Use save(..., strict=False) only for a deliberate, documented exception.")


def save(fig, name, pdf=True, strict=True):
    """Save vector PDF (figure of record) + QA PNG. `name` without extension.

    Figures are sized to an EXACT Nature column width (W1 = 88 mm single, W2 = 180 mm double),
    so we do NOT pass bbox_inches='tight' (it re-crops and would change the physical width).
    Content must therefore fit inside the fixed canvas via tight_layout / constrained_layout /
    manual gridspec margins, with legends placed inside the axes. PDF dpi 600 sets the
    resolution of any rasterized layer; vector lines/text stay exact.

    A Nature-rule lint runs first (no suptitle; axes titles must be bare panel letters);
    pass strict=False to bypass it only for a deliberate, documented exception."""
    if strict:
        _nature_lint(fig)
    fig.savefig(str(QA / f"{name}.png"), dpi=300, facecolor="white")
    if pdf:
        fig.savefig(str(DRAFT_FIGS / f"{name}.pdf"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"  saved {name}  ->  draft/figures/{name}.pdf  +  qa/{name}.png")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
_PMAP = None
def phecode_map():
    global _PMAP
    if _PMAP is None:
        m = pd.read_csv(PHECODE_MAP_CSV)
        _PMAP = m.groupby("Phecode").first()[["PhecodeString", "PhecodeCategory"]]
    return _PMAP


def load_per_disease(model="embedding_disjoint_split"):
    """Per-disease C for all 388 evaluable outcomes, + label + category."""
    d = json.load(open(BENCH / "per_disease_c.json"))[model]["per_disease"]
    rows = [dict(phecode=k, c=v["c"], n_event=v["n_event"], n_eval=v["n_eval"],
                 is_mortality=v.get("is_mortality", False)) for k, v in d.items()]
    df = pd.DataFrame(rows)
    pm = phecode_map()
    df = df.merge(pm, left_on="phecode", right_index=True, how="left")
    df.loc[df.phecode == "time_to_death", ["PhecodeString", "PhecodeCategory"]] = ["All-cause mortality", "Mortality"]
    df["PhecodeCategory"] = df["PhecodeCategory"].fillna("Other")
    df["PhecodeString"] = df["PhecodeString"].fillna(df["phecode"])
    return df


def load_seq_vs_mean():
    """Per-disease C (the sequence-vs-mean panel of the saturation figure, Extended Data Fig 4):
    age-pretrained sequence model vs a mean of the same patches (and the embedding ceiling), on the
    aligned 388-outcome disjoint_split_v1 test set. Built by ukb_disease.agepretrain.fig6_per_disease."""
    df = pd.read_csv(BENCH / "fig6_seq_vs_mean.csv")
    df["phecode"] = df["phecode"].astype(str)
    pm = phecode_map()
    df = df.merge(pm, left_on="phecode", right_index=True, how="left")
    df.loc[df.phecode == "time_to_death", ["PhecodeString", "PhecodeCategory"]] = ["All-cause mortality", "Mortality"]
    df["PhecodeCategory"] = df["PhecodeCategory"].fillna("Other")
    df["PhecodeString"] = df["PhecodeString"].fillna(df["phecode"])
    return df


def load_demographics():
    """Predicted-vs-true demographic recovery (sex ROC, age + BMI scatters), disjoint_split
    test set. Built by ukb_disease.baseline.fig_demographics."""
    return np.load(BENCH / "demographics_recovery.npz")


def load_embedding_decode():
    """Age/sex/BMI decoded from the frozen 512-d disease-prediction embedding (linear readout,
    val-fit -> test): sex ROC, age + BMI predicted-vs-true. Distinct from load_demographics()
    (dedicated wrist estimators). Built by
    ukb_disease.investigations.age_mechanism.decode_demographics_from_embedding."""
    return np.load(BENCH / "embedding_decode.npz")


def benchmark():
    return json.load(open(BENCH / "concordance_benchmark.json"))


def per_disease_full():
    return json.load(open(BENCH / "per_disease_c.json"))


def load_selected_diseases():
    return pd.read_csv(SELECTED_DISEASES_CSV)


def load_clinical():
    df = pd.read_csv(CLIN_DIR / "per_disease_3arm.csv")
    summ = json.load(open(CLIN_DIR / "summary.json"))
    return df, summ


SUMMARY_DIR = UKB / "investigations" / "summary_baseline" / "output"


def load_summary_activity(feature_set="full"):
    """Per-disease 6-yr AUROC + C for the UKB summarized-activity-features baseline vs the
    wrist embedding (arms: summary / emb / emb_summary / clin_summary / clin), plus the
    subject-level paired-bootstrap comparisons. Built by
    ukb_disease.baseline.summary_baseline. Mirrors load_clinical()."""
    d = SUMMARY_DIR / feature_set
    df = pd.read_csv(d / "per_disease_summary.csv")
    sj = d / "summary.json"
    summ = json.load(open(sj)) if sj.exists() else None
    return df, summ


def load_age_contrib():
    df = pd.read_csv(AGEMECH_DIR / "per_disease_age_contribution.csv")
    pr = json.load(open(AGEMECH_DIR / "probe_results.json"))
    return df, pr


def load_residual():
    e1 = pd.read_csv(RESID_DIR / "exp1_per_disease.csv")
    e2 = pd.read_csv(RESID_DIR / "exp2_ladder_per_disease.csv")
    return e1, e2


def load_crossfit():
    summ = json.load(open(CROSSFIT / "summary.json"))
    boot = json.load(open(CROSSFIT / "bootstrap.json"))
    per = pd.read_csv(CROSSFIT / "per_phecode.csv")
    return summ, boot, per


def load_age_campaign():
    return json.load(open(AGE_CAMPAIGN / "COMPILED_results.json"))


def load_modality_profiles():
    return pd.read_csv(INTERP_SOURCE_ATTRIBUTION / "modality_profiles.csv")


def load_day_night_profiles():
    return pd.read_csv(INTERP_SOURCE_ATTRIBUTION / "day_night_profiles.csv")


# canonical source order + display labels for the per-disease source-attribution figure
SOURCE_ORDER = ["day_activity", "day_sleepbreath", "night_activity", "night_sleepbreath",
                "genetics", "age", "sex", "bmi"]


def load_source_attribution():
    """Per-disease per-source contribution (marginal ΔC) for the source-attribution heatmap.
    Long form: phecode, source (one of SOURCE_ORDER), dC, significant, available, n_event.
    Built by ukb_disease/interpretability/source_attribution.py."""
    return pd.read_csv(UKB / "investigations" / "source_attribution" / "output"
                       / "perdisease_sources.csv")


def load_pd_auprc():
    return json.load(open(UKB / "investigations" / "pd_auprc" / "results.json"))


def load_embedding_ci():
    """Per-disease embedding concordance + 1,000-resample bootstrap CI for the highlighted
    diseases (point == per_disease_c.json; CI is embedding's own bootstrap). Keyed by phecode."""
    return json.load(open(UKB / "investigations" / "selected_diseases_ci" / "embedding_ci.json"))["diseases"]


def load_per_disease_ci():
    """All-388 per-disease concordance + subject-level bootstrap 95% CI (built by
    ukb_disease.benchmark.per_disease_ci). Keyed by phecode -> {C, CI_lo, CI_hi, ...}.
    Point C == per_disease_c.json; used for Figure 1 panel b highlight whiskers."""
    return json.load(open(BENCH / "per_disease_c_ci.json"))["diseases"]


def load_per_category_ci():
    """Per-category mean-C + bootstrap 95% CI (built by ukb_disease.benchmark.per_disease_ci).
    DataFrame indexed by category with columns mean_C, CI_lo, CI_hi, n_diseases. Figure 1 panel c."""
    return pd.read_csv(BENCH / "per_category_c_ci.csv").set_index("category")


def load_per_region_ci():
    """Per-region transportability point estimates + bootstrap 95% CI (built by
    ukb_disease.paper_experiments.regional_transport_ci): per_region_c.csv columns plus
    mean_C_lo, mean_C_hi for both methods. Figure 1 panel d."""
    return pd.read_csv(BENCH / "per_region_c_ci.csv")


def load_screening_bundle(path=None):
    """Prevalent-disease screening figure bundle. Returns a dict of result frames
    plus the per-disease out-of-fold predictions used to draw ROC curves:
      sd            632-disease same-distribution panel (wrist vs demographics AUROC)
      confound      per-disease confound-audit AUROCs (emb / confound / emb+conf / demo)
      wear          within-wear-stratum AUROC of the embedding
      nearonset     PD AUROC by years-since-diagnosis bin (+ nearonset_json: Spearman)
      embedding           frozen-deep Embedding representation vs L-moment pooling
      assoc         per-feature standardized mean differences of the confound block
      oof           {phecode: {"wrist": df(eid,y,pred), "demo": df(eid,y,pred)}}
      summary       SUMMARY.json scalars + self-verification flag (if present)
    """
    base = Path(path) if path else SCREENING_BUNDLE
    oof = {}
    odir = base / "oof"
    if odir.exists():
        for f in sorted(odir.glob("*.csv")):
            ph, _, arm = f.stem.rpartition("_")          # "<phecode>_<arm>"
            oof.setdefault(ph, {})[arm] = pd.read_csv(f)
    out = dict(
        sd=pd.read_csv(base / "per_disease_sd.csv"),
        confound=pd.read_csv(base / "confound_audit.csv"),
        wear=pd.read_csv(base / "wear_strata.csv"),
        nearonset=pd.read_csv(base / "nearonset.csv"),
        embedding=pd.read_csv(base / "embedding_compare.csv"),
        assoc=pd.read_csv(base / "confound_association.csv"),
        oof=oof,
        nearonset_json=json.load(open(base / "nearonset.json")),
    )
    fa = base / "forest_arms.csv"
    if fa.exists():
        out["forest"] = pd.read_csv(fa)        # phecode, n_pos, arm, auroc, lo, hi (3 arms/outcome)
    if (base / "SUMMARY.json").exists():
        out["summary"] = json.load(open(base / "SUMMARY.json"))
    return out


# --------------------------------------------------------------------------- #
# Canonical scalar constants (aggregate published results used across figures)
# --------------------------------------------------------------------------- #
CONST = dict(
    N_TEST=5253, N_TRAIN_SUBJ=87652, N_TRAIN_REC=96016, N_PANEL=390, N_EVAL=388,
    MEAN_C_EMBEDDING=0.6880, MEAN_C_SEQUENCE_MODEL=0.6865, MORT_C=0.7672, MEDIAN_C=0.6951,
    UNC_EMBEDDING=0.5224, UNC_SEQUENCE_MODEL=0.5226, UNC_AGE_FLOOR=0.5122,
    PAIRED_WINRATE=52.6, PAIRED_P=0.0143, PAIRED_FAVOR=(208, 160), AGE_GUARDRAIL=63.2,
    NODEMO_C=0.68356, WITHDEMO_C=0.68803, AGESEX_DELTA=0.00447,
    AGESEX_DELTA_CI=(0.00054, 0.00848), AGE_ONLY_C=0.5987,
    AGE_RANGE_MEAN=62.35, AGE_RANGE_SD=8.05,
    AGE_DECODE_R2=0.62, SEX_DECODE_AUROC=0.99, BMI_DECODE_R2=0.47,
    RAW_HAZARD=0.6886, JOINT_INSAMPLE=0.7146, JOINT_HONEST=0.6685,
    JOINT_NO_AR_EMB_HONEST=0.6711, OPTIMISM_GAP=0.039,
    CLIN_AUROC=0.6379, EMB_AUROC=0.6914, CLINICAL_EMBEDDING_COMBINER=0.6696, CLINICAL_EMBEDDING_DEEP=0.692,
    # fig:clinic panel-a three-arm mean concordance (disjoint_split_v1, per-subject, 388-outcome
    # panel). Wrist-only arm is NODEMO_C=0.68356 above.
    CLIN_ONLY_C=0.63855, WRIST_DEMOBMI_C=0.68858,
    # activity-summary-only baseline (ridge Cox on the 100 UKB activity-summary fields), same
    # 388-panel/split.
    SUMMARY_ONLY_C=0.57725,
    # 384 = disease phecodes with an estimable AUROC (all-cause mortality excluded; see fig:clinic b/d)
    EMB_BEATS_CLIN_FRAC=0.781, EMB_BEATS_CLIN_N=(300, 384), EMB_OVER_CLIN_DAUROC=0.0534,
    SURROGATE_DEEP=0.6871, SURROGATE_C=0.6448, SURROGATE_GAP=0.042, SURROGATE_RECOVER=93.8,
    HA_ABLATE=0.042, AR_ABLATE=0.0085, ABLATE_BOTH_FLOOR=0.622,
    AGE_TEST_R=0.8675, AGE_TEST_CI=(0.8605, 0.8748), AGE_DEPLOY_R=0.8523,
    AGE_DEPLOY_MAE=3.39, AGE_HA_MEAN=0.8014, AGE_OXFORD=0.823,
    GBM_ENMO=0.478, GBM_ALL18=0.533,
    BMI_TEST_R=0.7683, BMI_TEST_CI=(0.7552, 0.7809), BMI_VAL_R=0.7611,
    BMI_PARTIAL=0.764, BMI_DEMO=0.129, BMI_OXFORD=0.716,
    # Prevalent-disease screening (wrist-only, same-distribution CV).
    # Parkinson's disease and multiple sclerosis (the two motor outcomes).
    SCR_PD_AUROC=0.917, SCR_PD_CI=(0.875, 0.953), SCR_PD_SENS95=0.84, SCR_PD_DEMO=0.700,
    SCR_MS_AUROC=0.785, SCR_MS_DEMO=0.640,
    # Selectivity across the 632-disease panel.
    SCR_WINS=98, SCR_PANEL=632, SCR_WIN_FRAC=0.16,
    SCR_MEAN_WRIST=0.574, SCR_MEAN_DEMO=0.637,
    # Physiological confound audit (PD): confound-only at chance; embedding unmoved.
    SCR_EMB=0.916, SCR_CONF_ONLY=0.502, SCR_EMB_CONF=0.910, SCR_MAXSMD=0.158,
    SCR_WEAR_LOW=0.861, SCR_WEAR_HIGH=0.936,
    # Near-onset (PD): recently-diagnosed detection, no duration dependence.
    SCR_RECENT=0.911, SCR_SPEARMAN=0.009, SCR_SPEARMAN_P=0.92,
    # Frozen ceiling (PD held-out): deep Embedding rep does not beat L-moment pooling.
    SCR_EMBEDDING_LMOMENT=0.941, SCR_EMBEDDING_REP=0.930,
    # External comparison: Oxford wrist-plus-6-covariate prevalent-PD detector.
    SCR_OXFORD=0.98,
)

# Ceiling-map levers (Δ mean-C vs daily-mean baseline): each is the relative change
# from its own modelling experiment; all are <=0 except within-day distributional pooling.
CEILING_LEVERS = [
    ("Encoder fine-tuning", -0.0499, "frozen encoder already near-optimal"),
    ("Per-patch deep supervision", -0.0224, ""),
    ("Covariance pooling", -0.0069, ""),
    ("Flexible-length context", -0.0052, ""),
    ("State-space (Mamba) architecture", -0.0043, ""),
    ("Per-disease specialization", -0.0030, ""),
    ("Tremor-channel gate", -0.0028, ""),
    ("Circadian-rhythm features", -0.0004, ""),
    ("Sleep-stage soft predictions", -0.0002, ""),
    ("Hard sleep/wake calls", -0.0001, ""),
    ("Input masking", 0.0000, "helps one encoder, hurts the other"),
    ("Within-day variation (distributional pooling)", +0.0016, "the one lever that helps"),
]

# Negative-results compendium (Δ mean-C vs the relevant baseline)
NEG_RESULTS = [
    ("Soft-stage prediction features (standalone)", -0.0350),
    ("Hard architecture statistics", -0.0286),
    ("Per-patch deep-supervised Mamba", -0.0224),
    ("Full-recording context (RoPE)", -0.0071),
    ("Flexible-length day transformer", -0.0052),
    ("Day-level Mamba", -0.0043),
    ("Per-disease (GI) specialization", -0.0028),
    ("Time-of-day embeddings", +0.0003),
    ("Soft/hard preds concatenated to model", -0.0002),
    ("Parkinson's-enriched sampling (PD only)", +0.0196),
]

# --------------------------------------------------------------------------- #
# Stage-by-stage pipeline search (single source for the Extended Data pipeline schematic and the
# Extended Data ablation table). The model pipeline is:
#   raw signal -> frozen encoder -> daily aggregation -> temporal model -> Cox loss.
# Each ablation perturbs exactly one stage. Only the daily-aggregation stage helps.
#
# The Δ mean-C values below are on the development split, not disjoint_split_v1; the core
# ablations are rerun on disjoint_split_v1 before submission.
#
# Each approach is (name, what-it-changes, delta-string, verdict in
# {"win","tie","lose","base","mixed"}). "delta" is a display string because some
# entries are qualitative ("ties", "baseline").
# The age-pretrained sequence model is in the saturation figure (Extended Data Fig 4), not in this table.
PIPELINE_STAGES = [
    {
        "stage": "Frozen encoder", "kind": "core",
        "question": "Can a better representation be learned?",
        "approaches": [
            ("Encoder fine-tuning (LoRA, full)", "Adapt the frozen encoder weights", "-0.050", "lose"),
            ("Age-pretrain initialisation", "Warm-start from an age objective", "ties", "tie"),
            ("Input masking", "Mask non-wear before pooling", "0.000", "tie"),
        ],
    },
    {
        "stage": "Daily aggregation", "kind": "core",
        "question": "How should a day be summarised?",
        "approaches": [
            ("Within-day distributional pooling", "Keep within-day L-moments", "+0.0016", "win"),
            ("Per-day mean", "Average the embedding over the day", "baseline", "base"),
            ("Covariance pooling", "Add cross-dimension structure", "-0.007", "lose"),
        ],
    },
    {
        "stage": "Temporal model", "kind": "core",
        "question": "How should days be combined?",
        "approaches": [
            ("State-space model (Mamba)", "Replace attention with a state-space model", "-0.004", "lose"),
            ("Per-patch deep-supervised SSM", "State-space model over patches", "-0.022", "lose"),
            ("Flexible / long context", "Lengthen the temporal context window", "-0.007", "lose"),
            ("Time-of-day input embeddings", "Add clock time to the input", "+0.000", "tie"),
        ],
    },
    {
        "stage": "Loss / objective", "kind": "core",
        "question": "Is the survival loss the bottleneck?",
        "approaches": [
            ("Discrete-time logistic hazard", "Per-bin logistic instead of Cox", "-0.008", "lose"),
            ("Auxiliary phecode heads", "Add parent-category heads", "-0.001", "tie"),
            ("Large-batch Cox", "Enlarge the risk set per batch", "-0.007", "lose"),
        ],
    },
    {
        "stage": "Added physiology features", "kind": "augmentation",
        "question": "Does the raw signal carry signal the embedding misses?",
        "approaches": [
            ("Hand-engineered physiology", "HRV, tremor, ENMO, circadian features", "-0.048", "lose"),
            ("Nightbeat heart rate", "Resting HR and HRV from the ballistocardiogram", "-0.001", "tie"),
            ("Sleep-stage predictions", "Concatenate the decoded sleep calls", "-0.000", "tie"),
            ("Tremor-channel gate (PD)", "Resting-tremor band power", "-0.003", "lose"),
        ],
    },
    {
        "stage": "Disease-specific training", "kind": "augmentation",
        "question": "Does tuning to one disease help the panel?",
        "approaches": [
            ("Per-disease specialisation",
             "Resample or reweight training towards one target disease", "-0.003", "lose"),
            ("Parkinson's-enriched sampling",
             "Oversample Parkinson's cases; helps that disease alone, not the panel",
             "+0.020 (PD only)", "mixed"),
        ],
    },
]

def write_ablation_table(path=None):
    """Emit the stage-grouped Extended Data ablation tabular from PIPELINE_STAGES,
    so the table and the Extended Data pipeline schematic never diverge. Writes the tabular only;
    the floating table + caption live in extended_data.tex."""
    import re
    from pathlib import Path
    if path is None:
        path = Path(__file__).resolve().parent.parent / "draft" / "tab_ablations.tex"

    def fmt_delta(s):
        m = re.match(r"^([+-]\d*\.?\d+)(.*)$", s.strip())
        return ("$%s$%s" % (m.group(1), m.group(2))) if m else s.strip()

    verdict_word = {"win": "\\textbf{helps}", "lose": "loses", "tie": "ties",
                    "base": "baseline", "mixed": "mixed"}
    out = [
        "% Auto-generated by figlib.write_ablation_table from PIPELINE_STAGES. Do not hand-edit.",
        "% Deltas are on the development split; the core ablations are rerun on",
        "% disjoint_split_v1 before submission.",
        "\\begin{tabular}{@{}p{0.30\\linewidth}p{0.37\\linewidth}r l@{}}",
        "\\toprule",
        "Approach & What it changes & $\\Delta$ mean C & Outcome \\\\",
        "\\midrule",
    ]
    for i, st in enumerate(PIPELINE_STAGES):
        if i:
            out.append("\\addlinespace[2pt]")
        out.append("\\multicolumn{4}{@{}l}{\\textbf{%s}: %s} \\\\[1pt]" % (st["stage"], st["question"]))
        for (name, mech, delta, verdict) in st["approaches"]:
            out.append("%s & %s & %s & %s \\\\" % (name, mech, fmt_delta(delta),
                                                   verdict_word.get(verdict, verdict)))
    out += ["\\bottomrule", "\\end{tabular}", ""]
    Path(path).write_text("\n".join(out))
    return str(path)


if __name__ == "__main__":
    print("figlib self-test")
    print("  wrote ablation table:", write_ablation_table())
    pd_df = load_per_disease(); print("  per_disease:", pd_df.shape, "mean c (disease only)=",
          round(pd_df[~pd_df.is_mortality].c.mean(), 4))
    print("  categories:", sorted(pd_df.PhecodeCategory.unique()))
    wl = load_selected_diseases(); print("  selected_diseases:", wl.shape)
    cl, cs = load_clinical(); print("  clinical 3arm:", cl.shape, "deep clin+emb auroc=",
          cs.get("clin_emb_deep_mean_auroc"))
    ac, pr = load_age_contrib(); print("  age_contrib:", ac.shape)
    e1, e2 = load_residual(); print("  residual:", e1.shape, e2.shape)
    cf_s, cf_b, cf_p = load_crossfit(); print("  crossfit arms:", list(cf_b["arms"].keys()) if isinstance(cf_b.get("arms"), dict) else "n/a")
    mp = load_modality_profiles(); print("  modality cells:", sorted(mp.cell.unique())[:12])
    dn = load_day_night_profiles(); print("  day_night cells:", sorted(dn.cell.unique()))
    print("  bench mean_c embedding:", benchmark()["models"]["embedding_disjoint_split"]["mean_c"])
    print("SELFTEST OK")
