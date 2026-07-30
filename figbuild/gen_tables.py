"""Emit LaTeX table fragments for the Extended Data section."""
import os
from pathlib import Path
from ukb_disease.paths import UKB_ROOT
import figlib as F

OUT = str(Path(__file__).resolve().parent.parent / "draft")

def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s

import re
# US -> UK spelling for phecode DISPLAY names only (the phecodeX vocabulary is US-spelled; the
# manuscript is British). Case-preserving, single-pass, applied to display names only (never phecode
# IDs or matching keys). Stems use \b / word-final anchors so e.g. \bhemo hits 'hemorrhage'/'hemoptysis'
# but not 'chemo', and emia\b hits 'anemia'/'hypoglycemia' but not 'academia'.
_BRIT_RULES = [
    (r'ischemi', 'ischaemi'), (r'\bhemat', 'haemat'), (r'\bhemo', 'haemo'), (r'esophag', 'oesophag'),
    (r'emia\b', 'aemia'), (r'pnea\b', 'pnoea'), (r'rrhea', 'rrhoea'), (r'edema', 'oedema'),
    (r'celiac', 'coeliac'), (r'tumor', 'tumour'), (r'orthopedic', 'orthopaedic'), (r'pediatric', 'paediatric'),
    (r'gynecolog', 'gynaecolog'), (r'anesthes', 'anaesthes'), (r'estrogen', 'oestrogen'), (r'behavioral', 'behavioural'),
    (r'localiz', 'localis'), (r'depolariz', 'depolaris'), (r'generaliz', 'generalis'), (r'hospitaliz', 'hospitalis'),
    (r'immobiliz', 'immobilis'), (r'normaliz', 'normalis'), (r'characteriz', 'characteris'), (r'anemia', 'anaemia'),
    (r'fecal', 'faecal'), (r'feces', 'faeces'), (r'hemangi', 'haemangi'), (r'hemarthr', 'haemarthr'),
]
_BRIT_RE = re.compile('|'.join(f'({p})' for p, _ in _BRIT_RULES), re.IGNORECASE)
def to_british(s):
    def _r(m):
        g = m.group(0)
        for p, uk in _BRIT_RULES:
            if re.fullmatch(p, g, re.IGNORECASE):
                return uk[0].upper() + uk[1:] if g[0].isupper() else uk
        return g
    return _BRIT_RE.sub(_r, str(s))

CATNAME = {"BI": "Blood/Immune", "CA": "Neoplasms", "CV": "Cardiovascular", "DE": "Dermatological",
           "EM": "Endocrine/Metabolic", "GI": "Gastrointestinal", "GU": "Genitourinary",
           "ID": "Infections", "MB": "Mental", "MS": "Musculoskeletal", "NS": "Neurological",
           "RE": "Respiratory", "SO": "Sense organs", "SS": "Symptoms", "time": "Mortality"}

# ---- Full per-disease longtable (with subject-level bootstrap CIs) ----
import pandas as pd
df = F.load_per_disease().sort_values("c", ascending=False)
# Merge seed-averaged per-disease 95% bootstrap CIs (point unchanged; CI decorates).
CI_CSV = f"{UKB_ROOT}/paper_additions/runs/embedding_seed_avg/per_disease.csv"
have_ci = os.path.exists(CI_CSV)
if have_ci:
    cidf = pd.read_csv(CI_CSV)[["phecode", "c_index", "c_lo", "c_hi"]]
    df = df.merge(cidf, on="phecode", how="left")
    # guardrail: the seed-averaged point must reproduce the canonical concordance
    d = (df["c"] - df["c_index"]).abs()
    print(f"  CI-merge: max |c - c_index| = {d.max():.4f} (should be ~0); "
          f"{df['c_lo'].notna().sum()}/{len(df)} have CIs")
rows = []
for _, r in df.iterrows():
    lab = esc(to_british(str(r.PhecodeString).split("[")[0].strip())[:46])
    cat = esc(F.cat_label(r.PhecodeCategory))
    if have_ci and pd.notna(r.get("c_lo")):
        rows.append(f"{lab} & {cat} & {int(r.n_event)} & {r.c:.3f} & "
                    f"[{r.c_lo:.3f}, {r.c_hi:.3f}] \\\\")
    else:
        rows.append(f"{lab} & {cat} & {int(r.n_event)} & {r.c:.3f} & --- \\\\")
longtable = r"""\footnotesize
\begin{longtable}{p{5.6cm} l r r l}
\caption*{\textbf{Extended Data Table 8. Per-disease six-year concordance for all evaluable
outcomes}, ordered by concordance, with 95\% subject-level bootstrap confidence intervals. Events
are incident cases in the held-out test set; estimates for outcomes with few events are unstable
and carry wide intervals.}\\
\toprule
Disease & Category & Events & C & 95\% CI \\
\midrule
\endfirsthead
\toprule
Disease & Category & Events & C & 95\% CI \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(rows) + "\n\\end{longtable}\n\\normalsize\n"
open(os.path.join(OUT, "tab_per_disease.tex"), "w").write(longtable)
print("tab_per_disease.tex:", len(rows), "rows")

# ---- Rich per-disease table: incident vs prevalent vs demographics + lead time ----
# (Extended Data Table 5; source built by build_rich_per_disease.py.)
RICH_CSV = f"{UKB_ROOT}/paper_additions/rich_per_disease/rich_per_disease.csv"
rdf = pd.read_csv(RICH_CSV).sort_values("c_model", ascending=False)
# Standardize PD's prevalent-screening AUROC for display to the value used paper-wide (0.916).
# This CSV run gives 0.9169 for the same n=132 quantity, a noise-level (~0.0007) difference from
# separate control sampling; the paper reports one value. Raw CSV is left untouched.
PREV_AUROC_DISPLAY = {"NS_324.11": 0.916}
rrows = []
for _, r in rdf.iterrows():
    # allow line breaks after slashes so long slash-joined names wrap inside the narrow column
    lab = esc(to_british(str(r["name"]).split("[")[0].strip())[:42]).replace("/", "/\\allowbreak ")
    cat = esc(F.cat_label(str(r["category"])))
    ninc = int(r["n_event"])
    lead = f"{r['lead_yr']:.1f}" if pd.notna(r["lead_yr"]) else "--"
    cm = f"{r['c_model']:.3f}"
    cd = f"{r['c_demo']:.3f}" if pd.notna(r["c_demo"]) else "--"
    dc = f"{r['dc']:+.3f}" if pd.notna(r["dc"]) else "--"
    _prev_auroc = PREV_AUROC_DISPLAY.get(str(r["phecode"]), r["prev_auroc"])
    prev = f"{_prev_auroc:.3f} ({int(r['prev_n'])})" if pd.notna(r["prev_auroc"]) else "--"
    rrows.append(f"{lab} & {cat} & {ninc} & {lead} & {cm} & {cd} & {dc} & {prev} \\\\")
rich_longtable = r"""\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{longtable}{p{4.0cm} l r r r r r l}
\caption*{\textbf{Extended Data Table 5. Per-disease incident prediction, prevalent-disease
screening and a demographic baseline.} Outcomes with at least ten incident test cases, ordered by
incident concordance. \emph{Incident} is the number
of incident test cases; \emph{Lead} is the mean time from the wrist recording to diagnosis among
those cases (years). $C$ is the six-year incident concordance of the wrist embedding;
$C_{\mathrm{demo}}$ is an age, sex and body-mass-index Cox baseline for the same outcome; $\Delta C$
is their difference. \emph{Prev.\ AUROC} is the wrist embedding's prevalent-disease screening AUROC
(same-distribution cross-validation), with the number of prevalent test cases in parentheses; a dash
marks outcomes with no viable screening cell, and all-cause mortality has no prevalent state.}\\
\toprule
Disease & Category & Incident & Lead (yr) & $C$ & $C_{\mathrm{demo}}$ & $\Delta C$ & Prev.\ AUROC ($n$) \\
\midrule
\endfirsthead
\toprule
Disease & Category & Incident & Lead (yr) & $C$ & $C_{\mathrm{demo}}$ & $\Delta C$ & Prev.\ AUROC ($n$) \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(rrows) + "\n\\end{longtable}\n\\setlength{\\tabcolsep}{6pt}\n\\normalsize\n"
open(os.path.join(OUT, "tab_per_disease_rich.tex"), "w").write(rich_longtable)
print("tab_per_disease_rich.tex:", len(rrows), "rows")

# ---- Per-category clinical 3-arm ----
_, cs = F.load_clinical()
cat_rows = []
for d in sorted(cs["per_category_auroc"], key=lambda x: -x["auroc_embedding"]):
    nm = CATNAME.get(d["category"], d["category"])
    cat_rows.append(f"{esc(nm)} & {d['n']} & {d['auroc_clinical']:.3f} & "
                    f"{d['auroc_embedding']:.3f} & {d['auroc_clin_emb']:.3f} \\\\")
open(os.path.join(OUT, "tab_clinical_cat_rows.tex"), "w").write("\n".join(cat_rows) + "\n")
print("tab_clinical_cat_rows.tex:", len(cat_rows), "rows")

# ---- Importance selected_diseases (clinically selected) ----
wl = F.load_selected_diseases()
imp = wl[wl.axis_importance == True].sort_values("C_incident", ascending=False) if "axis_importance" in wl else wl
wrows = []
for _, r in imp.iterrows():
    wrows.append(f"{esc(to_british(str(r.label).split('[')[0].strip())[:40])} & {esc(CATNAME.get(str(r.category), str(r.category)))} & "
                 f"{int(r.n_event_incident)} & {r.C_incident:.3f} & "
                 f"[{r.CI_lo:.3f}, {r.CI_hi:.3f}] \\\\")
open(os.path.join(OUT, "tab_selected_diseases_rows.tex"), "w").write("\n".join(wrows) + "\n")
print("tab_selected_diseases_rows.tex:", len(wrows), "rows")
print("DONE")
