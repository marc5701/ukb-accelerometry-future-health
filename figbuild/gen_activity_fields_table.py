#!/usr/bin/env python3
"""Generate draft/tab_activity_fields.tex: the Extended Data table listing all 100 UK Biobank
activity-summary fields with their exact UK Biobank Showcase descriptions.

Input:  draft/activity_field_definitions.csv  (field_id, group, title) taken verbatim from
        the UK Biobank Showcase (field.cgi?id=<ID>), structurally validated (monotonic
        intensity thresholds 1..2000 mg; sequential hours 00..23; days Mon..Sun).
Output: draft/tab_activity_fields.tex  (a grouped longtable).
"""
from pathlib import Path
import pandas as pd

DRAFT = str(Path(__file__).resolve().parent.parent / "draft")
CSV = f"{DRAFT}/activity_field_definitions.csv"
OUT = f"{DRAFT}/tab_activity_fields.tex"

GROUPS = [
    ("overall", "Overall"),
    ("day_of_week", "Average acceleration by day of week"),
    ("hour_of_day", "Average acceleration by hour of day"),
    ("intensity", "Acceleration intensity distribution"),
]


def esc(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def fmt_title(s: str) -> str:
    s = esc(s)                     # esc leaves '<', '>', '=' untouched
    s = s.replace("<=", r"$\leq$")  # render the inequality in math mode
    return s


CAPTION = (
    r"\caption*{\textbf{Extended Data Table \theexttable. UK Biobank accelerometer summary "
    r"fields used as the activity baseline.} The 100 UK Biobank data-fields that constitute "
    r"the activity-summary panel compared against the wrist embedding (Figure~\ref{fig:clinic}c; "
    r"Methods). Field identifiers and descriptions are taken verbatim from the UK Biobank "
    r"Showcase. UK Biobank derives these fields from each participant's raw wrist recording with "
    r"the Oxford accelerometer-processing pipeline~\cite{doherty2017}. Acceleration averages are "
    r"reported in milli-gravities; each intensity-distribution field is the fraction of time "
    r"during which acceleration was at or below the stated threshold.}"
)

HEAD = (
    r"\footnotesize" "\n"
    r"\begin{longtable}{@{}l l@{}}" "\n"
    + CAPTION + r"\\" "\n"
    r"\toprule" "\n"
    r"Field & UK Biobank description \\" "\n"
    r"\midrule" "\n"
    r"\endfirsthead" "\n"
    r"\toprule" "\n"
    r"Field & UK Biobank description \\" "\n"
    r"\midrule" "\n"
    r"\endhead" "\n"
    r"\bottomrule" "\n"
    r"\endfoot" "\n"
)

df = pd.read_csv(CSV)
tmap = dict(zip(df.field_id, df.title))

rows = []
for gi, (gkey, glabel) in enumerate(GROUPS):
    if gi > 0:
        rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{2}{@{}l}{\textit{" + glabel + r"}}\\")
    for fid in df[df.group == gkey].field_id:
        rows.append(f"{int(fid)} & {fmt_title(tmap[fid])} " + r"\\")

TAIL = "\n" + r"\end{longtable}" + "\n" + r"\normalsize" + "\n"

with open(OUT, "w") as f:
    f.write(HEAD + "\n".join(rows) + TAIL)

print(f"wrote {OUT}: {len(df)} fields across {len(GROUPS)} groups")
