"""Fixed selected-disease list for benchmark monitoring.

A checked-in set of clinically wrist-plausible phecodes whose per-disease Harrell C
is tracked on every benchmark run, so a change that quietly degrades a clinically
important disease is caught even when the headline mean C (averaged over ~391
phecodes) would hide it. This list does not drive model selection; that is the
three-way paired comparison.

Two groups: the three neurodegenerative diseases made evaluable by the participant-disjoint
split, and a wrist-plausible set spanning movement/mobility, sleep, cardiometabolic,
respiratory, mood, and all-cause mortality.

Phecode ids use the category-prefixed vocabulary (NS neurological, CV cardiovascular,
EM endocrine/metabolic, RE respiratory, MB mental). A watched code below its
prevalence floor is reported, not treated as an error.
"""
from __future__ import annotations

import json
import os
import re

# phecode id -> human label. Ordered by clinical domain.
SELECTED_DISEASES: dict[str, str] = {
    # neurodegenerative (evaluable under the participant-disjoint split)
    "NS_324.11": "Parkinson's disease",
    "NS_328.11": "Alzheimer's disease",
    "NS_328.1":  "Dementias",
    # movement / mobility (most directly wrist-observable)
    "NS_350.5":  "Repeated falls",
    "NS_325":    "Abnormality of gait and mobility",
    # sleep
    "NS_333.1":  "Sleep apnea",
    # cardiometabolic
    "CV_416.2":  "Atrial fibrillation and flutter",
    "CV_424":    "Heart failure",
    "EM_202.2":  "Type 2 diabetes",
    "EM_236.1":  "Obesity",
    # respiratory
    "RE_474":    "Chronic obstructive pulmonary disease [COPD]",
    # mood
    "MB_286.2":  "Major depressive disorder",
    # mortality
    "time_to_death": "All-cause mortality",
}

# Subset that is new in the participant-disjoint panel vs the earlier fixed split (for reporting).
DISJOINT_SPLIT_NEW = ("NS_324.11", "NS_328.11", "NS_328.1")


# Pre-registered wrist-plausibility tag. This is a label only; it never decides which
# diseases are detectable (that is empirical). It marks which detected diseases are
# surprising: a strong signal where a wrist -> activity/sleep/circadian -> disease link
# is not obvious a priori. Declared here, before the scan runs, so it cannot be
# retrofitted to explain away a surprise.
#
# Plausible = the modalities a wrist accelerometer can observe, the same domains the
# importance list above enumerates: cardiometabolic (CV, EM), respiratory (RE), mood
# (MB), and the movement/sleep subset of neurological (NS). Other categories (cancers,
# dermatology, GI, GU, blood/immune, musculoskeletal-non-mobility, infections, sense
# organs) are low-prior.
PLAUSIBLE_PREFIXES = frozenset({"CV", "EM", "RE", "MB"})

# NS is split by keyword, not taken whole: only movement/mobility and sleep/circadian NS
# codes are wrist-plausible (Parkinson's, gait, falls, sleep apnea). Other NS codes
# (dementias, epilepsy, neuropathy, headache) are low-prior, so a strong wrist signal
# there is a genuine surprise. Matched on the phecode's human description.
NS_PLAUSIBLE_KEYWORDS = re.compile(
    r"parkinson|gait|mobilit|fall|tremor|dyskines|restless|chorea|ataxia|dystonia|"
    r"sleep|apnea|apnoea|insomnia|hypersomnia|narcolep|circadian|somnolen",
    re.IGNORECASE,
)


def is_plausible(phecode: str, description: str = "") -> bool:
    """Pre-registered wrist-plausibility tag. It never gates detectability; it only
    marks which detected diseases are surprising. `description` is the phecode's human
    label (needed for the NS movement/sleep split). Mortality is never a surprise."""
    if phecode == "time_to_death":
        return True
    prefix = phecode.split("_", 1)[0]
    if prefix in PLAUSIBLE_PREFIXES:
        return True
    if prefix == "NS":
        return bool(NS_PLAUSIBLE_KEYWORDS.search(description or ""))
    return False


# The data-driven tiers (detectability/surprise) and the per-disease reference snapshot
# live in selected_diseases_frozen.json, generated once by curate_selected_diseases.py and checked in.
FROZEN_PATH = os.path.join(os.path.dirname(__file__), "selected_diseases_frozen.json")


def load_frozen(path: str | None = None) -> dict | None:
    """Parsed selected_diseases_frozen.json, or None if it has not been curated yet, so the
    benchmark degrades gracefully to the importance-only list before the first curation."""
    p = path or FROZEN_PATH
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)
