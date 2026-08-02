# Wrist accelerometry and future health

Analysis code for the study that predicts incident disease across organ systems from
UK Biobank wrist accelerometry. One week of wrist accelerometry is embedded by two frozen,
previously published self-supervised accelerometer foundation models, the **Human Activity
(HA) model** and **AcceleRest**. The resulting participant-level embeddings are read by a
rolling-window Cox proportional-hazards model that estimates the time to first onset of each
of about 390 outcomes (389 phecodes plus all-cause mortality).

This repository contains the analysis code for full inspection and re-implementation. It
does not contain UK Biobank data, derived embeddings, or trained model weights.

## Code availability

The code central to this study is provided here for inspection and re-implementation. It
comprises the accelerometry preprocessing and feature/embedding-extraction pipeline, the
downstream disease-prediction model training and evaluation, and the scripts that generate
the figures and tables.

The code cannot be provided as a runnable capsule, for two reasons outside our control.
First, the analyses use individual-level UK Biobank data, which we are not permitted to
share or egress under the UK Biobank Material Transfer Agreement. Approved researchers can
obtain the identical data through a UK Biobank application (this study used Application
62249) and reproduce the results by running the code within the UK Biobank Research Analysis
Platform. Second, in line with UK Biobank policy, including its current interim position on
the use of AI in UK Biobank research, under which trained models and their weights are
treated as the equivalent of individual-level data, no model weights or participant-level
embeddings derived within the UK Biobank environment can be shared, egressed, published, or
commercialised. This applies both to the accelerometer foundation-model encoders and to the
downstream disease-prediction models trained on UK Biobank data.

The repository therefore contains the analysis code but not the UK Biobank data or any
trained model weights, and the results are reproducible only by approved researchers within
the UK Biobank environment. This statement will be updated if UK Biobank policy changes
before publication.

## Data and encoders

- **UK Biobank data** is available to approved researchers through the UK Biobank Access
  Management System (https://www.ukbiobank.ac.uk). This study used the resource under
  Application 62249. Individual-level data cannot be redistributed; derived embeddings and
  trained model weights are treated as equivalent to individual-level data and stay within
  the UK Biobank environment.
- **The two accelerometer encoders** (the Human Activity model and AcceleRest) are
  previously published and are obtained, frozen, from their original sources. They are
  loaded by `ukb_disease/extraction/run_extraction.py`. They are not redistributed here.

All input and output locations resolve under a single root. Set the `UKB_ROOT` environment
variable to your prepared data directory; see `ukb_disease/paths.py` and
`ukb_disease/baseline/config.py` for the expected subpaths.

## Pipeline

0. **Preprocessing** (`ukb_disease/preprocessing/`): convert raw Axivity `.cwa.gz`
   recordings into standardized 30 Hz H5 files (dropout handling, low-pass, resampling,
   gravity calibration, non-wear masking, day/night quality tables).
1. **Embedding extraction** (`ukb_disease/extraction/`): run the two frozen encoders over
   each preprocessed recording and cache per-patch embeddings, with non-wear epochs masked.
2. **Pooling and modelling** (`ukb_disease/baseline/`): pool per-patch embeddings into
   per-day summaries (mean, and distributional / L-moment pooling), then train the
   rolling-window Cox model over the outcome panel. Includes the clinical, activity-summary
   and demographic-recovery baselines.
3. **Evaluation** (`ukb_disease/benchmark/`): per-disease and pooled concordance, the
   concordance benchmark, and the curated selected-disease list.
4. **Figures and tables** (`figbuild/`): builders that turn the model outputs into the
   manuscript figures and LaTeX tables.

Supporting analyses: `ukb_disease/paper_experiments/` (regional transportability, prodromal
timing, added value beyond the shared risk component, calibration, decision-curve utility,
proportional-hazards checks), `ukb_disease/paper_additions/` (disease-risk correlation, the
Shared Actigraphic Risk Component, burden-adjusted false-positive analyses),
`ukb_disease/interpretability/` (per-disease source attribution), `ukb_disease/genetics/`
(polygenic-risk-score comparison), `ukb_disease/investigations/`, and
`ukb_disease/screening/` (prevalent-disease screening metrics).

## System requirements

- **Operating system.** The code was developed and run on Linux (64-bit). It has no
  operating-system-specific dependencies but has only been tested on Linux.
- **Python and packages.** Python 3.12.1, with the package versions pinned in `requirements.txt`
  (numpy 2.2.5, pandas 2.2.3, scipy 1.15.2, scikit-learn 1.5.2, statsmodels 0.14.4, lifelines 0.30.3,
  PyTorch 2.5.1, matplotlib 3.10.1, h5py 3.13.0, pyarrow 20.0.0, PyYAML 6.0.3, joblib 1.5.3,
  Pillow 11.2.1, actipy 3.5.0). These are the versions used for the results in the manuscript.
- **Hardware.** A CUDA-capable GPU is required for embedding extraction and for training the survival
  model; pooling, evaluation, and the figure and table scripts run on CPU. No other non-standard
  hardware is required.

## Running

```bash
pip install -r requirements.txt
export UKB_ROOT=/path/to/your/data

python -m ukb_disease.preprocessing.preprocess_ukb_accelerometry --help  # preprocess raw CWA
python -m ukb_disease.extraction.run_extraction                          # cache embeddings
python -m ukb_disease.baseline.train_rolling_window_cox --help           # train the survival model
python -m ukb_disease.benchmark.concordance_benchmark                    # evaluate
python figbuild/build_main.py                                            # build the main figures
```

Installing the pinned dependencies takes a few minutes on a standard machine.

The code is fully reproducible only by approved researchers within the UK Biobank Research
Analysis Platform. Generated intermediate and output files (embeddings, per-day summaries,
run outputs, figure data) are written under `UKB_ROOT`, may contain participant identifiers,
and are not part of this repository; the `.gitignore` excludes data and artifact files so
they stay within the UK Biobank environment.

A few figure inputs are prepared outside this code and read at build time: Figure 1 is
composed in external illustration software from the panel exports (`figbuild/build_fig1*.py`,
`figbuild/export_fig1_panels_svg.py`), and a small number of descriptor and ablation summary
tables are prepared manually.

## Demo

A self-contained demonstration on example data cannot be provided, because the only inputs to the
pipeline are individual-level UK Biobank accelerometry, which cannot be shared or egressed under the
UK Biobank Material Transfer Agreement (see [Code availability](#code-availability)). Approved UK Biobank
researchers can run the pipeline unchanged on the identical data within the UK Biobank Research Analysis
Platform. When run on UK Biobank data, it produces, in order, the cached per-patch and per-day
embeddings, the per-participant predicted-hazard matrix over the outcome panel, the per-disease and
pooled concordance tables, and the manuscript figures and tables. Run time is dominated by the embedding
extraction over the full cohort on a GPU; the subsequent pooling, model training, and evaluation are
comparatively fast.

## Figure and table crosswalk

Output filenames are historical and are kept so they match the manuscript sources; this
table is the authoritative map from each manuscript item to the code that produces it.
In-code `figN` / `edN` labels refer to these historical output filenames, not to the
displayed number in the manuscript; use this table to translate between the two.

| Manuscript item | Content | Producer | Output file |
|---|---|---|---|
| Figure 1 | Overview and transportability | hand-composed from `figbuild/build_fig1*.py`, `export_fig1_panels_svg.py` | fig1.pdf |
| Figure 2 | Shared Actigraphic Risk Component and correlations | `figbuild/build_axis_merged.py::fig_merged` | fig_disease_corr.pdf |
| Figure 3 | Age, sex, BMI and clinical baseline | `figbuild/build_main.py::fig4_wrist_vs_clinic` | fig4_wrist_vs_clinic.pdf |
| Figure 4 | Per-disease source attribution | `figbuild/build_ed.py::fig_source_attribution` | fig_source_attribution.pdf |
| Figure 5 | Prodromal neurodegeneration | `figbuild/build_paper_experiments.py::fig3_prodromal` | fig3_prodromal.pdf |
| Ext Data (residual signal) | Signal beyond a flexible demographic model | `figbuild/build_ed.py::ed6_residual` | ed6_residual.pdf |
| Ext Data (PD benchmark) | Parkinson's disease vs a bespoke model | `figbuild/build_ed.py::ed9_pd_benchmark` | ed9_pd_benchmark.pdf |
| Ext Data (genetics forest) | Genetics fusion across matched outcomes | `figbuild/build_genetics.py::ed_genetics_forest` | ed_genetics_forest.pdf |
| Ext Data (ablations) | Feature-source ladder and input titration | `figbuild/build_main.py::fig6_saturation` | fig6_saturation.pdf |
| Ext Data (calibration, utility) | Calibration and decision-curve utility | `figbuild/build_paper_experiments.py::fig6_calib_utility` | fig6_calib_utility.pdf |
| Ext Data (screening) | Prevalent-disease screening | `figbuild/build_main.py::fig_screening_prevalent` | fig_screening.pdf |
| Ext Data (source attribution, full) | Source attribution for all matched outcomes | `figbuild/build_ed.py::fig_source_attribution_supp` | fig_source_attribution_supp.pdf |
| Ext Data (axis supplement) | Anatomy of the shared risk component | `figbuild/build_axis_merged.py::fig_axis_supp` | fig_axis_supp.pdf |
| Ext Data (disease-specific) | Discrimination beyond the shared component | `figbuild/build_phewas_specific.py` | fig_phewas_specific.pdf |
| Ext Data (descriptors) | Shared component vs health descriptors | `figbuild/build_sharc_descriptors.py::fig_sharc_descriptors` | fig_sarc_descriptors.pdf |
| Supplementary (age/sex) | Linear recovery of age, sex, BMI | `figbuild/build_ed.py::ed5_agesex` | ed5_agesex.pdf |
| Supplementary (matched genetics) | Genetics must be disease-matched | `figbuild/build_genetics.py::ed_genetics_matched` | ed_genetics_matched.pdf |
| Supplementary (screening genetics) | High-specificity screening with a PRS | `figbuild/build_supplementary.py::ed_screening_genetics` | ed_screening_genetics.pdf |
| Supplementary (false positives) | What the false positives carry | `figbuild/build_supplementary.py::ed_confusion_targets` | ed_confusion_targets.pdf |
| Supplementary (residual correlation) | Correlation after the shared component | `figbuild/build_additions.py::ed_shared_residual_corr` | ed_shared_residual_corr.pdf |
| Table (ablations) | Stage-by-stage search | `figbuild/figlib.py::write_ablation_table` | tab_ablations.tex |
| Table (per-disease, rich) | Incident, prevalent and demographic per disease | `figbuild/gen_tables.py` | tab_per_disease_rich.tex |
| Table (genetics) | Polygenic-score cross-reference | `ukb_disease/genetics/make_table.py` | tab_genetics.tex |
| Table (per-disease concordance) | Per-disease concordance longtable | `figbuild/gen_tables.py` | tab_per_disease.tex |
| Table (washout) | Largest washout concordance losses | `figbuild/gen_washout_table.py` | tab_washout_drops.tex |
| Table (activity fields) | Accelerometer summary fields used as a baseline | `figbuild/gen_activity_fields_table.py` | tab_activity_fields.tex |

Three input tables in the supplement (`tab_confusion_defrail`, `tab_screening_genetics`,
`tab_corrections`) are prepared by hand from the corresponding analysis outputs.

## Terminology

The code uses the manuscript's vocabulary. For reference:

- **participant-disjoint split** (`disjoint_split`): the train / validation / test split, held
  disjoint at the participant level.
- **wrist embedding** (`embedding`): the frozen accelerometer-embedding representation read by
  the survival model.
- **HA** = Human Activity model, **AR** = AcceleRest: the two frozen encoders.
- **SHARC**: the Shared Actigraphic Risk Component, the leading principal component of the
  demographic-residualised per-disease predicted risks.
- **rolling-window Cox** (`train_rolling_window_cox`): the multilabel Cox proportional-hazards
  survival model over three-day windows.
- **distributional / L-moment pooling**: per-day pooling that retains the first sample
  L-moments in addition to the mean.
- **source attribution**: the per-representation, day and night contribution decomposition.
- **concordance benchmark**: per-disease and pooled concordance evaluation.

## Citation

If you use this code, please cite the associated paper (full reference to be added on
publication).

## License

CC BY-NC 4.0. See [LICENSE](LICENSE).
