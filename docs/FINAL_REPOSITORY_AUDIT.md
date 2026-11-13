# Final scientific repository audit

**Audit date:** 2026-11-13

**Scope:** Read-only scientific and release-safety review before the first research-data commit

**Repository:** `D:\IEEE_TIM_related\TM_journal\flexible-pcb-parasitic-admittance`

No scientific data, figure, parameter, or source algorithm was modified. The only file
created by this audit is this report. No file was staged, committed, pushed, published,
or used for expensive re-optimization.

## 1. Publication-figure verification

All five PNG figures were visually inspected. Their corresponding PDF/SVG variants,
where present, are recorded in `docs/provenance.csv` and passed byte-integrity checks.
The project owner additionally confirmed on 2026-11-13 that all publication figures
display normally and are scientifically acceptable as rendered.

### Figure 2 — unloaded matrix at 100 kHz: PASS WITH PACKAGING GAP

- Figure: `figures/publication/Fig2_terminal_pair_compensation.{png,pdf,svg}`.
- Loading condition: unloaded reference, 100 kHz, all 91 terminal pairs.
- Generator: `src/scientific/rebuild_Fig2_Fig3_Fig4_E2_windows_fixed_download.py`.
- Original source: `Chapter4_E2_One_Run_Validated/run_20261108T184226_941000Z/E2_Fig2_data.csv`.
- Audit SHA-256: `7cb72fcd070a48ee0a521a9dfb215bff7d8e96805a326fb9b71a443f4ea89626`.
- The original CSV exists and its recalculated hash matches the figure audit. It has
  273 rows: 91 pairs for each of three initialization fractions. The plotted figure uses
  the 30% initialization.
- The plot correctly distinguishes raw measured impedance, parasitic-compensated
  impedance, and intrinsic-network forward response.
- Scientific limitation: the unloaded intrinsic baseline was calibrated using all 91
  unloaded paths, so this is an in-sample consistency visualization, not independent
  intrinsic-impedance ground truth.
- **Packaging gap:** `E2_Fig2_data.csv` is not present in this repository. The audit JSON
  identifies its external absolute path and hash, but a fresh checkout cannot recreate
  Fig. 2 without obtaining that file.

### Figure 3 — five-frequency held-out prediction errors: PASS

- Figure: `figures/publication/Fig3_frequency_validation.{png,pdf,svg}`.
- Source: `results/figure_sources/Fig3_frequency_source.csv` and
  `results/figure_sources/figure_build_audit.json`.
- Generator: `src/scientific/rebuild_Fig2_Fig3_Fig4_E2_windows_fixed_download.py`.
- Frequencies: 46.4159, 100, 215.443, 464.159 kHz, and 1 MHz.
- The plot shows held-out complex-impedance relative L2 error for the original unloaded
  baseline and the frequency-recalibrated unloaded baseline. It is not a plot of
  compensated-versus-intrinsic response.
- Audited means are 18.3260% before and 7.2468% after recalibration for the figure's
  corrected-symmetric source selection. This figure-specific after value is close to,
  but not numerically identical with, the 7.2489% complete 630-row aggregate reported
  by `scripts/reproduce_reported_metrics.py`; the audit records the source policy.

### Figure 4 — 19 individually loaded sensors: PASS

- Figure: `figures/publication/Fig4_sensorwise_holdout.{png,pdf,svg}`.
- Source: `results/figure_sources/Fig4_sensorwise_source.csv`.
- Generator: `src/scientific/rebuild_Fig2_Fig3_Fig4_E2_windows_fixed_download.py`.
- Loading conditions: the 19 individually loaded sensing elements S1–S19.
- S2 complex-Z relative L2 mean is `0.1096440018040373`, i.e. **10.9644%**, agreeing
  with the reported 10.96%. S2 is the largest bar; S13 is the minimum at 5.4711%.
- Each sensor value aggregates 30 computational evaluations (five frequencies, two
  folds, three starts) of one physical scan, not 30 independent physical experiments.

### Figure 5 — single S6 loading at 100 kHz: PASS

- Figure: `figures/publication/Fig_loaded_recovery_sensor6_100000Hz_init30pct_fold0.{png,pdf}`.
- Source table: `results/figure_sources/Fig_loaded_recovery_sensor6_100000Hz_init30pct_fold0_91pair_data.csv`.
- Generator: `src/scientific/E2_loaded_recovery_figures_phase_legend.py`.
- Original sweep: `2026_11_05-09_41_39_541138_sensor6_sweep009_91pairs_10freq.txt`.
- Packaged validation sweep index: 8; selected frequency: 99,999.99999999997 Hz.
- The source table contains all 91 complex measurements, 46 fitting pairs and 45
  held-out pairs. Its measured values agree with the packaged NPZ to a maximum absolute
  serialization difference of `2.91e-11` ohm.
- Complete-model measured-versus-predicted relative L2 over all 91 pairs is 6.2794%;
  the held-out complex-Z error recorded by the audit is 5.8140%.

### Figure 6 — simultaneous S2 and S16 loading at 100 kHz: PASS

- Figure: `figures/publication/Fig_loaded_recovery_sensor2_16_100000Hz_init30pct_fold0.{png,pdf}`.
- Source table: `results/figure_sources/Fig_loaded_recovery_sensor2_16_100000Hz_init30pct_fold0_91pair_data.csv`.
- Generator: `src/scientific/E2_loaded_recovery_figures_phase_legend.py`.
- Original sweep: `2026_11_05-09_37_35_189824_sensor2-16_sweep005_91pairs_10freq.txt`.
- Packaged validation sweep index: 4; selected frequency: 99,999.99999999997 Hz.
- The source table contains all 91 complex measurements, 46 fitting pairs and 45
  held-out pairs. Its measured values agree with the packaged NPZ to a maximum absolute
  serialization difference of `2.91e-11` ohm.
- Complete-model measured-versus-predicted relative L2 over all 91 pairs is 10.5304%;
  the held-out complex-Z error recorded by the audit is 10.8518%.

### Direct comparison of Figures 5 and 6

The plots use different raw files and different packaged sweep indices. For their 91
complex measured impedances,

`||Z_S6 - Z_S2,S16||_2 / ||Z_S6||_2 = 0.2003013289` (**20.0301%**).

The symmetric relative difference is

`2 ||Z_S6 - Z_S2,S16||_2 / (||Z_S6||_2 + ||Z_S2,S16||_2) = 0.2081767400`
(**20.8177%**). The arrays are not identical.

### Model-equation consistency for Figures 5 and 6: PASS

The preserved generator defines

- `Z_intrinsic = 1 / Y_intrinsic`,
- `Z_full = 1 / (Y_intrinsic + Yp)`, and
- `Z_compensated = 1 / (1 / Z_measured - Yp)`.

Reconstruction with the 30% parameter table at 100 kHz reproduced both compensated
arrays with maximum pointwise relative discrepancy below `8.2e-16` and both complete
forward arrays below `6.0e-16`. The complete-model equivalence guard in the generator
also compares the explicit formula with the shared prediction function. The green
intrinsic response is correctly compared with the compensated orange response only as
a consistency diagnostic; held-out performance uses raw measured impedance versus the
complete parasitic-aware prediction.

The parameter CSV uses the same pair-index order as the figure tables, although labels
for pairs among nodes 8–14 use the opposite textual orientation. This is the documented
unordered-pair label normalization and does not change index alignment.

## 2. README review

### Verified as correct

- All three Markdown links resolve inside the repository.
- All listed top-level directories and referenced verification scripts exist.
- Figure 1 is explicitly described as excluded and is not linked as an existing image.
- The README distinguishes lightweight saved-result aggregation from expensive
  historical identification.
- It does not claim fully portable end-to-end reproduction; the reproducibility guide
  explicitly states that preserved scientific scripts retain original absolute paths.
- The 14-node, 19-element, 91-pair description; 644-state identification; 22-sweep
  validation; five-frequency evaluation; and 45/46 held-out split are consistent with
  the packaged data and source code.
- The README correctly distinguishes measured, compensated, intrinsic forward, and
  complete forward responses.

### Corrections recommended before a public release

1. **Data availability was updated after this audit.** The private repository now
   contains 14 original historical acquisition TXT files and 22 original multifrequency
   sweep TXT files. Public redistribution is still not authorized by their presence.
2. **Internal model shorthand corrected.** Publication-facing documentation now uses
   “selected weak-S2-prior fixed-parasitics parameterization.” `E2` is retained only
   where necessary to identify byte-identical source filenames, archived result fields,
   and historical run metadata.
3. **Qualify the no-parasitics statement.** State explicitly that it is a **quick
   diagnostic ablation** that inherits the proposed model's historical intrinsic
   reference as its no-load prior and is not a fully independently optimized historical
   control.
4. **Correct figure-source completeness.** The README currently says figure-source
   tables are packaged. Change this to “selected figure-source tables” or add the exact
   audited Fig. 2 CSV after release approval. Fig. 2 is not reproducible from a fresh
   checkout as currently packaged.
5. **Separate verification from figure regeneration.** The quick commands verify
   integrity and saved metrics but do not regenerate Figs. 2–6. This is mostly clear,
   but an explicit sentence would eliminate ambiguity.

No broken link or false claim of a newly executed optimization was found.

## 3. Verified numerical results

The existing lightweight checks completed successfully without optimization:

- Historical states selected: 644 from an 804-state archive.
- Independent validation: 22 sweeps × 10 frequencies × 91 pairs.
- Loaded held-out evaluations: 630.
- Selected fixed-parasitics complex-Z relative L2: `0.07248932352514484` = **7.2489%**.
- Selected fixed-parasitics magnitude MAPE: `0.05101220518186292` = **5.1012%**.
- Selected fixed-parasitics phase MAE: **1.4585956612°**.
- No-parasitics complex-Z relative L2: **30.7216%**.
- No-parasitics magnitude MAPE: **21.6947%**.
- No-parasitics phase MAE: **1.461339°**.
- Provenance verification: 57 packaged research files matched their recorded SHA-256
  hashes.

The no-parasitics run audit explicitly labels its mode `quick_ablation`. It inherits the
selected fixed-parasitics parameterization's historical intrinsic reference for the
no-load prior, sets `Yp = 0`,
and refits unloaded/loaded intrinsic parameters. It is a predictive diagnostic, not a
fully independent historical identification.

## 4. Reproducibility limitations

- No expensive historical re-identification was performed in this audit.
- The preserved scientific scripts contain absolute `D:\IEEE_TIM_related\...` paths.
  They are auditable source snapshots, not portable entry points.
- Fig. 2's exact source CSV is absent from the repository.
- Raw acquisition files are present: 14 historical TXT files containing 804 states
  (first 644 used) and 22 multifrequency TXT sweeps. Their hashes match the extraction
  audit and validation manifest.
- Unloaded calibration uses all 91 unloaded paths and is in-sample.
- The 630 rows reuse 21 physical loaded scans across frequencies, folds, and starts.
- No direct intrinsic-impedance or parasitic-free physical ground truth is available.
- PyTorch checkpoints and optimization histories are absent, so warm-start continuation
  of the original identification is not packaged.
- The five-frequency study is exploratory because model selection used 100 kHz results.

## 5. Release-safety and staging classification

### Safe candidates for a local documentation/code commit

- `.gitignore`
- `README.md` after the corrections above
- `requirements.txt`
- `docs/*.md` except that release claims must remain conservative
- `scripts/*.py`
- `data/metadata/` topology/pair-order files
- `src/README.md`, `data/README.md`, `parameters/README.md`, `results/README.md`, and
  `figures/README.md`

The scientific source snapshots under `src/scientific/` contain internal absolute
project paths but no detected credentials, email addresses, API keys, or usernames.
They are suitable for a private local commit; path disclosure should be reviewed before
public release.

### Awaiting explicit release permission

- All files under `data/identification/` and `data/validation/`.
- All files under `data/Load_label/` and `data/Multifrequency/`.
- Measurement-derived tables under `results/validation/`, `results/no_parasitics/`, and
  `results/figure_sources/`, especially the two `*91pair_data.csv` files.
- Publication figures under `figures/publication/` if journal/data-owner policy applies.
- `docs/provenance.csv` and JSON audit/fit-summary files, because they expose internal
  absolute project paths and original measurement filenames.
- Final parameter tables under `parameters/selected_weak_s2_prior/` pending confirmation that
  fitted research parameters are approved for release.
- The omitted Fig. 2 source CSV, if later added.
- Figure 1, which must remain excluded until adapted-image redistribution permission is
  documented.

No `.pt`, `.pth`, `.ckpt`, ZIP, 7z, or RAR files are present. No credentials or personal
contact information were detected. Internal Windows project paths and timestamped raw
measurement filenames are present in source, CSV/JSON audit, and provenance files.

### `.gitignore` assessment

The ignore file covers Python caches, virtual environments, editor state, `.env` files,
common private/backup/temp patterns, logs, and generated audit output. It intentionally
does **not** ignore measurement archives, parameters, result tables, figures, provenance,
or scientific source snapshots. Therefore a broad `git add .` would include every
release-sensitive research artifact. `.gitignore` is not an authorization control;
use an explicit allow-list when staging.

## 6. Recommendation

**NO-GO for a single first research-data commit at this time.**

The scientific checks pass, but release authorization for the experimental archives and
measurement-derived artifacts has not been documented, and the exact Fig. 2 source CSV
is missing from the repository. Do not use `git add .`.

The project owner's confirmation that all figures display normally resolves the visual
figure-review question, but not the experimental-data permission or Fig. 2 source-data
packaging issues.

### Public terminology decision

`E2` is an internal historical experiment/run identifier and must not be presented as
the public model name. Public text uses “forward-model-based identification and
compensation with fixed path-dependent parasitic admittances” for the method and
“selected weak-S2-prior fixed-parasitics parameterization” for the selected parameters.
Internal identifiers remain in immutable source/result artifacts only for provenance.

**GO for a narrowly scoped local documentation/code-only commit** after reviewing the
README corrections, using an explicit path allow-list. A later research-data commit can
proceed after coauthor/institutional approval is recorded and the Fig. 2 source decision
(include with approval or document its intentional external availability) is resolved.
