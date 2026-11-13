# Distributed parasitic-admittance identification in a flexible PCB sensor matrix

This private repository packages the measured data, fixed parameters, scientific
source code, numerical results, and publication figures used for the associated
*tm* journal manuscript. The system has 14 terminal nodes, 19 sensing elements,
and all 91 unordered terminal-pair impedance measurements.

The repository now includes the 14 original `data/Load_label/` acquisition files.
They contain 804 measured states at 100 kHz; the published historical identification
uses the first 644 states exactly as specified by the preserved source. It also includes
the 22 original files in `data/Multifrequency/`: one unloaded reference and 21 loaded
10-frequency sweeps.
Reported results use five frequencies, frequency-specific unloaded calibration,
complementary 45/46-pair loaded fits, and predictions on the held-out pairs.

## Quick verification

```powershell
python -m pip install -r requirements.txt
python scripts/export_metadata.py
python scripts/build_provenance_manifest.py
python scripts/validate_repository.py --verify-provenance
python scripts/reproduce_reported_metrics.py
```

These commands do not run the expensive historical identification. The last command
re-aggregates published metrics from packaged per-evaluation tables.

## Layout

- `src/scientific/`: byte-identical scientific scripts from the research workspace.
- `scripts/`: portable integrity, metadata, provenance, and metric checks.
- `data/identification/`: historical impedance archive and extraction records.
- `data/validation/`: 22-sweep multifrequency archive and extraction records.
- `data/Load_label/`: 14 original 100 kHz acquisition files (804 states; first 644 used).
- `data/Multifrequency/`: 22 original multifrequency sweep files.
- `data/metadata/`: generated topology and canonical pair ordering.
- `parameters/`: final selected weak-S2-prior fixed-parasitics parameters for the
  5%, 30%, and 70% initialization fractions.
- `results/`: held-out validation, no-parasitics control, and figure-source tables.
- `figures/publication/`: final Figs. 2–6 in available formats.
- `docs/`: source inventory, provenance, release review, and reproducibility details.

## Expected results

Across 630 computational evaluations (21 loaded scans × 5 frequencies × 2
partitions × 3 starts), the fixed-parasitics method gives mean complex-impedance
relative L2 error 7.249%, magnitude MAPE 5.101%, and phase MAE 1.459°. The
recalibrated no-parasitics control gives 30.722%, 21.695%, and 1.461°,
respectively. These are repeated evaluations of 21 physical loaded sweeps, not
630 independent hardware measurements.

Measured, parasitic-compensated, intrinsic forward-predicted, and complete
parasitic-aware forward-predicted responses remain distinct. Held-out errors compare
measured impedance with the complete forward prediction, including parasitics.

## Limitations and release status

- Unloaded calibration uses all 91 unloaded pairs and is therefore in-sample.
- No direct parasitic-free or intrinsic-impedance ground truth is available.
- The no-parasitics result is a diagnostic ablation, not an independently optimized
  historical control.
- The five-frequency evaluation is exploratory because the selected weak-S2-prior
  fixed-parasitics parameterization was chosen using 100 kHz validation.
- Figure 1 is excluded: it is adapted from prior work and redistribution permission
  is not established.
- Experimental archives require explicit review before any public release.
- The publication figures were manually reviewed by the project owner on 2026-11-13
  and confirmed to display normally.

See [the reproducibility guide](docs/REPRODUCIBILITY.md), [source inventory](docs/SOURCE_INVENTORY.md),
and [release review](docs/RELEASE_REVIEW.md).

## Terminology and archived identifiers

The publication-facing method name is **forward-model-based identification and
compensation with fixed path-dependent parasitic admittances**. The selected parameter
set is described as the **selected weak-S2-prior fixed-parasitics parameterization**.
`E2` is only a historical internal run identifier. It is retained inside byte-identical
source filenames, archived result columns, and audit metadata solely for traceability;
it is not the public name of the model or method.
