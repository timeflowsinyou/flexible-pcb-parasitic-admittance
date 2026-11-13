# Reproducibility guide

## Scope and environment

Original scientific scripts are copied byte-for-byte under `src/scientific`; their
absolute workspace paths are retained for provenance. Portable scripts under `scripts`
validate the package and reproduce reported aggregation without changing inputs.

Dependencies were derived from imports: NumPy, pandas, SciPy, Matplotlib, and PyTorch.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/export_metadata.py
python scripts/build_provenance_manifest.py
python scripts/validate_repository.py --verify-provenance
python scripts/reproduce_reported_metrics.py
python -m compileall -q scripts src/scientific
```

These checks do not run optimization. `build_provenance_manifest.py` confirms that
every packaged research artifact is byte-identical to its original.

## Scientific dependency sequence

1. `V1_extract_multifrequency_91pairs_v1_1.py` extracts the 22 raw sweeps.
2. `experiment_I5_checkpointed_convergence.py` defines the grounded nodal model and
   original 644-state identification.
3. `experiment_I5_S2_E1_E2_E3.py` produces the selected weak-S2-prior parameter starts.
4. `V2_2_two_offset_hypotheses.py` supplies shared forward routines.
5. `V2_4_fixed_parasitic_intrinsic_drift.py` performs frequency-specific unloaded
   calibration and loaded-state fitting.
6. `V2_6_validate_22_sweeps_fixed_parasitics.py` evaluates all sweeps.
7. `V2_7_publication_validation_v1_1.py` and
   `validate_E2_locality_multifrequency.py` aggregate validation.
8. `E2_vs_NoParasitics_control.py` runs the diagnostic ablation.
9. The two figure scripts generate Figs. 2–6.

The preserved sources still reference the original workspace. Changing their execution
contract without a full expensive rerun would weaken provenance. For a full rerun, use
the documented original layout or configure copies in a separate working directory.
Identification was not run during repository setup.

## Data conventions

- Historical `Z_measured`: `(804, 91)`; the source intentionally selects the first
  644 states for identification.
- Validation `Z_measured`: `(22, 10, 91)`; `known_pressed_mask`: `(22, 19)`.
- `data/Load_label/` contains the 14 original acquisition files from which the
  804-state archive was extracted; all 14 hashes match the extraction audit.
- `data/Multifrequency/` contains the 22 original multifrequency files; all 22 hashes
  match `data/validation/V1_manifest.csv`.
- Complex impedance is in ohms and frequency is in hertz.
- Pair order is the upper triangle of clockwise node order
  `[1,2,3,4,5,6,7,14,13,12,11,10,9,8]`; node 14 is grounded. The validation
  archive normalizes each displayed label to ascending node IDs, while preserving
  the same unordered-pair index order.
- Each selected parameter start contains 91 `(Gp, Cp)` path rows and 19 unloaded
  `(G, B)` rows.

## Metric integrity and expected values

Loaded active-element parameters are fitted on one parity subset. The complementary
45 or 46 pairs are held out. Metrics compare measured impedance with the complete
parasitic-aware prediction, not compensated measurements with intrinsic-only response.

Expected selected fixed-parasitics means are complex-Z L2 `0.07248932352514488`, complex-Y L2
`0.11221025025086662`, magnitude MAPE `0.051012205181862974`, and phase MAE
`1.4585956612044686°`. Expected recalibrated no-parasitics values are complex-Z L2
`0.30721563956543607`, magnitude MAPE `0.21694665987847198`, and phase MAE
`1.4613390296959967°`.

Final Figs. 2–6 and source tables are packaged. Figure 1 remains absent pending
redistribution approval. These checks verify saved artifacts and aggregation; they are
not a fresh optimization or independent replication.

### Terminology

The scientific method is described publicly as forward-model-based identification and
compensation with fixed path-dependent parasitic admittances. The selected parameter set
uses a weak logarithmic prior on unloaded S2 and is referred to as the selected
weak-S2-prior fixed-parasitics parameterization. `E2` appears only in preserved original
filenames, result fields, and run metadata as an internal historical identifier.
