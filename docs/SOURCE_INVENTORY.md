# Source inventory and dependency trace

Exact hashes and packaged destinations are in `docs/provenance.csv`.

| Purpose | Original source | Packaged location |
|---|---|---|
| Grounded model and 644-state identification | `Simulation_Experiment/experiment_I5_checkpointed_convergence.py` | `src/scientific/` |
| Weak-S2-prior identification | `Simulation_Experiment/experiment_I5_S2_E1_E2_E3.py` | `src/scientific/` |
| Historical measurements | `Simulation_Experiment/real91_extract/run_20261103T145251_758871Z/` | `data/identification/` |
| Original historical acquisition files | `Data/Load_label/*.txt` | `data/Load_label/` |
| Multifrequency extraction | `Simulation_Experiment/V1_extract_multifrequency_91pairs_v1_1.py` | `src/scientific/` |
| Independent measurements | `Simulation_Experiment/V1_multifrequency_audit/run_20261105T113619_277534Z/` | `data/validation/` |
| Original independent sweep files | `Data/MultiFrequency_Measurements/2026_11_05/*.txt` | `data/Multifrequency/` |
| Shared validation model | `V2_2_two_offset_hypotheses.py` | `src/scientific/` |
| Calibration and loaded fitting | `V2_4_fixed_parasitic_intrinsic_drift.py` | `src/scientific/` |
| 22-sweep held-out evaluation | `V2_6_validate_22_sweeps_fixed_parasitics.py` | `src/scientific/` |
| Publication aggregation | `V2_7_publication_validation_v1_1.py`; `validate_E2_locality_multifrequency.py` | `src/scientific/` |
| Final parameters | internal source path `experiment_I5_S2_E1_E2_E3/E2_weak_S2_prior/init_{05,30,70}pct/` | `parameters/selected_weak_s2_prior/` |
| Final validation | preserved internal validation runs | `results/validation/` |
| No-parasitics control | run `20261110T163257_799182Z` | `results/no_parasitics/` |
| Figs. 2–4 | run `Chapter4_E2_Figures_Updated/run_20261109T135226_686207Z` | `figures/publication/` |
| Figs. 5–6 | run `E2_loaded_recovery_figures_corrected/run_20261110T151759_451311Z` | `figures/publication/` |

## Exclusions and uncertainties

- Figure 1 is adapted from reference [10]; redistribution permission is unknown.
- The 14 raw historical TXT files and 22 raw multifrequency TXT files are now packaged
  and match their extraction-audit/manifest hashes. Public release still requires review.
- Checkpoints and optimization histories were omitted. Final CSV/JSON parameters are
  sufficient for validation; warm-start continuation requires original checkpoints.
- Original scripts retain absolute paths for byte-level auditability. A fully portable
  end-to-end optimization driver is not claimed without rerunning the expensive fit.
- The 630 evaluations share 21 physical loaded sweeps.
- Names containing `E2` in this inventory are preserved internal run/file identifiers,
  not publication-facing method names.
