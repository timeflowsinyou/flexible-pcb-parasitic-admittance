# Forward-Model-Based Estimation and Compensation of Distributed Parasitic Admittances in a Flexible PCB Sensor Matrix

Reproducibility repository for a research manuscript on a flexible PCB sensor matrix containing **14 terminal nodes and 19 sensing elements**. The model predicts **91 terminal-pair complex impedances** using a physics-based intrinsic network model and fixed, path-dependent equivalent parasitic admittances.

> **Repository status:** Initial project scaffold only. No experimental data, parameter files, or full analysis pipeline have been added yet. Do not treat this repository as a complete reproducibility package until the checklist below is complete.

## Study datasets

- **Identification:** 644 experimental loading states, each with 91 complex terminal-pair measurements at 100 kHz. These support joint identification of intrinsic sensing-element parameters and shared parasitic-admittance parameters.
- **Independent validation:** 22 complete measurement sweeps, each covering 91 terminal pairs at 10 frequencies (1 kHz to 1 MHz). These include one unloaded reference and 21 loaded measurements.
- **Primary performance reporting:** Five evaluation frequencies: approximately 46.4, 100, 215.4, 464.2 kHz and 1 MHz. Validation uses complementary 45/46 pair partitions for loaded-state parameter fitting and held-out prediction.

## Scope of the code

The eventual release will distinguish:

1. Shared equivalent parasitic-admittance identification from historical measurements.
2. Frequency-specific intrinsic unloaded-state recalibration with fixed parasitic parameters.
3. Loaded-state fitting and held-out prediction using the full measurement forward model.
4. Parasitic-free control experiments and the figures/tables reported in the manuscript.

**Important:** The effective parasitic model is not a direct measurement or unique physical decomposition of every parasitic component. The model's 100 kHz identification and its independent multifrequency predictive validation are different stages.

## Repository layout

```text
data/
  identification/    # Original, immutable historical 100 kHz measurements
  validation/        # Original, immutable independent multifrequency sweeps
  metadata/          # Terminal-pair order, sensing-element map, loading labels
  processed/         # Reproducibly derived data only
src/                 # Reproducible parameter estimation and validation code
scripts/             # Data audits and pipeline entry points
parameters/          # Identified parameters, provenance, numerical precision
results/             # Aggregate numerical outputs and audit manifests
figures/             # Source generation scripts and publication figures
docs/                # Reproducibility, changelog and method notes
```

## Current data availability

Experimental data supporting the findings of the manuscript are available from the corresponding author upon reasonable request, subject to permission to share. This initial repository does **not** contain the datasets yet. A public release plan and archive/DOI can be added after agreement with the coauthors.

## Reproducibility requirements before publication

- [ ] Inventory the 644 historical identification states and validate 91-pair ordering.
- [ ] Inventory the 22 independent validation sweeps and validate the 10 frequencies.
- [ ] Preserve original inputs and record SHA-256 hashes.
- [ ] Add all final source scripts and their dependencies, excluding unrelated experiments.
- [ ] Add final fixed equivalent parasitic parameter files, with provenance.
- [ ] Provide executable commands to reproduce the manuscript's key metrics and Figs. 2–6.
- [ ] Confirm permission to distribute all experimental data, figures and code.
- [ ] Choose an appropriate software license in consultation with the coauthors.
- [ ] Record the release commit and, if desired, create a permanent Zenodo archive.

## Provenance and audit

Source measurements must not be overwritten by processing scripts. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The included standalone [`scripts/create_sha256_manifest.py`](scripts/create_sha256_manifest.py) creates a timestamped audit of files under `data/` and can verify an earlier manifest.
