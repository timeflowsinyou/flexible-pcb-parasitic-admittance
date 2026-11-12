# Data layout and provenance

- `identification/`: original 644 experimental states at 100 kHz, 91 terminal pairs per state; source measurements remain unchanged.
- `validation/`: original 22 independent sweeps, 91 terminal pairs and 10 excitation frequencies per sweep; preserve loading labels.
- `metadata/`: exact mapping from pair index to terminal node pair, original sensing-element numbering and activation labels, acquisition details.
- `processed/`: generated artifacts only; document the generator script, configuration and corresponding raw input hashes.

**No experimental files are included in this starter archive.** Do not fabricate, alter, or interpolate missing measurements to fill gaps.
