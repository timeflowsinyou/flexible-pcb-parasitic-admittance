# Data layout and provenance

- `identification/`: unchanged audited archive containing 804 states at 100 kHz; the identification source selects the first 644 states.
- `validation/`: unchanged audited archive of 22 sweeps, 91 terminal pairs and 10 frequencies, with loading labels.
- `Load_label/`: 14 unchanged original TXT acquisition files containing the 804
  historical states; the manuscript identification uses the first 644 states.
- `Multifrequency/`: 22 unchanged original TXT sweeps (one unloaded and 21 loaded),
  each containing all 91 pairs at 10 frequencies.
- `metadata/`: exact mapping from pair index to terminal node pair, original sensing-element numbering and activation labels, acquisition details.
All packaged research inputs, including the newly added raw TXT files, are hashed in
`docs/provenance.csv`. Experimental files
require explicit approval before public release. Do not fabricate, alter, interpolate,
or silently repair measurements.
