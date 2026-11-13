# Measurement metadata

`scripts/export_metadata.py` writes the physical edges, ordered terminal pairs, and
network description from the validated scientific constants. The NPZ and CSV manifests
also retain frequencies, conditions, pressed-sensor masks, source filenames, and hashes.

- Ordered terminal-node pairs for every recorded sweep (91 unique unordered pairs).
- Mapping between each of the 19 sensing elements and its nodes.
- Exact 10 excitation frequencies, in measurement order.
- Definition of loading condition labels and which sensing elements were pressed.
- Mapping between identification samples and acquisition files/sessions.
- Acquisition settings, including reference resistance and impedance correction method.

The validation archive uses ascending node IDs in displayed pair labels while retaining
the same unordered-pair index order as identification. Metadata is generated and checked,
not inferred from figures.
