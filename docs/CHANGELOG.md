# Research reproducibility changelog

- **Initial scaffold** — Created project structure, provenance guidance and SHA-256 audit utility. No scientific datasets or paper numerical outputs modified or included.
- **Reproducibility package (2026-11-12)** — Added byte-identical paper-specific
  scientific sources, audited processed measurement archives, final selected
  weak-S2-prior fixed-parasitics parameters,
  held-out and no-parasitics result tables, Figs. 2–6, SHA-256 provenance, portable
  integrity checks, and release-review documentation. No optimization was run.
- **Raw-data and figure-review update (2026-11-13)** — Documented 14 original
  `Load_label` acquisition files (804 states, first 644 used) and 22 original
  multifrequency sweeps; added them to provenance/integrity checks. Recorded the project
  owner's confirmation that all publication figures display normally.
- **Publication terminology update (2026-11-13)** — Removed the internal `E2` run label
  from public method descriptions, adopted publication terminology, and renamed the
  packaged parameter directory to `selected_weak_s2_prior`. Preserved internal source
  filenames and result fields remain unchanged for provenance.

All subsequent transformations should document source filenames and hashes, script version, parameters, output filenames and hashes, and any intentional corrections. Keep raw data unchanged.
