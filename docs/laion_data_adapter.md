# LAION data adapter and pilot acquisition

`LaionNiftiAdapter` implements the shared `DatasetAdapter.responses` boundary for
local T1w 4D beta NIfTI files, trial TSVs and a supplied binary 3D mask. Install
`.[data]`. Inputs remain read-only; no resampling, OTC construction, reliability
thresholding, normalization or implicit trial subsampling takes place.

## Verified dataset contract

The [official beta documentation](https://laion-fmri.hebartlab.com/glmsingle_betas.html)
was checked on 2026-10-26. The final TYPED NIfTI has a trial axis; the TSV uses
`session`, `run`, zero-based `beta_index` and `label`. Label is the exact stimulus
filename, not an invented sequential ID. The downloaded pilot TSV for sub-01,
ses-01 contains 1044 rows and 769 distinct labels, with session value `ses-01`.
The adapter joins by an explicit index policy even when TSV rows are shuffled.
It rejects missing/duplicate indices within that policy and inconsistent sessions.
The downloaded TSV contradicts the website: beta_index resets to 0 in each run.
The pilot therefore requires `run_index` and explicit acquisition run order
01–12; its summary lists 87 trials in each run. The pinned
[official loader](https://github.com/ViCCo-Group/LAION-fMRI/blob/cf82765ced029de81c3bec7254b8cbd6a65e3686/laion_fmri/subject.py)
reads TSV rows without sorting and creates session_trial from row position.
For this pilot, that order agrees with run offsets plus within-run beta_index.
We preserve original beta_index and expose a separate resolved volume_index;
no automatic detection or fallback changes the mapping. See D20.

The public beta sidecar describes units as percent signal change, whereas the
website warns that TYPED shrinkage complicates that interpretation. The adapter
performs no unit conversion and makes no claim to resolve this discrepancy.

## Explicit local manifest

Put a manifest beneath configured `data_root`; its file bindings are relative to
that root. The following is a schema example, not an approved scientific ROI:

```json
{
  "schema_version": 1,
  "subject_id": "sub-01",
  "roi_id": "pilot-mask",
  "space": "T1w",
  "mask": "masks/pilot-mask.nii.gz",
  "sessions": [
    {"session_id": "ses-01", "betas": "session/betas.nii.gz", "trials": "session/trials.tsv",
     "index_mode": "run_index", "run_order": ["01", "02"]}
  ],
  "aggregation": "mean_all_listed_trials_v1",
  "normalization": "none",
  "provenance": {
    "dataset_release": "record the inspected release",
    "roi_definition": "record mask source and exact selection policy",
    "reliability": "record the policy; no implicit NCSNR filter",
    "upstream_normalization": "record upstream processing",
    "session_coverage": "one pilot session, not complete subject coverage"
  }
}
```

```python
from revision_fmri import LaionNiftiAdapter
adapter = LaionNiftiAdapter.from_config(manifest="manifests/pilot.json")
responses = adapter.responses("sub-01", "pilot-mask", requested_image_ids)
# Pass responses to the existing ReproductionPipeline with aligned features/split.
```

The run_order example must be replaced with the complete acquisition order.
For session-global beta_index use `session_index` and an empty run_order.
`adapter.trials()` returns validated metadata without reading beta values.
`configs/laion_pilot.example.json` supplies the observed pilot mapping and
a clearly named I/O-only mask; copy it under data_root before using it.

The explicit aggregation option averages every occurrence of each requested
label across the listed sessions, accumulating in float64. It preserves caller
image order and reports each image's repeat count. This is a named adaptation
(D08/D16), not a claim to reproduce original NSD normalization. No default
scientific LAION protocol has been selected. The image-level EvaluationSplit
keeps all those repeats together. Trial-level betas are not exposed as separate
images, and unknown requested labels fail rather than silently disappearing.

Mask and beta shapes/affines must agree (absolute affine tolerance 1e-5, no
relative tolerance). Mask values must be binary; nonfinite selected betas fail.
Voxel IDs encode T1w array coordinates; subject and grid hashes disambiguate
native spaces. Metadata contains source-file, manifest, grid and mask hashes,
plus upstream processing and ROI declarations. Caller provenance must be free
of local paths. The adapter checks supplied geometry and declared identities;
it cannot prove a mislabeled file belongs to the claimed participant.

Only one 3D volume is decoded at a time. Compressed NIfTI access may be slow, and
whole-file SHA-256 checks add sequential I/O. This is the initial fidelity-oriented
reader, not a validated high-throughput implementation. Header/TSV validation
and small fixtures do not establish full real-data parity. Surface/HDF5 inputs,
original session normalization, OTC construction and NCSNR recovery remain pending.

## Acquisition

[Official access instructions](https://laion-fmri.hebartlab.com/data_access.html)
allow anonymous public S3 access to fMRI files; raw stimulus images require the
user's Data Use Agreement. No form is submitted on the user's behalf.

```bash
# Inventory only by default; paths always come from local config.
python scripts/download_laion_pilot.py --config configs/paths.local.yaml
# Small public pilot metadata:
python scripts/download_laion_pilot.py --config configs/paths.local.yaml --download
# Explicitly authorized single 1,155,267,333-byte beta NIfTI:
python scripts/download_laion_pilot.py --config configs/paths.local.yaml --download --include-betas
```

This narrow downloader preserves S3 keys beneath configured data_root, uses
conditional ETag requests, checks byte length and records SHA-256 receipts.
It never overwrites existing files; reruns verify recorded local hashes. ETags
are object identity tokens, not assumed MD5 checksums. Interrupted `.part` files
are retained for inspection and are not silently deleted or resumed. It does
not fetch duplicate HDF5 betas, raw BOLD, checkpoints or gated stimulus images.

The user authorized data downloads and local long-task execution on 2026-10-26.
The previous ssh-jz instruction is superseded. GPU experiments remain separately
authorized work. All actual paths/hardware scheduling settings stay local.

The optional `--include-mask` download flag obtains the small laion-general
T1w res-1pt8 mask solely for I/O checks. With the manifest in data_root, run:

```bash
python scripts/validate_laion_pilot.py --config configs/paths.local.yaml --manifest manifests/pilot_io_run_index_v1.json
```

The script reads three singly presented images through the shared adapter;
it performs no feature inference, fitting or ROI reliability selection.
