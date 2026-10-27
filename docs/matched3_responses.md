# Matched-three response API

Owner decision D08/F9 (2026-10-26) selects `matched3_first_chronological_v1` for
primary replication. Both remaining collaborators must still sign off before
merge. This is a LAION adaptation of NSD's three-repeat design, not recovery of
NSD trial identities or temporal spacing.

## Shared implementation

`aggregate_matched3` accepts an iterable of `TrialSession` objects, each with a
trial-by-voxel matrix and explicit `TrialPresentation` metadata. Acquisition
session/run ranks are integers; the within-run order value is finite, nonnegative
and unique within a run. `TrialPresentation` uses explicit onset metadata by
default; the beta-index source instead sets onset to null and supplies beta_index.
Mixed ordering sources fail. Trial IDs are globally unique. Input/TSV iteration order is not acquisition
order. Invalid chronology, duplicate IDs/ranks, or ambiguous within-run ties fail.

Validity is an explicit Boolean per trial, never inferred from response magnitude.
Invalid rows are excluded from normalization and selection. A nonfinite **valid**
beta fails; it is not silently reclassified invalid. An entirely invalid session
is audited and contributes no response. No data-dependent quality threshold is
introduced.

Operation order is:

1. Validate metadata and order valid session rows by run/within-run order.
2. Call the existing `normalize_session` with an **explicit** policy on all valid
   rows, including repeats that may later be unselected. No image averaging first.
3. Select each requested image's earliest three valid presentations by acquisition
   session, run and the explicitly declared within-run source. At least three are required for **every** image.
4. Average exactly those three float32 normalized vectors with float64 accumulation,
   preserving the existing data reader's accumulation precision. No RNG.

`train_only_session_v1` requires explicit training-image IDs and fits only those
valid rows. Every session with valid trials must have enough training rows and
nonconstant training voxels; otherwise it fails. `legacy_deepnsd_session_v1` uses
all valid session rows as in the released operation order, and requires
`training_image_ids=None`. The independent evaluator still rejects this legacy
normalization tag (D21). Neither policy has a default. F9 did not decide this
separate normalization-scope question or authorize test-dependent preprocessing.

The implementation retains one normalized session matrix and at most three
voxel vectors per image across sessions, plus trial metadata. It does not allocate
GPU tensors or load an entire dataset's beta arrays simultaneously. Real ROI
matrices can still be large; session/host memory must be budgeted before runs.

## Local NIfTI adapter

Set manifest `aggregation` to the matched-three name and `normalization` explicitly
to one of the supported session policies. Call `responses_matched3` with required
`repeat_policy`, `normalization_policy`, `training_image_ids`, `validity_policy`,
`ordering_source` (no ordering default).
These must agree with the manifest; ordinary `responses()` rejects a matched-three
manifest to prevent accidentally running the old all-repeat path.

The adapter reuses existing geometry, beta-volume/TSV joins and source hashing.
Numeric BIDS session/run labels (`01`, `ses-01`, `run-01`) define acquisition ranks;
non-numeric labels fail rather than guessing chronology. Choose a source explicitly:

- `tsv_onset_v1`: requires a finite, nonnegative within-run `onset` column.
- `glmsingle_beta_index_order_v1`: requires `index_mode="run_index"`; uses the
  validated per-run `beta_index` as chronological rank, independently of TSV row
  order and the manifest's volume-concatenation order. This is an **assumption**,
  not verified acquisition timing. `onset` remains null, not a fictitious time.
  Verify run/trial-label alignment and rank against BIDS events/onsets when they
  become available. An events.tsv join is not implemented in this PR.

The real pilot TSV contains only `session, run, beta_index, label`. The second
policy supports it without manufacturing onset or validity columns. Run-local
indices reset each run (D20); session-global indexing is rejected for this policy.

Choose validity explicitly:

- `tsv_valid_column_v1`: every TSV row needs `valid` = 0/1/false/true.
- `all_presentations_valid_v1`: explicit declaration that all listed rows are valid;
  an existing false validity flag contradicts it and fails.

No validity inference is made from the official pilot metadata. Missing onset
under the onset policy,
validity information under the chosen policy, or fewer than three presentations
fails. Existing all-listed-trial/normalization-none behavior remains available
for separately named secondary analyses; primary entry points reject it.

## Manifest and report provenance

A successful aggregation writes `trial_selections/<content_sha256>.json` beneath
configured `results_root`, with an envelope containing the hash and manifest.
Creation is exclusive; existing identical content is verified and reused, never
overwritten. Artifact references are relative, not institutional paths.

Schema v2 distinguishes `within_run_order`, `ordering_source`, `beta_index` and
nullable `onset` per trial, and embeds source provenance including validity and
ordering declarations. Old schema-v1 artifacts are not silently reinterpreted.
The manifest contains policy, no-RNG seed, image/voxel/subject/ROI/release identity,
valid candidate presentations, the three selected trial IDs with acquisition
metadata, available counts, per-image and overall session spans, session input
and metadata hashes, normalization fit/output hashes, training-ID digest and
averaged-response digest. All requested images remain represented. It also
records invalid-row counts and excluded entirely invalid sessions.

The ResponseSet embeds this manifest/hash/reference and sets `aggregation` to the
policy name with repeat_counts exactly three. Primary evaluation verifies digest,
response identity/binding, selection as the first three valid presentations,
counts and chronology. Train-only normalization must use exactly the primary
training-ID sequence. It persists/verifies the selection artifact and records
its hash/reference/session span plus JL N/eps/components/seed in report provenance.
This is consistency verification, not cryptographic certification of external
trial metadata; original metadata/source hashes must remain available for audit.

First-three presentations may cluster in early LAION sessions; session spans make
that difference reportable. All-repeat averaging (12/4) and the NSD-image bridge
are deferred to the generalization-award backlog after faithful reproduction.

Tests use synthetic trial matrices and tiny local NIfTI/TSV fixtures only. No full
1121-image matched-three real-data coverage or scientific evaluation has been run.

## N1 decision remains pending

No normalization is designated primary. C6's all-session normalization, the
existing train-only alternative, and the reviewer's proposed exclude-test
alternative are choices for the owner/collaborators. `exclude_test_session_v1`
is not implemented. The independent evaluator still rejects the legacy tag;
its acceptance of train-only is an interface constraint, **not** a primary
scientific decision. Primary scientific execution remains blocked on N1.

B1 validation includes a shuffled, four-column pilot-format metadata test with
per-run index resets (no beta file) and a tiny synthetic NIfTI integration test.
A read-only check of all 1044 rows of the actual pilot TSV also passes explicit
beta-index conversion. Its SHA-256 is
`77c7e3c1360272bec1384b019e87306949acd777adf9a8093d93e1a9b931df73`.
This verifies schema/index handling, not chronology against independent events.
