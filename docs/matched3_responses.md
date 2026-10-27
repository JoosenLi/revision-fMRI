# Matched-three response API

Owner decisions F10/F11 (2026-10-27) supersede the earlier first-chronological
rule and resolve N1: primary replication requires `matched3_distinct_sessions_v1`
and `legacy_deepnsd_session_v1`. Remaining collaborator sign-offs are pending.
This is an adaptation of NSD's three-repeat design, not recovery of NSD trial
identities, temporal spacing, or historical averaged-beta exports.

## Acquisition ordering (B1)

The real pilot TSV contains only `session, run, beta_index, label`. Its matching
beta JSON sidecar states: “4D NIfTI with one volume per trial, ordered by acquisition
time across runs.” Therefore `laion_volume_acquisition_order_v1` orders trials by
**(numeric BIDS session number, volume_index)**, not by fabricated onset times.
Run-local beta_index plus the explicitly supplied run concatenation order maps
to volume_index; the adapter validates complete one-to-one volume coverage.
For a release with verified session-global indices the existing session_index
mapping is also accepted. Neither mapping nor run order is inferred from row order.

The NIfTI manifest must explicitly name `beta_sidecar` alongside each session's
betas/trials/index_mode/run_order for volume ordering. The sidecar must contain
the acquisition-order statement; it is hashed alongside the beta and TSV files.
Missing/contradictory ordering evidence fails. Numeric run labels are retained
as provenance; the volume index controls ordering across runs. BIDS events.tsv
onsets are an optional future cross-check, not implemented or required here.

Metadata-only verification on the actual pilot: 1044 rows, 12 runs x 87 trials,
volume indices 0–1043. TSV SHA-256:
`77c7e3c1360272bec1384b019e87306949acd777adf9a8093d93e1a9b931df73`.
Beta sidecar SHA-256:
`6fc426ca03ca2bc63ed46598fb1bd1e13e20d300d0948bca8e305fe808dcf47f`.
No beta values were read in that check; the pilot alone does not cover the full pool.

The earlier `tsv_onset_v1` and `glmsingle_beta_index_order_v1` ordering sources
remain available for the superseded first-chronological secondary policy. The
latter is an assumption about run-local ranks, distinct from sidecar-backed
volume ordering. New distinct-session selection requires volume ordering.

## Shared implementation and F10 selection

`aggregate_matched3` takes `TrialSession` objects with a trial-by-voxel matrix and
explicit `TrialPresentation` metadata. Volume ordering requires null onset and
an integer volume_index; mixed sources and duplicate acquisition keys fail.
Input iteration order does not determine chronology. Validity is explicit,
never inferred from beta magnitude. A nonfinite valid beta fails; it is not
silently marked invalid. An entirely invalid session is audited and contributes
no selected responses.

Operation order:

1. Normalize **all valid rows** within each session, before any repeat selection.
2. For each requested image, group valid trials by session. Take the earliest
   presentation from each of the three earliest sessions containing that image.
3. If fewer than three sessions are available, fill the remaining slots with the
   earliest unused presentations. Record `filled_within_session=true`; do not
   discard such images. Examples: 2+2 or 3+1 across sessions yield three trials
   drawn from two sessions; three trials in one session are also permitted.
4. Fewer than three valid presentations in total fails without dropping images,
   averaging fewer repeats, or writing an output artifact.
5. Sort selected trials chronologically and average their float32 normalized
   vectors with float64 accumulation. No RNG.

Only three selected voxel vectors per image are retained between sessions;
complete eligible trial metadata is retained for verification. The reader holds
one session's ROI matrix at a time. Real host memory still needs budgeting.

The old `matched3_first_chronological_v1` remains an explicitly named secondary
policy; the primary split loader and evaluator reject it. `repeat_policy` has no
default, and the adapter requires agreement with the manifest's aggregation tag.

## F11: primary C6 normalization and narrow exception

Primary requires `legacy_deepnsd_session_v1`: per-voxel z-score with ddof=0 over
**all valid trials in a session**, including shared, subject-unique and OOD
images, then cast to float32 before selecting/averaging repeats. Set
`training_image_ids=None`. This preserves the released C6 normalization operation
and its order; it does not prove the historical averaged-beta export provenance.

Test-image betas contribute to unsupervised session mean/std. This is an explicit
exception to independent preprocessing, consistent with GLMsingle already fitting
sessions jointly. No test labels/scores or test responses enter ridge fitting,
alpha choice or layer selection. C7 filtering remains excluded. Consequently,
perturbing raw test trials may change session statistics/training responses;
the selector's independence claim applies to already-normalized inputs.

`evaluate_primary_laion_model` passes the explicit flag
`normalization_exception='c6_legacy_session_zscore'` through the shared evaluator.
It validates the exact canonical primary split, source/order digest, 561/560
counts, matched-three manifest and legacy normalization before accepting it.
`evaluate_model`, `evaluate_cached_model`, and `evaluate_layers` without the
flag continue rejecting legacy-normalized responses. The flag cannot authorize
an arbitrary split, old repeat policy or train-only normalization. Reports retain
the normalization policy and exception flag; no provenance tag is hidden/renamed.

`train_only_session_v1` is available only as a separately named adaptation and is
rejected by primary evaluation. `exclude_test_session_v1` is a deferred sensitivity
analysis and remains unimplemented. No default normalization is added to the API.

## Adapter call and durable provenance

`LaionNiftiAdapter.responses_matched3` requires explicit `repeat_policy`,
`normalization_policy`, `training_image_ids`, `ordering_source`, `validity_policy`.
For primary, select the policies above. Validity must be declared as either
`all_presentations_valid_v1` (an explicit declaration, not inferred QC) or
`tsv_valid_column_v1` (requires every row's Boolean flag). An explicit false flag
contradicts an all-valid declaration and fails. Valid-trial exclusions are audited.

Schema-v3 selection JSON is content-addressed under configured
`results_root/trial_selections/<sha256>.json`. Existing identical content is verified;
differing content is never overwritten. Earlier schemas are not reinterpreted.
Artifacts contain selected/eligible trial IDs, source, volume/run/beta indices,
nullable onset, normalization/input/source hashes, response binding, per-image
session spans and `filled_within_session`. Summary counts report three-distinct vs
filled images overall and per LAION / THINGS+ / NSD group, using exact published
label prefixes (other labels are explicitly `other`). The count of images with
repeated selected sessions is also recorded. Primary reports include this summary
and the manifest hash/reference, alongside the unchanged JL projection provenance.

Tests cover 3+ distinct sessions, 2-session and 1-session fill, invalid earliest
trials, shortages, permutation invariance, source-group counts, manifest tampering,
real four-column TSV format, sidecar checks/hashes, normalization of non-pool
trials, primary acceptance and generic-path rejection. No complete real-data
aggregation, GPU experiment, checkpoint download or scientific score was run.
