# Primary LAION split-half protocol

The repository owner's 2026-10-26 decision makes
`laion_splithalf_even_odd_v1` the primary replication split. This implements
PR #9 F4a–F4d. Formal review by all three collaborators is still required before
merge. Official tau/cluster/random/OOD assignments are secondary extension
analyses, not substitutes for the primary split.

## Pinned image order and source

Released DeepNSD C1 uses stimulus CSV order, `ids[::2]` for training and
`ids[1::2]` for testing. The NSD shared1000 entry point remains exactly 500/500
with its existing ID/digest validation and protocol ID. Both datasets now use
the same positional-alternation helper; no independent splitting algorithm is
introduced.

The LAION adaptation takes **exactly** `variants[0].train_ids` from official
`laion_fmri/splits/data/shared/ood.json` at commit
`cf82765ced029de81c3bec7254b8cbd6a65e3686`. This is the published 1121-image
non-OOD universe. Only this ordered universe is reused: the official OOD
train/test assignment is not the primary split. OOD test IDs are excluded.

- Source file SHA-256: `d592f700f2ce32c9733800ba524da3b4168af3e53365b25e70aaa993769df842`.
- Canonical order SHA-256: `8299304097406a108d94db054763edea759ccfe9d1d4446a7ba6e3dd50006db9`.
- Order digest uses `metadata_digest`: JSON list, sorted object keys, compact
  separators, UTF-8, SHA-256. It includes the exact ID order and spelling.
- Train: `ids[::2]`, **561 images**; test: `ids[1::2]`, **560 images**. With an odd
  pool size this is the nearest positional 50/50 split, not exactly equal counts.

The published list happens to be lexicographically sorted; runtime code never
sorts it. Both file identity and ordered-ID digest are checked. Reordering,
source changes or incomplete pools fail, without intersection or fallback.
The existing configured local metadata store is reused; no data download occurs
inside the protocol loader.

The source list was independently fetched and checked against local pinned bytes
on 2026-10-26. Label-derived composition: 641 LAION-cluster + 240 THINGS+ images
with 12 repetitions, and 140 NSD special1000 + 100 NSD special100 images with
4 repetitions. All 641 LAION cluster identifiers are distinct; this verifies no
same-cluster siblings in the list, not a general visual near-duplicate guarantee.
These are scheduled-repeat labels, not proof that local responses contain every
scheduled trial.

## Decided repeat and projection policies

Owner decisions (2026-10-26, PR #9 D10/F7 and PR #10 D08/F9) now specify:

- **Repeats:** `matched3_first_chronological_v1`, first three valid presentations
  per image ordered by session acquisition rank, run rank, then onset. No RNG.
  The repeat-policy argument remains mandatory with no default, but primary
  entry points reject other policy names. All-repeat averaging is deferred.
- **SRP:** `jl_recomputed_eps0.1`, using the original C3 JL formula on the **full
  split universe** before splitting: 1121 -> **6018**, 1000 -> **5920**.
  Epsilon 0.1, random_state 0, density auto and original upsampling are preserved.
  A primary run cannot silently use 5920, an arbitrary count or another policy.

```python
split = load_primary_laion_split(
    config, dataset_release=reviewed_release,
    repeat_policy="matched3_first_chronological_v1")
pipeline = ReproductionPipeline(config, DeepNSDProtocol.jl_recomputed(split.split))
responses = adapter.responses_matched3(
    subject_id, roi_id, canonical_image_ids,
    repeat_policy="matched3_first_chronological_v1",
    normalization_policy=reviewed_normalization_policy,
    training_image_ids=normalization_training_ids,
    validity_policy=reviewed_validity_policy)
report = pipeline.evaluate_primary_laion_model(
    features=features, responses=responses, dataset_release=reviewed_release,
    repeat_policy="matched3_first_chronological_v1")
```

`canonical_image_ids` must match the feature row order and cover the complete
primary pool. For `train_only_session_v1`, `normalization_training_ids` must be
exactly `split.split.train_ids`. Normalization scope is still explicit: this
change does not silently choose the legacy all-session policy or change the
independent evaluator's rejection of legacy all-session normalization (D21).

The shared [matched-three API](matched3_responses.md) normalizes valid session
rows **before** selecting/averaging the three responses. Fewer than three valid
presentations for any requested image fails; missing images never shrink the
1121 pool. The reader persists a content-addressed trial-selection JSON under
configured results storage, including eligible/selected trial IDs, per-image
session spans, normalization hashes and response binding. The primary evaluator
checks policy, exact repeat counts, manifest integrity and normalization fit-ID
provenance, persists/verifies the artifact, and includes its hash plus the
projection specification (N, epsilon, dimensions, seed, density, upsampling) in
the report. Selected trials may cluster in early sessions, unlike NSD's spread
repetitions; this is a documented LAION adaptation, not a hidden equivalence.

Both owner decisions are implemented; formal sign-off from the other two
collaborators remains required. The NSD image bridge and all-repeat averaging
are generalization-award backlog items, after faithful reproduction.

## Held-out evaluation and limits

The primary facade delegates to existing `evaluate_model` / `evaluate_layers`:
cRSA and veRSA choose their own layer using training-half scores only, then
report the selected layer's test-half score. C7's test-dependent eligibility
filter is not included. Complete image coverage is required; OOD/extra images
must be explicitly excluded before evaluation. No partial-session pool shrinking.

The generic cached API and `RecordedSplit` remain available; the new primary
facade evaluates already prepared feature arrays and adds explicit repeat-policy
validation/provenance. A dedicated primary cached convenience facade is not added.
Using lower-level generic evaluators does not apply these additional primary
policy checks automatically. Primary-policy claims must use the primary facade.

Tests cover 561/560 order, OOD exclusion, missing/reordered/source-mutated pools,
mandatory repeat policy, unchanged NSD1000 behavior and primary facade delegation.
Perturbing only test-half responses leaves training scores and selected layers
unchanged while changing reported test scores. Selector tests use synthetic
responses/features and a deterministic test encoder; they do not execute a
6018-column ridge job. Separate tests verify the exact 5920/6018 JL results, and
existing source-parity fixtures cover SRP/ridge numerics. No scientific
result, GPU run or complete real-data evaluation is claimed.
