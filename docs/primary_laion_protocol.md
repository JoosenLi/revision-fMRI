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

## API and repeat-policy boundary

```python
split = load_primary_laion_split(
    config, dataset_release=reviewed_release, repeat_policy=reviewed_repeat_policy)
report = pipeline.evaluate_primary_laion_model(
    features=features, responses=responses,
    dataset_release=reviewed_release, repeat_policy=reviewed_repeat_policy)
```

Both repeat-policy arguments are mandatory and have **no default**. The policy
is a nonempty caller-supplied aggregation identifier, not a new automatic
aggregation algorithm. The evaluation facade requires it to match
`responses.provenance['aggregation']` and requires the response dataset release
to match. It preserves the response repeat-count hash in the report. The split
provenance digest also records the declared policy so incompatible preparations
are not described as the same run specification.

Choosing all available repeats versus a matched-three policy remains open (D08).
The existing data reader implements only explicitly requested
`mean_all_listed_trials_v1`; that name means all occurrences in the listed
sessions, not verified complete 12/4 acquisition coverage. Matched-three trial
selection/aggregation is not implemented here. A matching policy label does not
prove that external averaging was correct: the eventual trial-selection manifest,
actual counts, completeness and any sampling seed must be audited separately.
This PR neither selects trials nor silently assigns a repeat policy.

SRP dimensions (D10) remain explicit through `DeepNSDProtocol(n_components=...,
projection_policy=...)`; no default 5920 or recomputed budget is chosen for 1121.

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
policy checks automatically.

Tests cover 561/560 order, OOD exclusion, missing/reordered/source-mutated pools,
mandatory repeat policy, unchanged NSD1000 behavior and primary facade delegation.
Perturbing only test-half responses leaves training scores and selected layers
unchanged while changing reported test scores. Numerical tests use synthetic
responses/features and an explicitly small test-only SRP budget. No scientific
result, GPU run or complete real-data evaluation is claimed.
