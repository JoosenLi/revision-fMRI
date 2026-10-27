# Team decisions: ridge alpha grid and OTC ROI definition

Status: **open for collaborator decision** · Opened 2026-10-27 · Owner: Qiaoxin LI (@JoosenLi)

The priority of re:vision is to reproduce Conwell et al. (Nature Communications,
2024) faithfully. AGENTS.md requires all three collaborators to approve changes to
original hyperparameters, ROI/reliability rules, repetition handling, splits and
precision policy. This document collects the decisions that are still open and
the owner decisions that still need the other two sign-offs.

**How to respond:** fill in your column in the [answer sheet](#answer-sheet) at the
end, either by committing to this branch or by commenting on the pull request.
For each question, pick an option letter (or write "other" plus a short note).
The reviewer's recommendation is marked ★. Supporting evidence is cited so that
each item can be checked independently.

Evidence labels used below: **P** = paper text, **C** = released DeepNSD code
(`ColinConwell/DeepNSD@36f6a34`), **L** = LAION-fMRI documentation/data.

---

## Part A — Owner decisions awaiting sign-off

These are already implemented (PRs #10 and #11) but need the other two
collaborators' approval. Details: `docs/primary_laion_protocol.md`,
`docs/matched3_responses.md`, `docs/replication_differences.md`.

| ID | Topic | Original (P/C) | Decision for the LAION primary analysis |
|---|---|---|---|
| A1 | Train/test split | C: even/odd positions of the stimulus list, 500/500 on NSD shared1000 | Same rule on the 1,121 non-OOD shared LAION images in the pinned official order → 561/560. OOD images excluded. Official LAION splits (tau/cluster/random/ood) become secondary analyses only. |
| A2 | SRP dimensions | C: `johnson_lindenstrauss_min_dim(n_images, eps=0.1)` → 5,920 for 1,000 images | Same formula on all 1,121 images → **6,018** |
| A3 | Repeats | P/C: 3 repeats per NSD shared image | Exactly 3 per image, taken from **3 different sessions** where possible (earliest presentation in each of the 3 earliest sessions; filled from the earliest unused presentation only if the image appears in fewer than 3 sessions). All-repeat averaging is deferred. |
| A4 | Session normalization | C: per-session, per-voxel z-score over **all** trials, then average | Identical (all valid trials in each LAION session). This is a disclosed, primary-only exception to the "no test responses in fitting" rule: test-image betas contribute only to per-voxel mean/SD, never to labels, ridge fitting, alpha or layer selection. |
| A5 | Layer selection | C: maximum training score; the released R script also first filters layers by a test-minus-train criterion (C7) | Maximum training score only; the C7 test-dependent filter is excluded and disclosed. |
| A6 | Deferred (generalization award) | — | NSD bridge analysis on the 240 NSD images in the LAION shared pool; all-repeat (12/4) averaging. |

---

## Part B — Q1: Ridge alpha grid

**What alpha is.** veRSA first predicts every voxel from (projected) model
features with ridge regression. Alpha is the regularization strength: small
values fit the training data closely, large values shrink predictions. The
original method chooses alpha **separately for each voxel** from a candidate
list, using leave-one-out cross-validation within the training images. The
candidate list therefore affects the scores.

**The discrepancy.**

| Source | Candidate alphas |
|---|---|
| P: Methods text | range 10⁻¹ to 10⁷ (number of steps not stated) |
| C: `main_analysis.py` (the code path that produced the results) | `np.logspace(-1, 5, 7)` = 0.1, 1, 10, 100, 1,000, 10⁴, 10⁵ |

The two disagree only at the upper end (10⁵ vs 10⁷). This matters only for
voxels whose best alpha lies at the upper bound of the released grid. We can
measure this on real data: the share of voxels whose selected alpha is 10⁵.

**Options**
- **A ★** Primary analysis uses the released code grid (7 values, 10⁻¹…10⁵),
  because that grid generated the published scores. Report the paper/code
  discrepancy, and report the share of voxels at the 10⁵ boundary in the first
  real run. If that share is substantial (proposal: >10 % of voxels in any
  subject/ROI), add a sensitivity analysis with an extended grid under a
  separate protocol ID.
- **B** Primary analysis uses an extended grid up to 10⁷ (step count must also
  be decided, e.g. 9 values `logspace(-1, 7, 9)`), and the released grid becomes
  the sensitivity analysis.
- **C** Run both as co-primary analyses.

---

## Part C — OTC region of interest

### C.1 What the paper uses (P)

The main results use occipitotemporal cortex (OTC):

```
OTC = reliable ∩ ( (nsdgeneral ∩ {midventral, ventral, midlateral, lateral streams})
                   ∪ category_union )
category_union = OFA, FFA-1, FFA-2, EBA, FBA-1, FBA-2, PPA, OPA, OWFA, VWFA-1, VWFA-2   (RSC excluded)
category voxels: localizer t > 1        reliable: NCSNR > 0.2
```

Published voxel counts (C: `publication/source_data/voxel_metadata_OTC.csv`):
subject 1 = 8,088; subject 2 = 7,528; subject 5 = 8,015; subject 7 = 5,849
(total 29,480; lowest saved NCSNR = 0.20001).

### C.2 Paper vs released code (both NSD)

| Detail | Paper (P) | Released code (C: `nsd_parser.py`) |
|---|---|---|
| Reliability threshold | NCSNR **>** 0.2 | removes values **<** 0.2, so exactly 0.2 is kept |
| Category threshold | t **>** 1 | removes values < 1, so exactly 1 is kept |
| Early visual cortex | not in OTC | the default union function also includes early visual labels |

The published voxel list matches the paper's counts, so the paper's OTC is the
intended target. The code's default function must not be called and labelled
"OTC".

### C.3 NSD vs LAION (the main difficulty)

| Component | NSD (paper) | Available in LAION (L: rois.html, glmsingle_betas.html) | Gap |
|---|---|---|---|
| General visual area | `nsdgeneral` | `laion-general` (GLMsingle R² visual selectivity) | Different definition |
| Streams | functional streams: midventral, ventral, midlateral, lateral | sectors `laion-EVC`, `laion-dorsal`, `laion-lateral`, `laion-ventral` (retinotopic ROIs + anatomical borders; subdivisions of `laion-general`) | Anatomical, not functional; no separate "mid" zones |
| Face / scene / word areas | OFA, FFA-1/2, PPA, OPA, VWFA-1/2 at t > 1 | same names, delineated at a **liberal t > 0** (manual) | Larger regions; contrast maps "will be released" but are not available yet |
| Body areas | FBA-1, FBA-2 | single `FBA` | Merged |
| OWFA | present | **no equivalent** (`mfs-words` is a different region) | Missing |
| RSC (excluded) | RSC | `MPA` (LAION's RSC) | Exclude MPA |
| Extra LAION areas | — | pSTS-faces, SPL, mfs-words, l-/v-objects, MT/MST | Not in the original set |
| Reliability | NCSNR > 0.2 | **noise ceiling in percent (NC%)**, NSD variance-based estimator (Allen et al., 2022), per-session and cross-session maps (4-rep, 12-rep, all-rep) | NC% ≠ NCSNR; same threshold cannot be reused |

Note: for the NSD estimator, NC% = 100 · NCSNR² / (NCSNR² + 1/n), where n is
the number of repeats averaged. NCSNR can therefore be recovered from NC% if n
and the map's scope are known exactly. LAION z-scores betas within each run
before estimating, whereas NSD z-scored within session.

### C.4 Questions

**Q2 — Stream component.**
- **A ★** Use `laion-lateral ∪ laion-ventral` as the adaptation of NSD's four
  lateral/ventral streams. Since both are subdivisions of `laion-general`, the
  intersection with `laion-general` is implied. Exclude `laion-EVC` and
  `laion-dorsal`. Disclose that NSD's intermediate "mid" streams have no
  separate LAION counterpart and may partly fall in `laion-EVC`.
- **B** As A, and also add the part of `laion-EVC` beyond V1–V3 to approximate
  NSD's mid streams (requires defining that boundary).
- **C** Other.

**Q3 — Category threshold (paper t > 1; LAION masks t > 0).**
- **A ★** Final primary analysis re-thresholds at t > 1 once LAION releases the
  underlying contrast maps. Until then, pilot runs use the t > 0 masks, labelled
  `interim_t0` and never reported as primary results.
- **B** Adopt the LAION t > 0 masks as the permanent primary adaptation and
  disclose that categories are larger than in the paper.
- **C** Request the contrast maps from the LAION team now and do not run
  anything with categories until then.

**Q4 — Missing OWFA.**
- **A ★** Omit OWFA and disclose it. Do not substitute `mfs-words`.
- **B** Substitute `mfs-words` as the closest available character-selective
  region (labelled as a substitution).

**Q5 — Category set composition.**
- **A ★** Use OFA, FFA-1, FFA-2, EBA, FBA (merged), PPA, OPA, VWFA-1, VWFA-2.
  Exclude MPA (= RSC) and all extra LAION regions (pSTS-faces, SPL, mfs-words,
  objects, motion).
- **B** Other (specify).

**Q6 — Reliability measure.**
- **A ★** Recompute NCSNR ourselves with the NSD estimator from the betas used
  in the primary analysis (session-z-scored as in A4, images with ≥ 3 repeats,
  using the same 3-repeat selection as A3), then apply 0.2. Use the
  NC%→NCSNR conversion of LAION's maps (option B) as a cross-check. Disclose
  that reliability is estimated from data that include test images, as in NSD.
- **B** Convert LAION's cross-session NC% map (12-rep or 4-rep, with known n)
  to NCSNR using the formula above, then apply 0.2.
- **C** Use LAION NC% directly with a threshold chosen to match NCSNR 0.2 for a
  stated n (equivalent to B, but expressed in NC units).

**Q7 — Threshold boundary.**
- **A ★** Use the paper's strict `>` for NCSNR (and for t, if Q3-A).
- **B** Use the released code's `>=`.

(The practical difference is limited to voxels exactly on the threshold.)

**Q8 — Sanity check against published voxel counts.**
- **A ★** Report LAION OTC voxel counts per subject next to the paper's counts
  (5,849–8,088) as a diagnostic only. **Never tune thresholds to match them.**
  Flag for review if a subject differs by more than a factor of 2.
- **B** No count comparison.

---

## Answer sheet

Members: **M1** = Qiaoxin LI (owner) · **M2** = _name_ · **M3** = _name_.
Write an option letter, "approve" / "object", or "other: …". Add comments below
the table.

| Item | M1 | M2 | M3 | Final decision |
|---|---|---|---|---|
| A1 Split (561/560 even/odd) | approve |  |  |  |
| A2 SRP dims (6,018) | approve |  |  |  |
| A3 Matched-3, distinct sessions | approve |  |  |  |
| A4 C6 legacy session z-score (primary-only exception) | approve |  |  |  |
| A5 Training-only selection; C7 excluded | approve |  |  |  |
| A6 Deferred items | approve |  |  |  |
| Q1 Alpha grid |  |  |  |  |
| Q2 Stream component |  |  |  |  |
| Q3 Category threshold |  |  |  |  |
| Q4 OWFA |  |  |  |  |
| Q5 Category set |  |  |  |  |
| Q6 Reliability measure |  |  |  |  |
| Q7 Threshold boundary |  |  |  |  |
| Q8 Voxel-count sanity check |  |  |  |  |

### Comments

- M1:
- M2:
- M3:

Once all three columns are filled, the owner records the final decisions in
`docs/deviations.md` (D01, D05, D07, D11) and `docs/replication_differences.md`,
and implementation proceeds.
