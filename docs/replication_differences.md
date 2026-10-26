# re:vision replication differences — living final-report ledger

Last updated: 2026-10-26. This is the maintained report-facing account of what
re:vision preserves, changes and cannot yet establish relative to Conwell et al.
(Nature Communications, 2024). It is not evidence that the full experiment has
run. Update this ledger with every methodological PR; retain decision history
rather than rewriting unresolved differences as verified fidelity.

Evidence labels: **P** paper/supplement; **C** verified released code;
**A** LAION or engineering adaptation; **U** unresolved. Primary references are
[the original-paper summary](original_paper.md), [methodology audit](methodology_audit.md),
[deviation register](deviations.md), [model manifest](model_manifest.md), and
[ROI mapping](roi_mapping.md). Audit C1–C9 and deviation D01–D22 identifiers below
provide the detailed source pointers; a released-code observation is not proof
of the authors' historical run configuration.

## Differences and implications for the final report

| Topic | Original evidence | re:vision implementation / decision | Scientific implication and remaining action |
| --- | --- | --- | --- |
| Dataset / stimulus pool | P/C: NSD shared1000 analysis | A: primary pool is 1121 non-OOD LAION shared images; OOD excluded | Different image distribution, participants and acquisition; do not describe this as identical-data replication. Record subject/session/release hashes. |
| Primary split (D09, C1) | C: CSV-order even positions train, odd positions test; 500/500 | A: preserve alternation on the pinned published LAION list; 561/560, protocol `laion_splithalf_even_odd_v1` | Order is adapted from NSD CSV to official LAION list. Not a random split or exact equal halves. Owner decision recorded; all-collaborator review remains required. |
| Secondary splits (D09) | C: official LAION tau/cluster/random/OOD are different protocols | A: retained as explicitly secondary adaptation/extension analyses | Do not combine these scores with primary split-half results or present them as the original replication split. Fold aggregation remains unresolved. |
| Repeated responses (D08/F9) | P/C: original NSD shared images have 3 repetitions | A: owner decided matched-3 on 2026-10-26; implement `matched3_first_chronological_v1` after explicit session normalization, with no RNG | First three valid acquisition-ordered trials may cluster in early sessions unlike NSD. Save eligible/selected trials and session spans; fail below three. Other collaborator sign-offs pending. All-repeat analysis deferred to generalization-award backlog. |
| Response normalization (D16/D21, C6) | C: released session z-scoring can use all trials; exact exported-beta history U | A: explicit normalization primitives; independent evaluator rejects legacy all-session provenance | Report normalization scope. Train-only alternative differs from released session preprocessing; do not claim historical export equivalence. |
| ROI and reliability (D05/D07/D11) | P/C: OTC region definitions; paper strict thresholds vs inclusive parser, generic EVC inclusion discrepancy | A/U: explicit ROI primitives exist; LAION anatomical mapping/reliability policy still requires review | LAION NC percentages are not NCSNR. No silent threshold conversion or complete OTC-equivalence claim. |
| Checkpoints and dependencies (D06) | C: many mutable loader defaults; historical bytes and versions U | A: local SHA-verified offline loader, currently AlexNet compatibility profile only | Matching a caller SHA does not verify the paper checkpoint. Historical weights/environment still unresolved; no weights downloaded by this implementation. |
| Image preprocessing (C9/D04/D22) | C: torchvision direct square 224 resize + ImageNet normalization; C2 explicitly converts RGB | C/A: exact source transform reused under pinned modern compatibility versions; shared reader must convert RGB explicitly | Historical interpolation/library behavior U; do not substitute modern weight-enum resize/crop. Other families and random-model normalization differences remain unresolved. |
| Hook and candidate procedure (C2/D15) | C: execution-order hooks, non-inplace ReLU, random probe/hash dedup, later half of deduplicated list, full-dataset second dedup | C/A: merged PR #9 provides single-batch source reuse; full candidate procedure remains unimplemented | Published layer metadata can count pre-dedup outputs. No scientific candidate list is approved until probe/dedup/later-half fidelity is validated. |
| cRSA (C5) | P/C: Pearson correlation of upper-triangle correlation-distance RDMs on raw features | C: original routine reused behind validated common API | No SRP, rank correlation or crossnobis substitution for cRSA. Nonfinite/degenerate inputs now fail explicitly (D13). |
| SRP and veRSA (C3/D10/F7) | C: sparse random projection before split; original JL formula eps=0.1, seed=0, auto density, upsampling; 1000 -> 5920 | C/A: owner decided full-pool recomputation on 2026-10-26; primary `jl_recomputed_eps0.1` fixes 1121 -> 6018 | Primary evaluator enforces this policy and records N/epsilon/dimensions/seed. It adapts sample count while retaining the original formula. Other collaborator sign-offs pending. |
| Ridge alpha grid (D01/C1/C4) | P: range extends to 10^7; released C1 default is seven log-spaced values 10^-1 through 10^5 | C: released-code grid is canonical backend behavior | Paper/code discrepancy remains unresolved. Historical run configuration needed; alternatives must have separate protocol IDs. |
| Ridge scoring / predictions (D03/D19) | C: per-voxel Pearson LOOCV alpha choice, centered training LOOCV predictions; test intercept restored; L2 normalization observed in compatibility environment | C: vendored original backend preserves these semantics | Do not replace with modern RidgeCV MSE defaults, in-sample predictions or restored training offsets. Compatibility parity is not historical-environment proof. |
| Layer selection (D02/C7) | C: maximize training RSA, but released R filter can first use test-minus-train scores; historical effect U | A: primary evaluation uses training-half scores only; selected test-half score reported; C7 filter excluded | Intentional deviation from released test-dependent filtering to maintain independent evaluation. Must be disclosed, not described as exact reproduction of every released analysis step. |
| Precision (D18) | C: original ridge path uses float64 | C/A: float32 vision features; float64 reference ridge; TF32/autocast rejected | Float32 ridge requires separate explicit configuration, deviation and parity evidence; not enabled by preference alone. |
| Statistical comparisons (C8/D12) | P/C: source-specific fixed/mixed formulas; SLIP interaction differs from paper pseudocode | U: specifications audited; numerical statistical backends contract-only | No final inference, CI or p-value replication claim until source formulas/reference levels/bootstrap details are validated. |
| Operational safeguards (D13/D14) | C: downloads, implicit device behavior and limited validation in source | A: local ignored YAML, hashes, frozen/eval models, explicit CUDA, validation and provenance | Engineering changes are distinct from scientific alternatives; they do not establish checkpoint, candidate or dataset fidelity. |

## Canonical primary pool evidence

Source: official `laion_fmri/splits/data/shared/ood.json`, variant 0 `train_ids`,
at `cf82765ced029de81c3bec7254b8cbd6a65e3686`.
Source SHA-256: `d592f700f2ce32c9733800ba524da3b4168af3e53365b25e70aaa993769df842`.
Ordered-ID SHA-256: `8299304097406a108d94db054763edea759ccfe9d1d4446a7ba6e3dd50006db9`.
See [primary protocol](primary_laion_protocol.md) for digest serialization/API.
No sorting is performed at runtime.

Verified label composition: 641 LAION-cluster (12 reps), 240 THINGS+ (12 reps),
140 NSD special1000 (4 reps), 100 NSD special100 (4 reps). There are 641 distinct
LAION cluster identifiers, so no same-cluster siblings in this list; this is not
a complete image-level near-duplicate analysis. Labels describe scheduled repeats,
not verified local trial completeness. The 240 NSD-image bridge is **deferred until after faithful reproduction**, as
a candidate for the **generalization-award stage**. Exact overlap with original
NSD shared1000 still needs ID mapping; no bridge analysis has been run.

## Decision and validation history

- 2026-10-26: owner explicitly chooses original-style even/odd split-half as
  primary; official LAION splits become secondary (PR #9 F4 update). F4a–F4d
  implemented in the follow-up branch from main. Other collaborator sign-offs
  remain pending. At initial F4 implementation repetition and SRP budget were
  open; subsequent owner decisions below supersede that status.
- 2026-10-26, owner D10/F7: recompute original JL formula on full split N=1121,
  eps=0.1 -> 6018; seed 0/auto density/upsampling preserved. Implemented/enforced;
  remaining collaborators must sign off before merge.
- 2026-10-26, owner D08/F9 and explicit owner instruction: primary uses
  `matched3_first_chronological_v1`. Normalize valid session rows, then average
  earliest three valid presentations, with session-span/selection manifests.
  This deterministic LAION rule is an adaptation; no original NSD trial identities
  or temporal spread are reproduced. Float32 normalized responses are accumulated
  in float64, matching the existing reader's mean-accumulation policy. Invalid
  trials require explicit metadata; nonfinite valid betas fail. Other sign-offs
  remain pending. Normalization scope (D21) remains an explicit separate choice.
- 2026-10-26, owner F8/F9d: NSD bridge and all-repeat averaging (12/4) are deferred
  until after faithful reproduction, as generalization-award backlog items.
- Canonical source file independently fetched and matched against local pinned
  metadata. Runtime source/order checks yield 561/560. No beta or image download
  was needed for this change.
- Synthetic tests verify unchanged NSD1000 splitting and primary training-only
  layer selection under test-response perturbation. No real-data effect size,
  model ranking, final statistical result or full reproduction is available.

For the final report, attach the merged protocol revision, approval history,
actual data/ROI/split/checkpoint/transform/layer hashes, trial-selection manifest,
software versions, precision and seeds to each result. Do not promote predictions
or open reviewer recommendations to observed scientific results.
