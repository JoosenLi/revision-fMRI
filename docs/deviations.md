# Deviations and unresolved decisions

Status as of 2026-10-24. No scientific run is authorized or launched. A proposal is not an implemented methodology choice. Each future run must record its protocol ID and resolved decisions below.

| ID | Evidence / conflict | Current handling | Resolution / required test |
|---|---|---|---|
| D01 | Paper ridge range ends 10^7; released default ends 10^5 | Unresolved; code-grid constant is explicitly labeled; paper grid has no invented step count | Recover authors' run configuration; compare both only under separate protocol IDs |
| D02 | Code filters layers by standardized test−train difference before maximizing training score | Unresolved historical usage; new selector uses training scores only (intentional exclusion of test-dependent filtering) | Reconcile raw/exported results; separate legacy and independent evaluation paths |
| D03 | Stored LOOCV predictions are centered; test predictions restore intercept | Preserve as an audit requirement, not silently correct | Golden test with nonzero voxel means against legacy RidgeCVMod |
| D04 | Paper removes normalization for random models; timm branch retains it | Unresolved | Recover historical transform and persist exact transform spec per model |
| D05 | NCSNR/t strict paper thresholds vs inclusive parser boundaries; generic parser includes EVC | Published OTC metadata is reference artifact, not generic parser defaults | Match original mask counts and exact voxel IDs; tests at threshold equality |
| D06 | Model/library defaults, RNG states and checkpoint hashes incompletely specified | All models remain unapproved for extraction | Resolve local bytes, dependency versions, init seeds and candidate-layer fingerprints |
| D07 | LAION anatomical visual sectors are not NSD streams; some category ROIs differ | Proposed `laion_otc_adapted_v1`, not original OTC | Explicit masks and reviewed mapping; missing ROIs are errors, not silent omissions |
| D08 | LAION repeats and image counts differ from NSD shared1000×3 | Trial-aware data contract; no automatic subsampling/aggregation chosen | Persist image IDs, repeated-image grouping and counts; choose all-repeat vs matched-three protocol explicitly |
| D09 | re:vision tau/cluster/OOD splits differ from original even/odd half split | Required separate adaptation; proposed shared-pool primary analysis | Pin split release/content; layer/alpha/scaler selection only inside outer training images |
| D10 | Original 5,920 SRP dimensions derives from 1,000 samples | No LAION default selected | Decide fixed-budget versus JL-recomputed policy; report dimensions, epsilon, density, seed |
| D11 | LAION voxel NC (%) and NCSNR are not interchangeable with RSM ceiling | No threshold in percentage units substituted | Recover/recompute NCSNR from correct variance and repeat convention; GSN remains separate |
| D12 | SLIP released statistics include architecture interaction absent in paper pseudocode | Preserve both descriptions; no Python replacement fitted | Compare R design matrices, coefficients, SEs and reference levels |
| D13 | Source malformed NaN warning and implicit image alignment | New metrics/data APIs reject nonfinite/degenerate patterns and mismatched IDs | Implemented validation deviation; test deterministic failures |
| D14 | Upstream auto-downloads and hardcoded paths, implicit CPU fallback | New local-only config/checkpoint guard; no network model backend; no CPU training | Missing local resource fails; future feature backend must require CUDA explicitly |
| D15 | Stochastic hook dedup, custom-loader packaging and model-specific resolution issues | Original implementation documented; execution deferred | Original execution fingerprints before deterministic replacement or bug fixes |
| D16 | LAION response space/session scaling may differ | No implicit beta re-scaling or space conversion | Preserve voxel ID/affine/space and normalization provenance; inspect local metadata before implementation |
| D18 | User prefers float32 when adequate; original ridge uses float64 | YAML separates feature precision (float32) and ridge precision (float64) | Float32 ridge needs conditioning/alpha/prediction/RSA/layer-choice parity tests and joint review |
| D17 | 64 random-task rows plus a random Taskonomy encoder labeled taskonomy | Preserve raw labels and initialization status separately | Reconcile paper trained/random counts and downstream grouping before reclassification |
| D19 | Executing released C4 on sklearn 1.3.2 reveals `copy_X=True` binds to positional `normalize`, enabling feature L2 normalization | Corrected the initial static audit; reference requires explicit `legacy_l2_normalize`; True matches unmodified source in the pinned compatibility environment | Five synthetic source fixtures plus brute-force LOO tests pass; original authors' dependency version/intent remains unresolved; no silent choice for scientific runs |

2026-10-26 implementation update: D03's centered LOO behavior is now numerically
verified against unmodified source; D10 remains an explicit caller-supplied
projection count; D18's float32 ridge option remains unavailable in the numerical
reference and fails explicitly. D01/D02 and all LAION scientific decisions remain
unresolved. See `encoding_reference.md` for scope and validation tolerances.

Changes to engineering (configuration, validation, source hashing) are implemented. Scientific changes listed as proposals or unresolved must not be promoted to original-method claims. No new regularization, feature pooling, PCA, pretrained replacement, network fine-tuning or modern default scoring is introduced.

Source-reuse update (2026-10-26): the canonical backend directly uses upstream code and selects the released C1 alpha grid (D01) and observed original preprocessing (D19), without patching the vendor file. These are explicitly source-backed choices in the compatibility environment, not claims about the original execution environment. D02 remains an explicit difference in the independent evaluation path. See `upstream_reuse.md`.

Data adapter update (2026-10-26): D08/D16 now have an explicit opt-in `mean_all_listed_trials_v1` with no additional normalization. This reads all occurrences in the supplied session list and uses float64 accumulation; it is not selected as the scientific reproduction protocol. No OTC or reliability threshold is inferred. See `laion_data_adapter.md`.

### D20 — LAION pilot beta indices reset per run

The public sub-01/ses-01 TSV has 1044 rows but only 87 distinct beta_index values (0–86), repeated across 12 runs. The website describes session-global indices. The official loader at `cf82765ced029de81c3bec7254b8cbd6a65e3686` preserves TSV row order and constructs a separate session_trial index. The GLMsingle summary lists acquisition sources in run order 01–12 and 87 trials per run. The explicit `run_index` adapter maps volume_index = cumulative preceding run lengths + beta_index; this matches the pilot TSV order. Other releases require their own verified mapping. No silent fallback to row order is allowed.

### D21 — Session normalization scope

C6 z-scores across all trials within a session before concatenation/casting to float32. `legacy_deepnsd_session_v1` preserves that operation; a separately named `train_only_session_v1` fits only caller-specified training trials. Both are explicit primitives, not enabled automatically in the LAION reader. Independent evaluation rejects the legacy provenance tag. This documents response preprocessing scope separately from C7 layer filtering; exact historical averaged-beta export provenance remains unresolved.

### D09 — Official split implementation update

D09 implementation update (2026-10-26): original shared-pool assignment files at
LAION source commit `cf82765ced029de81c3bec7254b8cbd6a65e3686` are hash-locked and
imported unchanged. No primary split is implicitly selected. Exact upstream ID
order is preserved. The one-session pilot is incomplete; coverage diagnostics
do not truncate the official split or establish scientific-run readiness.

### D22 — Offline AlexNet vision compatibility profile

C9 constructor/transform functions are now reused verbatim. The adapter disables
automatic pretrained loading and strictly restores a configured SHA-verified
local state_dict (D14), requires RGB PIL and float32 with TF32/autocast disabled
(D18), and pins torch 2.6.0 / torchvision 0.21.0 / Pillow 12.0.0. This is an
explicit compatibility adaptation, not a recovered historical environment.
Historical weights and resize-library behavior remain unresolved. Synthetic
transform parity and mocked loader lifecycle tests do not validate real AlexNet
weights or CUDA execution. See [offline vision](offline_vision.md).
Requires 3-collaborator review before merging.
