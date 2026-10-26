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
