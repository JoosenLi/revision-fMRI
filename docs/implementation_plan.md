# Implementation tasks and unit-test plan

This plan follows the audit; it does not authorize GPU experiments, large downloads, training or fine-tuning. Documentation, interfaces, unified YAML config parsing, local checksum verification, raw cRSA and training-score-only layer selection are implemented. A small-array SRP/float64 ridge/veRSA reference and held-out layer evaluator are now implemented; see [encoding_reference.md](encoding_reference.md). LAION I/O, model execution, hook extraction and statistics remain pending.

## Work packages for three collaborators

Suggested ownership is by component rather than institution-specific paths; collaborators can change ownership without changing the APIs.

| Task | Suggested owner | Deliverable / exit condition | Unit and parity tests |
|---|---|---|---|
| T1 Resolve source discrepancies | Joint methodological review | Decide D01–D06, D12, D15; record protocol IDs and author/source evidence; pin Python/R environments | Source hashes and eight 2024/current equality checks; detect conflicting alpha/transform/specs |
| T2 Local data + ROI adapter | Collaborator A | Read already-local trials/betas/masks; explicit subject, voxel axes, space, session normalization, repeats and split provenance | Permuted/missing IDs; duplicates across splits; variable repeats; no trial leakage; nonfinite data; volume affine/shape and surface indexing mismatch; mask union overlaps; threshold equality; absent OWFA/t maps fail |
| T3 Models + transforms | Collaborator B | Bind logical model IDs to local files; SHA-256 + dependency revision + full transform trace; freeze network, eval, CUDA-only backend | Missing/hash-mismatched checkpoint; no outbound download; frozen parameters; CUDA unavailable fails; rectangular, grayscale and RGBA images; interpolation/crop/normalization fingerprints for every family; CLIP input sizes; IPCL returned vs val transform; random timm discrepancy |
| T4 Hook/candidate extraction | Collaborator B | Reproduce legacy execution naming, output shape handling and stochastic hash dedup; save full candidate order/seed state | Reused modules; parent/child duplicates; tuple outputs; scalars; ambiguous batch axes; inplace ReLU; random dummy input; late-half off-by-one; real-input enumeration vs dummy differences; whole-tensor flattening; batch independence |
| T5 SRP and encoding | Collaborator C | Historical CPU numerical reference for small arrays only, then validated compute backend if later authorized; no vision training | JL(1000,.1)=5920; signed sparse auto-density matrix; seed=0 repeatability; same-width layer reuse; upsampling; outer-train-only scaler; constants; per-voxel different alphas; ties; nonzero means; centered stored LOO predictions; intercept-restored test predictions; exact code grid; brute-force LOO at fixed outer scaler |
| T6 Nested evaluation | Collaborator C with A | Explicit outer IDs and train-only selection; distinct paper/code/LAION profiles; cRSA raw features and veRSA predicted patterns | Test-target perturbation cannot change chosen layer/alpha/scaler; training predictions are not in-sample; select independently per subject/ROI/metric; deterministic tie order; synthetic cRSA/veRSA analytic examples; invalid patterns; exact upper triangle |
| T7 Statistics | Collaborator C, joint review | Keep R reference implementation/environment; export tidy per-subject results; formula/reference-level registry | Golden design matrix and coefficients/SE/CI against R; SLIP interaction and ANOVA; paired ImageNet model random intercept; trained/random random slope; subject-centered bootstrap; no silent correction or Fisher transform |
| T8 Reproducibility handoff | All | One small fixture runnable at all institutions; manifests contain identities/hashes, not roots; same outputs within declared tolerances | Move config/working directory; different local roots same logical IDs; cache invalidation for weights/transform/layer order/image order/SRP/split changes; package install smoke test; imports have no model/network side effects |

## Order and decision gates

1. Finish source/version reconciliation before numerical implementation. Unknown values remain unknown; choosing a value creates a documented protocol decision.
2. Agree the adapted ROI, reliability scope, beta normalization, repeat policy, shared pool and exact split revision. No automatic “equivalent OTC” claim.
3. Establish small numerical fixtures on **is248155**. Current tests are CPU-only array/configuration checks, not training. Future model tests require the two RTX 6000 GPUs and must fail rather than switch training/inference to CPU.
4. Test legacy RidgeCVMod under a compatible isolated sklearn environment, then compare the port against it. Do not use today's private sklearn APIs without a version check. CPU reference linear algebra is not CPU vision-network training.
5. Before later experiments, choose original half-split baseline versus re:vision tau/cluster/OOD adaptations, SRP dimension policy and explicit legacy-filter handling. A modern independent evaluation may be scientifically appropriate but is a deviation from C7, not evidence of source parity.
6. Long tasks run via `ssh jz` only after separately authorized. This phase creates no remote job and no scheduler automation.

## Tests implemented now

YAML configuration portability/environment override, CUDA policy, per-stage float32/float64 settings; missing local resources/checksums; shared manifest membership/comparison counts; image/voxel axis and ID constraints; repeat counts; split disjointness/coverage; raw-feature cRSA against an independent scipy calculation; diagonal/lower-triangle exclusion; reject undefined pattern correlations and projected cRSA; train-only selector ties/nonfinite rejection. See `tests/test_contracts.py`. Ridge/SRP/held-out evaluation now have synthetic source-parity tests (42 total tests passed on 2026-10-26). Hook, ROI adaptation, float32 ridge, GPU and statistical parity tests remain **planned**. D19 was discovered by executing the original source and requires methodological review.

## Precision validation before experiments

Feature inference/cache precision defaults to float32. Original RidgeCVMod converts to float64, so float64 is the fidelity reference for encoding. A float32 ridge alternative must be explicitly configured and tested on both well-conditioned and nearly collinear synthetic features, large alpha ranges, nonzero response means and near-tied alpha/layer scores. Compare selected alpha per voxel, centered LOO predictions, intercept-restored predictions, cRSA/veRSA and final layer choices. Choose numerical tolerances before viewing scientific results. Report mismatches; do not silently switch methods. No GPU memory-saving benefit establishes numerical equivalence. A float64 reference that fits existing float32 feature caches does not require storing all original features in float64.

## Source-reuse implementation update

The canonical encoding path is now `DeepNSDEncodingEvaluator`: original RidgeCVMod file plus verbatim RSA/SRP functions, pinned compatibility environment and packaged license/provenance. The SVD implementation is retained as a test oracle. 53 tests pass; original-code calls, hashes, settings and environment failure paths are covered. See [upstream_reuse.md](upstream_reuse.md). Data/model/hook/statistics adapters remain pending; D01/D02/D19 are documented, not silently reconciled.
