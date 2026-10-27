# Methodology audit

Labels: **P** = verified 2024 paper/supplement; **C** = verified released source; **A** = required/proposed LAION adaptation; **U** = unresolved. Source links below pin the audited release. No model loader was executed and no scientific experiment was run.

## Source index

Base: [DeepNSD commit](https://github.com/ColinConwell/DeepNSD/tree/36f6a34727dccb5eec8fa49364b210a049a388d7).

| ID | File and relevant symbols |
|---|---|
| C1 | [main_analysis.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/main_analysis.py): `get_splithalf_xy`, `run_benchmarking`, `get_results_max`, CLI |
| C2 | [feature_extraction.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/model_opts/feature_extraction.py): `get_module_name`, `get_feature_maps`, `remove_duplicate_feature_maps`, `get_empty_feature_maps`, `get_all_feature_maps` |
| C3 | [feature_reduction.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/model_opts/feature_reduction.py): `compute_jl_lemma`, `get_feature_map_srps`, `srp_extraction` |
| C4 | [ridge_gcv_mod.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/ridge_gcv_mod.py): `_RidgeGCVMod.fit`, `_BaseRidgeCVMod.fit` |
| C5 | [mapping_methods.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/model_opts/mapping_methods.py): `compare_rdms`, `max_transform` |
| C6 | [nsd_parser.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/brain_data/nsd_parser.py): `load_NSD_brain_data`, `load_NSD_benchmark_ROI_metadata`; [benchmark.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/brain_data/benchmark.py) |
| C7 | [parse_results.R](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/parse_results.R): `remove_degenerate_layers`, `process_result_files`, `get_model_layer_max` |
| C8 | [all_results.R](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/publication/all_results.R), `figure_gen.R`, `environment/R-Packages.csv` |
| C9 | [model_options.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/model_opts/model_options.py), `model_code/_options.py`, `model_code/ipcl_codebase/__init__.py` |

## 1. Checkpoints

**P:** model identities and comparison structure appear in SI Table 1. **C:** see [model_manifest.md](model_manifest.md) and `manifests/models.json`: every one of the 224 published models is cross-referenced with comparison membership, raw source metadata, loader reference and checkpoint resolution status. Explicit URLs are preserved, including VISSL training variants, SLIP epochs, IPCL files and SEER variants. Torchvision `pretrained=True`, timm untagged model names and hub `main` are not immutable checkpoint specifications.

**U:** no claim of byte-verified checkpoints is made. Upstream CLIP/Taskonomy URL tables were inspected separately, but their historical package revisions were not pinned by DeepNSD. Recover original environment and state hashes before authorizing extraction. **A:** use local checkpoint paths plus content hashes; never auto-download or resolve `DEFAULT` weights.

## 2. Evaluation transforms

**P:** model evaluation transforms; no input normalization for random models. **C (C9):**

| Loader path | Actual preprocessing |
|---|---|
| Torchvision classification | PIL `Resize((224,224))`, `ToTensor`, normalize mean (0.485,0.456,0.406), std (0.229,0.224,0.225) |
| Torchvision random | Same resize/tensor, no normalization |
| timm | `resolve_data_config({}, model=model_name)` then `create_transform`; replace resize size with center-crop size and remove center crop. No random-model normalization exception in this branch |
| `train_type == imagenet` dispatcher shortcut | Uses torchvision ImageNet transform, even before source-specific dispatch |
| VISSL / DINO | Resize directly to 224×224; tensor; ImageNet normalization |
| Taskonomy | Resize 256×256; tensor × 2 − 1; encoder only |
| OpenAI CLIP | Return `clip.load(...)[1]`; upstream transform: bicubic short-side resize to model input resolution, center crop, RGB, tensor, CLIP mean (0.48145466,0.4578275,0.40821073) and std (0.26862954,0.26130258,0.27577711) |
| SLIP / SEER / BiT experts | Resize 224×224, RGB conversion, tensor, ImageNet normalization |
| IPCL | Returned transform: short-side Resize(224), CenterCrop(224), tensor, checkpoint config normalization. ImageNet `ipcl1`: ImageNet mean/std; `ipcl7/8/9`: mean (0.5,0.5,0.5), std (0.2,0.2,0.2). Do not use `model.val_transform` (Resize 256) |
| MiDaS | Hub DPT transform for DPT models; small transform otherwise; numpy conversion and squeeze |
| YOLO | PIL bicubic resize 256×256, RGB channel slicing/gray replication, CHW tensor /255 |
| Detectron | Reverse RGB to BGR, ResizeShortestEdge([224,224],256) by default, model `preprocess_image` normalization/padding; backbone only |

**U:** timm's string-valued `model=` resolution is library-version dependent; normalization for random timm models conflicts with P. Interpolation/antialias/Pillow defaults must be captured from the historical environment. Dispatcher calls MiDaS/YOLO with `input_type` in the first positional argument, although their functions also expect a model name. Optional custom-model registration checks `custom_models`, while custom code is shipped under `model_code`; packaging must be reconciled. No universal modern transform is approved. Some transform getters themselves load/download a model: never invoke them just to inspect transforms in this audit phase.

## 3. Feature hooks and candidate layers

**P:** all distinct computational submodules, flatten all non-image axes; SI.4 later-half selection. **C (C1/C2):** hooks cover every module except `nn.Sequential` and `nn.ModuleList`, including non-container parent modules. Observed forward execution order defines feature order. Names are class names with occurrence counts, not `named_modules()` paths; counting uses substring matches against previously inserted names. Reused modules may yield multiple occurrences. Tuple/list tensor outputs gain one-based suffixes; mapping outputs are not recursively expanded. Outputs become detached CPU float32. A non-leading batch axis is moved only if uniquely identifiable; scalars/ambiguous axes are skipped. ReLU is changed to non-inplace before extraction, model is set to eval, and forward runs under `no_grad`.

Deduplication uses random dot-product hash bytes (not tensor identity), retaining the first hash occurrence. Initial enumeration uses three random 3×224×224 inputs, then CLI retains `names[floor(len(names)/2):]`. DataLoader extraction preallocates using three real images, runs batches of 64 without duplicate removal, then deduplicates again over complete feature arrays. Returned arrays flatten all non-batch dimensions. No global pooling, CLS-only extraction, PCA or attention-block-only selection is equivalent.

**U:** random dummy inputs/hash vectors are not seeded at these call sites; input resolution and shape heuristics may change which layers survive. Actual candidate lists and original RNG states are unavailable. **A:** persist full enumerated order, rejected outputs, dedup matches, selected candidate list, model/transform hashes and seed state. A deterministic replacement would be a documented deviation, not the legacy algorithm.

## 4. OTC, response estimates and reliability

**P:** see [roi_mapping.md](roi_mapping.md). **C (C6):** parser removes NCSNR **<** 0.2 and category t **<** 1, retaining equality (unlike P's strict inequalities). Default ROI group list also includes `early-visual`; that branch skips reliability filtering and enters the final OR union. The generic parser alone is therefore not an exact OTC-only specification. Saved OTC metadata resolves the published target: 29,480 rows with paper-matching per-subject counts, minimum NCSNR 0.20001102983951569. Some parser branches and exported ROI memberships require reconciliation before rebuilding masks from raw NSD.

Volume beta parser z-scores within session along trial axis before concatenation. Benchmark consumes precomputed voxel-by-image CSVs; responses are not re-standardized by `get_splithalf_xy`. Exact end-to-end provenance of the exported averaged betas remains U, including session normalization and ordering. Benchmark forms subject/ROI RDMs only for >10 voxels and drops RDMs with NaNs. Do not confuse this cleaning with a new reliability threshold.

## 5. cRSA

For image-by-feature matrix F, `D(F) = 1 - corrcoef(F)` correlates rows across features. Brain matrix B is images × voxels within a subject/ROI. Score is `Pearson(vec_upper(D(F)), vec_upper(D(B)))`, strictly above diagonal (`k=1`), excluding duplicate entries and diagonal. Using similarities instead of distances for both sides gives the same Pearson score. At 500 images, there are 124,750 entries. **C:** original (unprojected, unstandardized) features feed cRSA; no rank correlation default or crossnobis. NaN handling in original `compare_rdms` attempts a warning with a malformed second argument; new API raises an explicit error, recorded in deviations.

## 6. SRP and veRSA

**C (C3):** `johnson_lindenstrauss_min_dim(n_samples, eps=.1)`; `SparseRandomProjection(n_components, random_state=0)`; density defaults to `auto`, signed sparse entries follow sklearn's distribution. `upsampling=True`, including feature spaces smaller than projection count. Integer seed is reused on each fit; equal-width layers therefore reuse the same random mapping under the same sklearn implementation. SRP is performed before image splitting, its matrix depending on shape/seed rather than response values. Fit `StandardScaler` on projected even-index training rows only, transform odd-index test rows with it.

**C (C1):** predict each voxel, restrict predictions to subject/ROI, build Pearson-distance RDM from image-by-predicted-voxel patterns and compare its upper triangle with the actual brain RDM. Source metric name `wrsa` = veRSA; `srpr` is mean voxelwise Pearson prediction and is a different metric. **A:** keep both raw cRSA features and projected encoding features. Number of projections for a changed LAION image count needs explicit policy; 5,920 fixed vs JL recomputation are distinct protocols.

## 7. Ridge grid, objective and training predictions

**P:** range 10^-1–10^7. **C:** both CLI and function default use `np.logspace(-1,5,7)`, i.e. [0.1,1,10,100,1000,10000,100000]. No evidence identifies which generated the published scores; expose separate specifications, with the paper grid length unresolved.

C4 subclasses private sklearn ridge APIs. Data enter as float64; intercept fitting is enabled. **Correction after executable verification on 2026-10-26:** the initial audit inferred that internal feature normalization was disabled from `_normalize=False` and commented arguments. In sklearn 1.3.2, `_preprocess_data(X, y, fit_intercept, normalize=False, copy=True, ...)` receives `self.copy_X` as the fourth positional argument, so the default `copy_X=True` actually enables feature L2 normalization. This is verified released-code behavior in a compatible environment, not proof of the authors' historical environment. The reference now requires an explicit normalization policy; see D19 and [encoding_reference.md](encoding_reference.md).

For each alpha, C4 computes `p = y_centered - c / diag(G_inverse)`. Each voxel selects alpha maximizing Pearson correlation across its training-image LOOCV predictions. Strict `>` updates retain the first alpha in ties. These values are stored in `cv_values_` with shape images × voxels × alphas; they are predictions, not squared residuals. No y offset is restored into stored values. C1 selects each voxel's alpha slice directly as the **training prediction matrix**. Test predictions use `X_test @ coef_.T + intercept_`, which restores response offsets.

This centered-LOOCV versus full-scale-test distinction can affect across-voxel RDMs; voxelwise Pearson's offset invariance does not remove the across-voxel issue. Do not replace training predictions with `predict(X_train)` or silently add y offsets. Outer training scaling is fitted once, not refitted within each LOOCV fold. A port needs golden comparisons with a compatible historical sklearn build, including nonzero voxel means. Current sklearn `RidgeCV` defaults (MSE, alpha sharing, changed cv storage semantics) are not substitutes.

## 8. Layer selection and held-out status

**C:** stimulus CSV order determines even indices (`::2`) training and odd indices (`1::2`) test. Preserve ordered stimulus IDs, not just a seed. Max training RSA selects a layer separately by model/training type, subject, region and metric; selected test score is then reported. Python and R paths keep the first retained tied maximum; the ordering must be persisted. Subjects are averaged after their own selection.

**Critical U/C7:** `process_result_files` calls `remove_degenerate_layers` *before* selection. It computes `scale(test - train)` and retains values > −3.291. Thus candidate eligibility can depend on held-out scores. This is verified in the released script, but we cannot prove how many published rows it removed without raw pre-filter results. Paper's claimed independence and this processing path must not be conflated. A separate `superlative_layers.csv` export chooses layers by mean test scores for auxiliary analyses; it is not the primary train-selected procedure. The primary package selector intentionally accepts training scores only; reproducing C7 filtering will require an explicitly named legacy path.

## 9. Statistics and metadata

**P:** raw Pearson RSA as outcome; additive subject fixed effects. Architecture uses CNN reference; Taskonomy uses denoising; contrastive comparison uses non-contrastive; language uses SimCLR; diet uses ImageNet; ImageNet size includes model random intercept. Fixed-effect Wald t and mixed-model Wald z approximations are described. Figure intervals are 95% subject-normalized bootstrap intervals. Do not assume Fisher-z-transformed outcomes, pairwise t tests, subjects-as-random-intercepts or multiple-comparison correction.

**C (C8):** `lm(score ~ contrast + subj_id)` for architecture and contrastive SSL; Taskonomy and IPCL use corresponding comparison columns. **SLIP actually fits `compare_goal_slip * architecture + subj_id`**, with references SimCLR and ViT-B, and also compares with subject-only models via ANOVA. This is richer than the paper's additive pseudocode. ImageNet-size: `lmer(score ~ compare_diet_imagenetsize + subj_id + (1|model))`. Trained/random: `lmer(score ~ compare_training + subj_id + (compare_training|model))`. Plot centering is score − subject mean + grand mean within comparison/metric; `mean_cl_boot` supplies intervals. Bootstrap exact RNG state and all inference-package defaults are U; the published R package-version table is available. No blanket p-value correction was found in these comparison blocks; preserve their actual formulas and report any correction as an addition.

Published metadata includes architecture, task/diet, source, display name, comparison indicators, layer depth, feature/parameter counts and top-1 accuracy where available. Copy metadata verbatim alongside normalized fields; do not use accuracy as a ranking target or equate 21K/22K naming, pretraining-only and fine-tuned checkpoints. Historical inconsistencies include IPCL Places256 label versus `places2` file; ImageNet-size ResMLP `_in22ft1k`; SLIP `ViT-L-*-CC12M` keys pointing to base checkpoints with metadata claiming YFCC15M. These CC12M models are not in the nine-model controlled SLIP set.

An additional classification discrepancy: 64 rows have `train_task=random`, but `random_weights_taskonomy` is labeled `train_task=taskonomy`. Thus there are 65 loader-level random-state cases in the 224-row inventory, despite the paper's 160-trained/64-random summary. The source's task-label-based trained filter can include that random encoder. Preserve raw task and actual initialization status separately; do not silently reclassify published group analyses.

## Adaptation and implementation boundary

A: versioned LAION image splits, repeated-trial aggregation, explicit ROI mapping/reliability definition, and collaborator-local paths. U: resolve the deviations before scientific runs. The canonical numerical backend now directly executes vendored C4 and verbatim RSA/SRP function excerpts, with pinned compatibility checks. The independent SVD implementation remains a test oracle. Model execution, LAION I/O, ROI reconstruction and statistical backends remain pending. See [upstream_reuse.md](upstream_reuse.md), [encoding_reference.md](encoding_reference.md) and [implementation_plan.md](implementation_plan.md). No claim of a functioning full reproduction pipeline or historical-environment recovery is made.

## LAION I/O evidence update — 2026-10-26

The pilot acquisition confirms 1044 TYPED volumes and 769 unique stimulus labels for sub-01/ses-01. The TSV index resets per run, contrary to the online session-global description; see D20 and `laion_data_adapter.md`. The local reader uses an explicitly selected index policy, checks geometry and preserves repeats. It does not resolve original beta normalization, OTC/NCSNR mapping or the layer-filter discrepancy.

## G1–G3 audit update — 2026-10-27

C6 >10-voxel subject/ROI eligibility is now enforced by shared evaluation with an
explicit error rather than silent omission. D19 is historically grounded by
official sklearn signatures across 0.24.2–1.3.2, not only the pinned execution
profile. The 1.4.0 call is incompatible. See [regression and source evidence](encoding_fidelity_regressions.md).
