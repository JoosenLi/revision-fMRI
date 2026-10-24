# Original study: evidence baseline

Audit date: 2026-10-24. This is an audit and API foundation, not a completed numerical reproduction.

## Sources and scope

Conwell, Prince, Kay, Alvarez & Konkle (2024), *A large-scale examination of inductive biases shaping high-level visual representation in brains and machines*, Nature Communications 15, 9383. [Version of record](https://doi.org/10.1038/s41467-024-53147-y); [full text mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC11526138/); [supplementary PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-53147-y/MediaObjects/41467_2024_53147_MOESM1_ESM.pdf).

The full Methods and supplementary text were inspected. SI Table 1 spans PDF pages 1–4; SI.2 examines effective dimensionality, SI.3 cross-subject rankings, SI.4 later-layer depth (page 6), and SI.5 early visual cortex. SI Table 1 provides model names, architecture, task, diet and scores, **not checkpoint hashes or an environment lock**. Its eRSA label corresponds to the paper's veRSA. Display labels can be inconsistent (e.g. two CLIP ViT-B/32 labels with different architectures); they are not unique checkpoint identifiers.

[DeepNSD](https://github.com/ColinConwell/DeepNSD) was inspected at `36f6a34727dccb5eec8fa49364b210a049a388d7` (2026-09-24). Eight key methods files were also fetched at the post-publication 2024 release commit `5efc1921078032b255532c3d30b527633ecd8b9f` (2024-11-02): main analysis, ridge modification, result parser, feature extraction/reduction, model options, custom options, and NSD parser. All eight are byte-identical. This establishes released-code continuity, not proof of the exact environment used for published results. Local read-only source snapshots are ignored under `.audit_sources/`; tracked file hashes and source references are under `docs/provenance/`. Later DeepJuice functionality is outside this baseline.

## Verified paper behavior

- 224 networks: 160 trained and 64 randomly initialized. Controlled comparisons concern architecture, objective and visual diet; available pretrained networks are compared, not retrained.
- NSD subjects 01, 02, 05 and 07; 1,000 shared COCO images with three presentations per subject. GLMsingle estimates are averaged across presentations.
- OTC includes the intersection of the general visual mask and mid/high ventral/lateral sectors, augmented with eleven category-selective ROIs. Paper thresholds: NCSNR > 0.2 and category contrast t > 1. Counts: 8,088 / 7,528 / 8,015 / 5,849; total 29,480.
- Computational submodules are features, flattened without spatial pooling. SI.4 explicitly restricts consideration to the later half of the model, explaining the 0.5–1.0 depth axis.
- Two disjoint 500-image sets: choose a layer on training images and report that layer on test images, independently per subject/linking method.
- cRSA uses Pearson image-pattern similarity and Pearson correspondence of model and brain geometries.
- veRSA uses sparse random projection (JL epsilon 0.1, 1,000 images, 5,920 dimensions even for smaller original feature spaces), voxelwise ridge encoding, and Pearson correspondence of predicted and observed population geometries. It is not mean voxelwise correlation.
- Paper ridge range: 10^-1 through 10^7; best alpha is chosen separately per voxel by Pearson correlation of LOOCV predictions. The precise grid length is not specified in the prose.
- Main comparison models include subject as a fixed effect; ImageNet-size comparisons add a model random intercept. Inference is conditional on these four subjects, not a population-level subject random-effects analysis.
- Within-subject RSM noise ceilings use the GSN generative procedure on independent data. Voxel NCSNR and RSM noise ceiling are different quantities.

## Controlled comparisons

| Comparison | Paper/released-result membership | Controlled dimensions |
|---|---|---|
| Architecture | 34 CNNs vs 21 transformers; 3 hybrids and 6 MLP mixers recorded separately | ImageNet-1K classification |
| Taskonomy | 24 trained encoders; random encoder recorded separately | ResNet50 encoder and Taskonomy indoor imagery |
| Self-supervision | 10 ResNet50 models, 6 contrastive / 4 non-contrastive | Architecture and nominal ImageNet diet |
| Language alignment | ViT S/B/L × SimCLR/CLIP/SLIP, nine 25-epoch checkpoints | YFCC15M and architecture within size |
| ImageNet size | 12 model pairs, 24 rows | Matched architecture; exact training/fine-tuning provenance still matters |
| IPCL diet | Four AlexNet-GN checkpoints: ImageNet, OpenImages, Places, VGGFace2 | Nominal architecture and IPCL objective |
| Initialization | 64 trained/random pairs in comparison metadata | Architecture; original random seeds not recovered |
| Expertise (supplementary metadata) | 11 BiT expert models | Explicit expert labels, not a substitute for the main sets |

Membership counts above were checked against the released `results_summary.csv` (224 veRSA rows), not inferred from all available loaders. `model_contrasts.csv` has 226 rows; model availability is not evidence of publication inclusion. See the complete [model manifest](model_manifest.md).

## What this does not establish

The paper, release and loaders do not completely identify historical library versions, random initial states, final candidate-layer lists or every downloaded checkpoint byte. Exact agreement with published numerical scores has not been tested. The audit exposes these gaps in [methodology_audit.md](methodology_audit.md), rather than silently filling them with current defaults.
