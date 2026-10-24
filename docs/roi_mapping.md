# ROI and reliability mapping

## Verified original target

Paper Methods, “Voxel selection procedure”: conceptual mask

`OTC = reliability ∩ ((nsdgeneral ∩ {midventral, ventral, midlateral, lateral}) ∪ category_union)`.

`category_union` comprises OFA, FFA-1, FFA-2, EBA, FBA-1, FBA-2, PPA, OPA, OWFA, VWFA-1, VWFA-2. RSC is excluded. Paper uses NCSNR > 0.2 and category-localizer t > 1. Union category patches before final unique voxel counting; patches can extend outside the general/stream intersection. Do not intersect the entire union with `nsdgeneral` a second time.

Released `nsd_parser.py::load_NSD_benchmark_ROI_metadata` excludes early/midparietal/parietal stream labels, removes values below thresholds (retaining equality), and by default includes `early-visual` in a broader union. It is not sufficient to call that default function and name its output OTC. See audit D05.

The checked `publication/source_data/voxel_metadata_OTC.csv` has 29,480 rows: subject 1 = 8,088; 2 = 7,528; 5 = 8,015; 7 = 5,849. Lowest saved NCSNR is 0.20001102983951569. This independently agrees with the paper counts. Masks/voxel IDs, not counts alone, are needed for an exact reconstruction.

## Verified LAION resources and proposed correspondences

The [LAION ROI documentation](https://laion-fmri.hebartlab.com/rois.html) was inspected on 2026-10-24. Native-surface ROIs are canonical; T1w 1.8 mm masks are derived. General/lateral/ventral sectors use different definitions from NSD. Category masks use liberal t > 0 delineations; underlying contrast maps are described as forthcoming. Therefore a t > 1 rethreshold cannot yet be assumed possible from public masks alone.

| NSD component | LAION candidate | Status |
|---|---|---|
| nsdgeneral | laion-general | Adaptation: GLMsingle R² visual-selectivity definition |
| mid/high lateral/ventral streams | laion-lateral ∪ laion-ventral | Adaptation: anatomical parcels, not functional streams |
| OFA / FFA-1 / FFA-2 | Same names | Candidate correspondence; localizer definition/space still differs |
| EBA | EBA | Candidate correspondence |
| FBA-1 ∪ FBA-2 | FBA | Merged label; no invented subdivision |
| PPA / OPA | Same names | Candidate correspondence |
| RSC exclusion | Exclude MPA | LAION documents MPA as RSC |
| OWFA | No direct documented equivalent | Unresolved; do not relabel mfs-words as OWFA |
| VWFA-1 / VWFA-2 | Same names | Candidate correspondence |
| Additional LAION categories | pSTS-faces, SPL, mfs-words, object/motion masks | Do not automatically add to the original category set |
| NCSNR > 0.2 | Recovered NCSNR from appropriate repeat statistics | Unresolved exact map/estimation choice; not NC > 0.2 |

Proposed **adapted**, not identical, target: reliable voxels in the union of `laion-general ∩ (laion-lateral ∪ laion-ventral)` and the mapped category masks, with an explicit decision for unavailable OWFA and missing t maps. No ROI mask has been generated. Treat laion-EVC and retinotopic EVC as distinct labels; neither belongs in the primary OTC union merely because it is available.

## Reliability and beta provenance

[LAION GLMsingle documentation](https://laion-fmri.hebartlab.com/glmsingle_betas.html) provides final TYPED single-trial estimates and per-session/cross-session voxel noise-ceiling maps. Its NCSNR calculation normalizes betas, estimates within-image residual noise and uses nonnegative residual signal variance; NC percentage also depends on repeat counts. Preserve that convention and map scope. The same numeric threshold cannot be applied to NC percentages or GLMsingle R². NCSNR is not the GSN RSM-level ceiling used in Conwell et al.

Before an adapter may return an ROI: verify subject, space, hemisphere, affine/grid or surface indexing, mask origin, t-threshold availability, reliability source/scope, selected voxel IDs, overlaps, exclusion counts, finite values and per-subject totals. Save all of this with content hashes. Reliability estimation using evaluation stimuli must be disclosed; a training-only or independent reliability estimate changes the historical design and needs a named adaptation.

## Image splitting and repeats

The [LAION split documentation](https://laion-fmri.hebartlab.com/train_test_splits.html) currently offers a shared pool of 1,121 regular images, `tau` (897 train / 224 test), five cluster holdouts, and an OOD test of 371 images. These are not NSD's 500/500 split. Use exact label IDs, group all repeats of an image on one side, pin the release and split digest, and select model layers only inside each outer training set. Shared pool is the proposed analogue because the original study uses cross-subject shared stimuli. Whether to use all repeats or a matched-repeat subset remains explicit and unresolved; no data were downloaded to choose it here.
