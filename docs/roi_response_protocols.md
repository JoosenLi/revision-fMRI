# ROI and session-response protocol APIs

This stage implements explicit numerical primitives, not an approved LAION
scientific protocol. All three collaborators must review the scientific choices
before merging. No real OTC mask or full-session normalization run was produced.

## Voxel maps and OTC selection

Public imports: `VoxelMap`, `OTCProtocol`, `ROISelection`, `select_otc`.
Each map carries identical ordered voxel IDs, subject, space, grid digest and
source provenance. Binary masks and NCSNR maps have distinct units; NC percentages
and GLMsingle R² cannot be passed as NCSNR. There is no automatic conversion or
reliability estimation backend.

```python
from revision_fmri import OTCProtocol, select_otc

protocol = OTCProtocol(
    protocol_id="team-reviewed-laion-otc-v1",
    category_names=reviewed_category_names,
    ncsnr_threshold=0.2,
    threshold_operator="gt",
    mapping_decisions={
        "category_threshold": reviewed_localizer_policy,
        "missing_regions": reviewed_owfa_and_fba_mapping,
        "stream_mapping": reviewed_stream_mapping,
        "reliability_scope": reviewed_reliability_scope,
    },
)
selection = select_otc(general=general_map, lateral=lateral_map,
                       ventral=ventral_map, categories=category_maps,
                       reliability=ncsnr_map, protocol=protocol)
roi_responses = selection.apply(responses)
```

The expression is `reliable & ((general & (lateral | ventral)) | categories)`.
Category masks can extend outside general; overlapping inputs count only once.
`gt` matches the strict paper threshold; `ge` matches the released parser's
threshold equality behavior. Neither recreates the generic parser's broader
EVC-containing default. Category names must match the supplied maps exactly;
missing/extra names fail. Map alignment is checked before any union. Response
application also checks subject, voxel order, space and grid. Protocol JSON,
source-map provenance, digests and overlap/exclusion counts remain in results.

This is a shared aligned-array API. Loading component masks, deriving NCSNR and
applying localizer t thresholds remain explicit upstream responsibilities; no
binary mask is used to invent a t statistic. The existing NIfTI data adapter
still accepts a supplied mask and does not implicitly invoke this constructor.
A future reviewed manifest can connect these primitives without duplicating them.

The [current LAION ROI documentation](https://laion-fmri.hebartlab.com/rois.html),
rechecked 2026-10-26, still describes category delineations at t > 0 and underlying
t maps as forthcoming. Missing OWFA, merged FBA, changed stream definitions and
reliability estimation scope remain decisions, not inferred equivalences.

## Response normalization before repeat averaging

Public import: `normalize_session(values, policy=..., fit_rows=...)`.
Input is one session's trial-by-voxel array; it returns `NormalizedSession`
containing values and portable normalization provenance. Call before image-repeat
aggregation, retaining trial metadata. Do not normalize already averaged images
and label that operation as the original session preprocessing.

Verified source: [DeepNSD nsd_parser.py](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/brain_data/nsd_parser.py),
`load_NSD_brain_data`, lines 417–435. The volume branch selects voxels, applies
`stats.zscore` along the trial axis within each session, then concatenates sessions
and casts to float32. The helper uses the same scipy operation on the transposed
trial-by-voxel API orientation, ddof=0, with the same final float32 storage.
It is a small adaptation of the operation, not a vendored copy of the entire
NSD file loader. The reference source hash is recorded in output provenance.
This does not resolve the historical provenance of exported averaged beta CSVs.

| Explicit policy | Fitting scope | Interpretation |
| --- | --- | --- |
| `legacy_deepnsd_session_v1` | All provided session trials; fit_rows must be omitted | Source-style operation, includes held-out trials if present |
| `train_only_session_v1` | Explicit unique training row indices in this session | Independent preprocessing adaptation, not original-source parity |

Both use population standard deviation and return float32 after float64
calculation. Constant fitting voxels and nonfinite results fail explicitly.
No silent voxel removal occurs. The training-only variant requires at least two
training rows per session and does not estimate statistics from held-out rows.
Source-style and training-only policies are deliberately separate: all-session
normalization can make training representations depend on held-out responses.
The current independent `evaluate_layers`/`ReproductionPipeline` rejects response
provenance tagged `legacy_deepnsd_session_v1`. A legacy end-to-end evaluation
protocol (including C7 filtering) remains unimplemented.

The LAION reader's `normalization="none"` remains unchanged. These helpers do
not silently activate session normalization, choose repeats, or create a new
production-scale array cache. Callers must preserve returned provenance when
constructing ResponseSet objects.

## Review and tests

Analytic fixtures cover category extent outside general, overlapping regions,
threshold equality, empty selection, wrong units, missing components and grid/
voxel-order mismatches. Normalization fixtures check the trial axis, ddof,
float32 output, explicit fit-row validation and invariance of training outputs
when held-out values change. Independent evaluation rejects the legacy tag.

Still unresolved: final category mapping, t-map availability, source/scope of
NCSNR, full-session versus training-only preprocessing, repeat subset, image split,
and historical CSV export details. Unit-test success is not approval of those
scientific decisions or evidence of full reproduction.
