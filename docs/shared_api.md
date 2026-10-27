# Shared API v1

Collaborators import the public contracts from `revision_fmri`. Analysis scripts
provide data and metadata; the shared package owns scientific calculations.
`ReproductionPipeline` encapsulates the vendored DeepNSD numerical backend.
Do not import `_vendor` from analyses or duplicate the metrics in adapters.

| Boundary | Public contract | Current implementation |
| --- | --- | --- |
| Configuration | `ReproductionPipeline.from_config(path, protocol=...)` | YAML through `config.load_config`; omitted path uses `REVISION_FMRI_CONFIG` |
| Data | `DatasetAdapter`, `ResponseSet` | Validated image/voxel IDs, responses, repeat metadata; Explicit-manifest `LaionNiftiAdapter` for local T1w NIfTI/TSV; scientific ROI construction pending |
| Models | `ModelSpec`, `ModelLoader` | Offline AlexNet compatibility loader and local hash verification; other families pending |
| Features | `FeatureExtractor`, `FeatureSet` | Validated raw arrays, source SRP and single-batch hook capture; complete extractor/candidate selection pending |
| Evaluation | `EvaluationSplit`, `DeepNSDProtocol`, `ReproductionPipeline` | Original ridge/SRP/RSA behind validated orchestration; train-only layer selection |
| Results | `EvaluationReport`, `MetricResult` | JSON summary, content digest, exclusive creation under configured results root |
| Statistics | `StatisticalSpec`, `StatisticalAdapter` | Contracts only; original statistical fits pending |

## One evaluation per model, subject and ROI

```python
from revision_fmri import DeepNSDProtocol, ReproductionPipeline

# projection_count and projection_policy must come from the reviewed protocol.
protocol = DeepNSDProtocol(
    protocol_id="team-reviewed-laion-v1",
    n_components=projection_count,
    projection_policy=projection_policy,
)
pipeline = ReproductionPipeline.from_config(protocol=protocol)
report = pipeline.evaluate_model(
    features=candidate_features,  # Sequence[FeatureSet], in audited layer order
    responses=responses,         # ResponseSet for one subject/ROI
    split=split,                 # EvaluationSplit with explicit image IDs
)
path = pipeline.save_report(report, run_id="subject01-model01-otc-v1")
```

No default LAION projection dimension or split is inferred. The original
1000-image JL policy yields 5920 components; applying that dimension or
recomputing it for LAION needs a documented protocol decision. The pipeline
fixes the released-code alpha grid, SRP seed zero, float64 ridge and centered
LOOCV training predictions. Features remain float32 by default. See
[upstream reuse](upstream_reuse.md), D01, D02 and D19 in
[deviations](deviations.md): source reuse does not resolve the historical
environment or the paper/code conflicts. This API excludes the released
test-dependent layer filter, as required for independent evaluation.

## Collaboration rules

The data adapter owner supplies image-by-voxel arrays, explicit image/voxel IDs,
repeat counts, ROI/space and provenance. Repeats must remain within partitions;
the API does not guess how to aggregate trials. The model/feature owner supplies
one raw flattened image-by-feature array per candidate layer, with checkpoint,
transform and layer provenance. The evaluation owner uses this pipeline and
the same reviewed protocol. All candidates must belong to one model and have
the identical image order as responses. The split must exactly partition those
images. No implicit ID reordering occurs. Ties follow candidate insertion order.

Provenance supplied by adapters must contain portable logical identifiers and
hashes, not local paths or secrets. The API preserves caller metadata; it does
not redact it. Runtime paths belong only in each collaborator's ignored YAML.

`report.to_dict()` produces JSON-compatible API version `1`: model/subject/ROI,
image/voxel IDs, protocol, selected layers, training/test scores, per-layer
training scores, per-voxel alphas and numerical provenance. `content_sha256`
hashes this content. `save_report` adds the digest in an envelope and writes
only beneath configured `results_root`; an existing filename raises
`FileExistsError`. Invalid IDs, incompatible protocol/settings or nonfinite
scores raise `ValueError`; incompatible numerical dependencies or source
integrity failures raise `RuntimeError`. These errors must not trigger fallback
algorithms. Array predictions and large caches need a future storage API and
are not included in the JSON summary.

This version supports small supplied arrays, not a complete LAION experiment.
The synthetic example below exercises the entire public boundary without
loading images/checkpoints or initializing a GPU:

```bash
python examples/evaluate_synthetic.py --config configs/paths.local.yaml
```

It prints a summary and writes nothing unless `--run-id` is supplied. It is a
software demonstration, not a scientific projection or split recommendation.

For the local data adapter and acquisition boundary, see [LAION data adapter](laion_data_adapter.md).

ROI/response primitives are now exported as `VoxelMap`, `OTCProtocol`, `ROISelection`,
`select_otc`, `NormalizedSession` and `normalize_session`. See
[protocol APIs](roi_response_protocols.md) for source fidelity, explicit adaptations
and integration limits. They do not automatically replace the reader's supplied
mask or normalization policy.

`FeatureIdentity`, `FeatureStore`, `CachedModel`, `RecordedSplit` and `load_split`
provide portable cached evaluation through `ReproductionPipeline.evaluate_cached_model`.
See [cached evaluation](cached_evaluation.md) for the storage and split contracts.

`load_official_split` returns the same RecordedSplit consumed by cached evaluation.
`partition_trials` preserves repeats and reports excluded/missing images;
`validate_holdout_family` validates a declared five-fold family. See
[official LAION split import](official_laion_splits.md).

`TorchvisionAlexNetLoader`, `LoadedVisionModel` and `ClassificationTransform`
provide the first optional offline vision compatibility backend. See
[offline vision](offline_vision.md) for strict configuration, source reuse and
untested real-checkpoint/CUDA boundaries. Single-batch hooks are available; candidate selection remains pending.

`CapturedHookBatch` and `capture_feature_batch` expose original single-batch hooks
with ordered raw activations and provenance. See [feature hooks](feature_hooks.md).
The complete `FeatureExtractor` contract, candidate selection and cache-writing
integration remain unimplemented; this batch primitive is not a candidate list.

`load_primary_laion_split` and `ReproductionPipeline.evaluate_primary_laion_model`
implement the primary 561/560 even/odd protocol. Both require `repeat_policy`
without a default; the facade enforces `matched3_distinct_sessions_v1`, verifies
the selection manifest/counts and reuses training-only selection. Construct its
encoding protocol using `DeepNSDProtocol.jl_recomputed(split.split)` (6018 for the
full 1121-image pool). See [primary protocol](primary_laion_protocol.md) and
[report-facing replication differences](replication_differences.md).

`TrialPresentation`, `TrialSession` and `aggregate_matched3` provide the common
chronological selection/normalization/mean API. `LaionNiftiAdapter.responses_matched3`
feeds it validated local trial/beta metadata. Selection JSON artifacts are written
only beneath configured results_root; see [matched-three responses](matched3_responses.md).

Matched-three NIfTI calls require `ordering_source` (no default). Primary uses
`laion_volume_acquisition_order_v1` with explicit per-session `beta_sidecar` paths.
Schema v3 records distinct-session preference, within-session fill and source-group
counts. Primary requires `legacy_deepnsd_session_v1` and uses the validated,
primary-only `c6_legacy_session_zscore` exception. Generic independent APIs retain
their rejection of legacy normalization. See [F10/F11](matched3_responses.md).

## Local stimulus preparation

`StimulusReader` and `StimulusBatch` provide config-bound, hash-verified, ID-ordered
RGB preparation using the existing classification transform. Outputs stay on CPU;
no model execution occurs. See [stimulus reader](stimulus_reader.md).
