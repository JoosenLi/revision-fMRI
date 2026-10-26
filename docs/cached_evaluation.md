# Pinned splits and shared feature artifacts

This stage connects stored raw features to the existing DeepNSD evaluation API.
It adds no model inference, split generation, scientific default, projection
cache, response preprocessing or statistical method.

## Artifact contract

`FeatureIdentity` records checkpoint, transform, candidate-manifest and stimulus
content digests, loader revision, explicit software versions and seed. Each
`FeatureStore` artifact additionally binds model/layer IDs, ordered image IDs,
array shape/dtype/content and feature provenance. All contribute to the artifact
key. Local roots do not. Supplying an identity is a declaration by the producer;
it does not prove a historical checkpoint or transform has been recovered.

Raw flattened float32/float64 features are supported. Float16, object arrays and
projected features are rejected. No precision conversion occurs. Generic raw
features are independent of subject/ROI/split; these identities belong in the
subsequent evaluation report, not in every raw-feature artifact. Any future cache
of fitted SRP/scalers/predictions must bind the fitting protocol/split separately.

```python
from revision_fmri import FeatureIdentity, FeatureStore, CachedModel

identity = FeatureIdentity(
    checkpoint_sha256=checkpoint_hash, transform_sha256=transform_hash,
    candidate_manifest_sha256=candidate_manifest_hash,
    stimuli_sha256=stimulus_content_manifest_hash,
    loader_revision=loader_commit, software_versions=extraction_versions,
    seed=extraction_seed,
)
store = FeatureStore.from_config()  # REVISION_FMRI_CONFIG or explicit YAML
keys = tuple(store.put(layer, identity=identity) for layer in raw_layer_features)
model = CachedModel(model_id, tuple(layer.layer_id for layer in raw_layer_features),
                    keys, identity)
```

Artifacts live at `feature_root/<key>/{values.npy,metadata.json,COMPLETE}`.
The store exclusively creates each key directory; `COMPLETE` is written last.
Existing complete or incomplete artifacts are never overwritten. Reuse them with
`get`, not another `put`. A failed write leaves an incomplete directory for
explicit inspection; retry does not erase it. Cooperating readers reject missing
markers, invalid metadata, changed values or identity mismatches. Pickle loading
is disabled; arrays are read-only memory maps and checked for finite values.
This is integrity checking, not protection against hostile concurrent filesystem
mutation. Checksums still read all bytes, and current validation may allocate
array-sized boolean buffers; large-scale throughput has not been benchmarked.

Move/copy the entire artifact directory between institutions under each person's
configured feature_root. Keep private filesystem paths out of supplied provenance.
The package preserves caller metadata; it does not automatically redact secrets.

## Explicit image split interchange

`load_split(config, filename=..., expected_sha256=...)` loads a UTF-8 JSON file
relative to configured data_root, verifies its exact byte digest and returns a
`RecordedSplit`. An example **synthetic**, not official LAION, manifest is:

```json
{
  "schema_version": 1,
  "protocol_id": "synthetic-split-v1",
  "dataset_release": "synthetic-fixture-v1",
  "source_revision": "synthetic-generator-v1",
  "pool": "synthetic",
  "train_ids": ["image-a", "image-b", "image-c"],
  "test_ids": ["image-d", "image-e", "image-f"]
}
```

This is the project's interchange format, not an invented description of the
LAION package's native split schema. Official shared-pool tau/cluster/OOD import is available separately via
`load_official_split`; no split is generated or selected by this interchange loader. IDs remain exact
strings in supplied order. Duplicate IDs, overlapping partitions and missing
provenance fail. Evaluation additionally requires exact coverage of response IDs.
Because ResponseSet groups all occurrences of each image, all repeats remain
in that image's partition. Trial/session policies are still the data adapter's
explicit responsibility. A file SHA binds byte content; even formatting changes
require a new digest.

## Evaluate from cache

```python
from revision_fmri import ReproductionPipeline, load_split
from revision_fmri.config import load_config

config = load_config()
split = load_split(config, filename="splits/reviewed.json",
                   expected_sha256=reviewed_split_file_hash)
pipeline = ReproductionPipeline(config, reviewed_protocol)
report = pipeline.evaluate_cached_model(model=model, responses=responses, split=split)
pipeline.save_report(report, run_id="subject01-model01-otc-v1")
```

This verifies each cache key, scientific identity, model ID and declared candidate
order, and requires response provenance `dataset_release` to match the split
release, then calls the same `evaluate_model` used for in-memory features. No cached
score or reference ranking bypasses evaluation. The report includes cache keys,
feature identity and split release/revision/pool, in addition to existing split,
ROI, response and numerical provenance. All-source session normalization remains
rejected in independent evaluation as before. No new LAION choices are implicit.

Tests exercise numeric equality with direct evaluation, report equality across
local roots, candidate/response mismatch, scientific identity invalidation,
changed image order/dtype, corruption, partial writes, symlinks, no overwrite and
pinned split errors. Model loading/hooks, additional official pools, reviewed ROI/
normalization integration and production-scale resource management remain pending.
