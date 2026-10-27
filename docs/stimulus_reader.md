# Shared local stimulus reader

`StimulusReader` prepares ID-aligned image batches for the existing model-specific
transform. It does not implement model inference or the full FeatureExtractor
contract. This stage is independent of candidate enumeration and GPU execution.

**Released-code behavior:** C2 `feature_extraction.py:423` at DeepNSD
`36f6a34727dccb5eec8fa49364b210a049a388d7` explicitly converts every opened image
with `.convert('RGB')`. The reader preserves this before invoking the existing
`ClassificationTransform` (C9: square Resize(224,224), ToTensor, ImageNet mean/std).
There is no added crop, EXIF transpose, augmentation, alpha compositing recipe or
modern weight-enum preprocessing. Pillow's ordinary first-frame behavior is
preserved for multi-frame files. Historical Pillow/decoder parity remains unresolved.

**Engineering / LAION adaptation:** explicit manifest, byte verification, ordered
IDs and config-relative storage replace implicit path discovery. No beta/repeat
aggregation, ROI mapping, split selection or normalization policy changes here.
Only the reviewed classification transform is supported; arbitrary callbacks and
other model-family transforms are rejected, not approximated.

## Manifest and API

Store the manifest beneath configured `stimuli_root`; all file paths are relative
to that root and may not escape it, including through symlinks. Files are already
local. No discovery, download or placeholder images occur.

```json
{
  "schema_version": 1,
  "dataset_release": "reviewed-release-id",
  "images": [
    {"image_id": "exact-label-from-trials", "file": "images/example.png",
     "sha256": "replace-with-full-64-character-lowercase-sha256"}
  ]
}
```

The example hash is a placeholder, not runnable input. Duplicate IDs and malformed
records fail. IDs are matched explicitly, never inferred from basenames.

```python
from revision_fmri import StimulusReader, ClassificationTransform
from revision_fmri.config import load_config

config = load_config()  # Explicit filename or REVISION_FMRI_CONFIG.
reader = StimulusReader(config, manifest="stimulus_manifest.json")
# Or use reader = StimulusReader.from_config(manifest="stimulus_manifest.json").
preprocess = ClassificationTransform()  # Same profile as loaded.preprocess.
for batch in reader.batches(image_ids=ordered_image_ids, batch_size=16,
                            preprocess=preprocess):
    # batch.image_ids; batch.inputs is CPU float32 N x 3 x 224 x 224.
    # Preserve batch.provenance in subsequent extraction artifacts.
    pass
```

All requested IDs are checked before the first batch is yielded. Request order is
preserved, including the final partial batch; no shuffling, sorting or dropping.
Each file is SHA256-checked immediately before those exact bytes are decoded,
so a changed file cannot be silently substituted between hashing and decoding.
Grayscale, RGBA and palette inputs are converted explicitly, never discarded.
Missing/corrupt/mismatched images fail instead of reducing the image pool.

This is CPU image preparation, **not CPU model inference or training**. The reader
never probes CUDA, constructs a network, loads weights or transfers tensors.
An authorized runner must explicitly move the tensor to the configured device
before calling `capture_feature_batch`; the latter still enforces its runtime
checks. No such runner is implemented in this PR. Batch source provenance is not
automatically attached by capture: the future extraction/cache orchestrator must
preserve it and link the shared input hash.

## Provenance and limits

Each batch carries manifest/release identity, requested/batch ordered-ID hashes,
image byte hashes, original mode/dimensions, decode rule, exact transform spec/hash,
software versions, float32 precision and output tensor hash. Seed is null (no
random operation). Institutional paths are absent. Relocating identical manifest
and image bytes beneath another configured root gives identical provenance.

The iterator holds one decoded image plus a batch's transformed tensors; it does
not preload the dataset. The caller must budget individual decode size and batch
size, and must not retain all batches if that exceeds memory. Earlier batches
may already have been consumed when a later file fails; this is not an atomic
whole-dataset extraction or cache transaction.

Synthetic tests verify RGB/L/RGBA/palette parity with the existing transform,
requested order, partial batches, batch-size-independent tensor values, relocation,
corrupt/missing/hash-mutated files, path/symlink escapes and explicit API failures.
No real stimuli/checkpoints were downloaded, no CUDA test or model forward was
run, and no feature cache or scientific results were produced. Candidate-layer
probe/dedup, dataset extraction/cache integration and statistical backends remain
unimplemented. Requires 3-collaborator review of the preprocessing boundary;
metric definitions, original hyperparameters, ROI/reliability rules, repetitions,
splits and precision policy are unchanged.
