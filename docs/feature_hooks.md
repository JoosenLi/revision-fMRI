# Original single-batch feature hooks

`capture_feature_batch(loaded, inputs, image_ids=..., config=...)` is the next
shared boundary after `TorchvisionAlexNetLoader`. It returns `CapturedHookBatch`
with image IDs, ordered flattened CPU float32 activations and provenance. It is
not yet an implementation of the complete `FeatureExtractor` contract and does
not produce a candidate manifest or write feature caches.

## Source fidelity

**Released-code behavior:** methodology audit C2, section 3. The hook routine,
ReLU conversion, naming, shape helpers and preparation function are exact
excerpts from `feature_extraction.py` at DeepNSD commit
`36f6a34727dccb5eec8fa49364b210a049a388d7`. Source SHA-256:
`72c358474f6264ebbbe97bc81eac262f95224a1c4da82ab0d79d96c882f4d0c5`.
The generator now preserves decorators as part of exact source excerpts.
Original imports are reduced; the stochastic dedup helper is included to satisfy
the original function dependency, but is never invoked by this batch API.

The wrapper calls `get_feature_maps` with `layers_to_retain=None`,
`remove_duplicates=False`, `enforce_input_shape=True`. This captures the raw
batch before candidate selection. It preserves:

- Forward execution order, including non-container parent modules and repeated
  calls to shared modules; `Sequential` and `ModuleList` are excluded.
- Original class-name/substring-count naming, not `named_modules()` paths.
- Non-inplace ReLU replacement before hooks, no-grad forward, detached CPU
  float32 tensors, and flattening of all non-image axes.
- One-based suffixes for tuple/list outputs. Mapping outputs are not traversed.
  Scalars and ambiguous non-leading batch axes are skipped. A leading axis
  matching batch size is accepted even if another dimension also matches.

**Paper evidence:** full computational-module comparisons motivate preserving
this enumeration. This primitive does not implement SI.4 candidate selection.

## Adaptation boundary

**LAION/engineering adaptations (D14, D15, D18):** explicit unique image IDs;
preprocessed tensor input; configuration-bound CUDA/device checks; frozen/eval
model checks; float32, no autocast/TF32; nonfinite/empty output rejection; portable
hashes. No tensor/model is implicitly moved by the wrapper. The original
preparation function's fallback is unreachable after the runtime checks. A
restoring CUDA device context binds its bare `.cuda()` call to the already
validated input/model device, including non-default configured GPUs.
The compatibility environment is the loader's pinned profile, with tqdm 4.67.1
for the vendored module import. Runtime versions must match loader provenance.

The caller uses the shared configuration loader and passes a loaded model plus
an already preprocessed `N x 3 x 224 x 224` tensor on its allowed device. The
future shared stimulus reader must perform explicit `.convert('RGB')` before the
original transform, as documented in [offline vision](offline_vision.md).
The caller must separately authorize model execution; this API does not launch
a job merely by importing it.

The model must have no existing forward/pre-forward hooks and be used
exclusively during capture. The wrapper removes this call's hooks even if the
original forward raises (upstream normally cleans up only on success). It
restores eval mode after the original ReLU replacement, because newly created
ReLU modules start with `training=True`. ReLU replacement persists; parameters
are not trained or changed. Cleanup uses torch's hook dictionaries under the
pinned compatibility profile. Concurrent model access is unsupported.

The loader's real GPU preflight is repeated immediately before capture. Its
one-byte available-memory check is only a resource sanity check, not an estimate
of activation memory. This in-memory primitive does not choose a safe batch size
or stream large activations to disk; callers must budget batch/host memory.

Provenance records loaded-model identity, source/dependency versions, image-order
and input hashes, ordered layer names and their digest, activation hashes and
precision. Stage is explicitly `unfiltered_single_batch_v1`. No random probe or
hash dedup is run, so the seed is null with an explanation. Skipped-output reasons
are upstream warnings, not yet a structured rejection manifest.

## Validation and remaining work

Tiny synthetic CPU modules test pre-ReLU linear outputs, repeated modules,
parent/container behavior, substring naming, tuple outputs, batch-axis movement,
scalar/ambiguous/mapping skipping, no-grad execution, finite-output validation,
repeatable hashes, and hook cleanup on failure. The test-only seam replaces
runtime/preparation checks; there is no public CPU inference option. CPU input
is explicitly rejected by the real runtime guard.

No AlexNet checkpoint, GPU forward, image dataset extraction or scientific score
was run for this change. Historical hook parity across other architectures and
real-device activation parity are unverified. Next steps are original unseeded
probe/dedup diagnostics, a reviewed candidate manifest (including later-half
selection), and bounded dataset extraction/cache integration. Original historical
RNG states remain unresolved; deterministic dedup substitution is not introduced.
Statistics remain contract-only.

**Requires 3-collaborator review:** feature definition/precision enforcement and
source-preserving hook semantics. No metrics, original ridge hyperparameters,
ROI/reliability, repetition handling or evaluation splits are changed.

## Candidate-enumeration acceptance spec (F1)

**Released-code behavior (C2, not yet implemented end to end):** preserve this
sequence from `main_analysis.py:243-250` and `feature_extraction.py:252-329`
at the pinned DeepNSD revision above:

1. `get_empty_feature_maps(model, names_only=True, input_shape=(3,224,224))`
   probes with **`torch.rand(3,3,224,224)`** (unseeded, not stimuli) and **hashkey
   dedup**, which uses an unseeded `torch.rand` projection vector.
2. `layer_subset = names[int(len(names)/2):]`: SI.4 later half **of the
   deduplicated list**, preserving order and Python integer truncation.
3. `get_all_feature_maps(model, loader, layers_to_retain=layer_subset)` uses
   default `remove_duplicates=True`: per-batch capture without dedup, then a
   **second hashkey dedup over full-dataset maps**, then flatten.

Do not halve the raw hook list or deduplicate each stimulus batch. Historical RNG
states are unresolved (D15). A future reproducible seeded diagnostic must record
its seed as an adaptation; the source procedure itself was unseeded. This PR
specifies acceptance, not a candidate-enumeration implementation.

## AlexNet expected golden values (F2/F3)

The `alexnet_classification` row in [published metadata](../manifests/models.json)
has `model_depth=22`, `total_feature_count=1099216`, and
`total_parameter_count=61100840`. The following hook/dedup interpretation is a
**reviewer-derived prediction**, to be confirmed by the first authorized real
run; it is not evidence that we have run historical weights:

- **22 raw hooked outputs**: 13 features + AdaptiveAvgPool2d + 7 classifier +
  the top-level AlexNet. Their summed flattened feature sizes are **1,099,216**.
  Thus these published counts correspond to the list **before dedup**.
- Expected duplicates: `AdaptiveAvgPool2d-1` equals `MaxPool2d-3` (6x6 to 6x6
  adaptive pooling); `Dropout-1` and `Dropout-2` are identities in eval;
  `AlexNet-1` equals `Linear-3`. Expected unique count: **18**, not 19.
- Later half `names[9:]`: **9 scored layers**, in order:
  `ReLU-4, Conv2d-5, ReLU-5, MaxPool2d-3, Linear-1, ReLU-6, Linear-2, ReLU-7, Linear-3`.

`test_real_gpu_random_alexnet_hook_golden` is opt-in via
`REVISION_FMRI_TEST_CUDA_FORWARD=1` and a local config through
`REVISION_FMRI_CONFIG`. It is **not run by default and has not been run**.
It uses random-init AlexNet (no checkpoint/download), the real preflight and
`capture_feature_batch`, checking all 22 names in exact order and the size sum.
It does not test candidate dedup, historical-weight parity or scientific scores.
The diagnostic constructor/input seed is 0, explicitly distinct from the original
unseeded enumeration. GPU selection comes from config; busy/unavailable/insufficient
resources do not trigger a CPU fallback. Authorize GPU work before opting in.
