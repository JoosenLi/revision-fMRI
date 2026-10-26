# Offline vision compatibility API

## Implemented scope and evidence

**Released-code behavior (C9, methodology audit sections 1–2):**
`get_torchvision_model` and `get_torchvision_transforms` are exact function
excerpts from DeepNSD commit `36f6a34727dccb5eec8fa49364b210a049a388d7`.
The extraction manifest records source and redistributed SHA-256 values and
function line ranges. Only module imports are reduced. The adapter passes the
fixed constructor expression `models.alexnet`; caller-controlled expressions
are never evaluated.

For `alexnet_classification`, preprocessing is PIL `Resize((224,224))`,
`ToTensor`, then ImageNet mean/std normalization. Aspect ratio is changed; no
center crop is added. Modern pretrained-weight transform recipes are not used.

**LAION adaptation / engineering boundary (D14, D18):** instantiate with
`pretrained=False`, then load the configured local, SHA-256-verified plain tensor
state_dict with `weights_only=True`, CPU staging and strict keys/shapes/dtypes.
No key stripping, classifier removal, wrapper extraction, downloads or alternative
checkpoint resolution occur. Freeze all parameters, call eval, then transfer to
an explicitly allowed CUDA device. CPU staging never performs a forward pass.
RGB PIL input is required; grayscale/RGBA/palette inputs are rejected at this
transform boundary. The future shared stimulus reader must explicitly apply
`Image.open(path).convert('RGB')` to **every** stimulus before calling the
transform, matching released `feature_extraction.py:423` at the pinned commit.
The guard forbids implicit conversion inside the transform; it is not permission
to drop non-RGB stimuli or invent per-script conversions. This matters for LAION
OOD stimuli such as gabor, illusions and shapes.

**Unresolved:** historical torchvision/Pillow versions, historical checkpoint
bytes and interpolation defaults are not recovered. The compatibility profile
pins torch 2.6.0, torchvision 0.21.0, Pillow 12.0.0 (CUDA build suffixes are
recorded). In this profile PIL resize is bilinear; torchvision's antialias
setting has no separate effect on PIL bilinear resizing. These versions are
not claimed as the paper environment. A valid SHA proves the caller's local
artifact identity, not that it is the paper checkpoint. No model in the
publication manifest becomes scientifically approved through this adapter.

## Shared interface

Install the optional `vision` extra in the project environment. Importing the
base package does not import torch/torchvision/PIL or initialize CUDA.

```python
from revision_fmri import ModelSpec, TorchvisionAlexNetLoader
from revision_fmri.config import load_config
from revision_fmri.vision import LOADER_REVISION, TRANSFORM_ID

config = load_config()  # REVISION_FMRI_CONFIG or pass an explicit local YAML
loader = TorchvisionAlexNetLoader(config)
spec = ModelSpec(
    model_id="alexnet_classification",
    checkpoint_sha256=reviewed_checkpoint_sha256,
    loader_revision=LOADER_REVISION,
    transform_id=TRANSFORM_ID,
    candidate_manifest_sha256=reviewed_candidate_manifest_sha256,
)
# Only after model execution is separately authorized:
# Explicitly disable torch.backends.cuda.matmul.allow_tf32 and
# torch.backends.cudnn.allow_tf32 in the execution context; restore on exit.
loaded = loader.load_configured(spec, gpu_id=selected_allowed_gpu_id)
# loaded.model, loaded.preprocess, loaded.provenance
```

Bind `checkpoints.alexnet_classification` in ignored local YAML; relative paths
resolve against configured `checkpoint_root`. Alternatively the ModelLoader
protocol's `load(spec, checkpoint, device=...)` is supported, but both artifact
and device must match configuration. No default GPU index is selected.

This profile requires float32 features, default torch float32, disabled TF32 and
no active autocast. It neither changes these global settings nor offers a
float64 vision fallback. Ridge policy remains separate and unchanged. The
constructor's temporary CPU RNG is seeded at zero inside a restoring context;
all initialized tensors are overwritten by strict checkpoint loading. This does
not recover random-control seeds.

Before transfer the adapter checks CUDA availability, the selected CUDA UUID
against `nvidia-smi` compute-process UUIDs, and available memory. Other processes
on the selected GPU cause failure. UUID/process-query failures also stop loading.
The memory test covers model parameters/buffers only: it is a snapshot, not a
reservation or a guarantee that a later feature batch will fit. A future extractor
must check resources again immediately before inference and budget activations.

Provenance contains checkpoint, transform and caller-supplied candidate-manifest
hashes, original source identity, full dependency versions, precision, constructor
seed and GPU snapshot. No local artifact path is included. The transform digest
incorporates source identity, input contract and dependency versions. Candidate
manifest content is **not validated** by this loader and no hooks are selected.
No features, caches or scientific results are written here.

## Validation and remaining work

CPU tests use actual torch tensors and Pillow transforms. Rectangular synthetic
RGB inputs match both the upstream transform and independent explicit bilinear
resize/normalization exactly. Loader lifecycle tests substitute a tiny linear
constructor and mock device transfer/preflight; they verify safe load arguments,
strict loading, freezing/eval, RNG restoration, provenance and failure boundaries.
Unit resource checks use synthetic CUDA/process reports with bare torch UUIDs;
already-prefixed UUIDs are also accepted, while empty/MIG identities fail closed.
The optional real-device regression calls `_gpu_preflight` for configured allowed
GPUs, querying properties, compute processes and memory only. It does not load
weights, allocate model tensors or run inference (CUDA queries may initialize a
runtime context). Run it with the vision extra installed:

```sh
REVISION_FMRI_TEST_CUDA=1 REVISION_FMRI_CONFIG=configs/paths.local.yaml \
  .venv/bin/python -m pytest -q tests/test_vision.py -k real_gpu_preflight_read_only
```

It skips without opt-in, CUDA or a configured GPU allowance. Once configured,
UUID/process-query failures and other compute processes fail the test rather
than being hidden as skips. This does not establish
actual AlexNet checkpoint compatibility, CUDA execution or paper-level parity.

No real checkpoint was loaded, no AlexNet forward pass/GPU experiment was run,
and no checkpoints were downloaded. Other model families, real hook enumeration,
execution-order deduplication, image-to-feature-cache extraction and historical
runtime validation remain unimplemented. The extractor must also preserve the
upstream `nn.ReLU(inplace=False)` conversion before registering hooks
(`feature_extraction.py:231`, audit C1/C2); a future regression must check that
conv hook outputs retain pre-ReLU values. Statistical backends remain contract-only.

**Protocol-review flag: Requires 3-collaborator review.** This introduces a pinned
preprocessing compatibility profile and enforces the feature precision policy.
It does not change metrics, ridge hyperparameters, ROI/reliability, repetitions
or evaluation splits.
