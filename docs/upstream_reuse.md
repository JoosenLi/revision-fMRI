# Reuse of original DeepNSD algorithms

The primary implementation policy is source fidelity. Original algorithms are
executed directly wherever feasible; our code supplies local configuration,
input checks, scientific IDs and provenance. Algorithm changes are not justified
by speed, convenience or apparent improvement over the original.

## Pinned source and redistribution

Repository: https://github.com/ColinConwell/DeepNSD
Commit: `36f6a34727dccb5eec8fa49364b210a049a388d7`.
The prior audit verified eight key method files against the 2024-11-02 release.

| Local component | Original component | Modification |
|---|---|---|
| `_vendor/deepnsd/ridge_gcv_mod.py` | `source_code/pressures/ridge_gcv_mod.py` | None: byte-identical whole file |
| `_vendor/deepnsd/rsa.py` | `model_opts/mapping_methods.py::compare_rdms` | Exact function body; minimal module imports |
| `_vendor/deepnsd/reduction.py` | `model_opts/feature_reduction.py` SRP/JL functions and cache helpers | Exact function bodies; minimal module imports to avoid torch/model loading |

`_vendor/deepnsd/manifest.json` records source paths/hashes, excerpt line ranges,
function hashes and packaged file hashes. `scripts/vendor_deepnsd.py` reproduces
this from already-local audited sources, rejecting unexpected source bytes.
The original GPL-3.0 text, attribution/adaptation notice and scikit-learn BSD
notice are retained and shipped in wheels. The combined project is distributed
under the repository's GPL-3.0 LICENSE. No checkpoints or participant data are
included. Package imports never fetch models or make network requests.

## Canonical path

```python
from revision_fmri.config import load_config
from revision_fmri.evaluation import (
    DeepNSDEncodingEvaluator, RELEASED_CODE_ALPHAS, evaluate_layers,
)

settings = load_config()
encoder = DeepNSDEncodingEvaluator.from_config(
    settings,
    n_components=5920,  # original 1000-image JL budget, not a chosen LAION default
)
# Given precomputed, ID-aligned raw FeatureSets and a ResponseSet:
# result = evaluate_layers(candidates, responses, split, encoder,
#                          alphas=RELEASED_CODE_ALPHAS)
```

The canonical backend enforces seed 0, float64 ridge and the exact released
`np.logspace(-1,5,7).tolist()` grid, including its floating-point representation.
It does not expose a switch to disable the original preprocessing behavior.
It checks NumPy 1.26.4, SciPy 1.11.4, sklearn 1.3.2 and tqdm 4.67.1 before using
private sklearn APIs; mismatches fail with an explicit message, without a modern
RidgeCV or independent-SVD fallback. This is a tested compatibility environment,
not proof of the author's original run environment.

The original `get_feature_map_srps` ndarray/no-cache path performs SRP with
upsampling. The wrapper reconstructs the same seed-defined sparse matrix using
`fit` solely to hash its contents; it does not transform the features a second
time. This adds bookkeeping overhead but changes no projected values. The
original cache-writing/deleting helper branches are never called by the public
adapter. All future artifact storage must use configured roots.

Outer-training-only StandardScaler and prediction assembly follow C1. The
original `RidgeCVMod` selects per-voxel alphas and returns centered `cv_values_`;
we select the original chosen-alpha slices exactly. Test predictions use
`X_test.dot(coef_.T) + intercept_`. RSA comparison now calls the original
`compare_rdms` function, after shared input validation. No univariate voxel-score
average substitutes for veRSA.

`deepnsd_shared1000_split` provides the exact even/odd split for an explicit,
ordered list of exactly 1000 IDs. It does not invent an ordering or apply NSD's
split to an arbitrary LAION pool. The generic layer evaluator accepts reviewed
explicit split IDs and the actual candidate hook outputs supplied by callers.

## Conflicts remain explicit

- D01: source grid ends at 10^5; paper range ends at 10^7. The canonical source
  profile uses the released grid and records it; this does not establish which
  grid generated the published figures. Alternative grids need a separately
  named protocol, not mutation of the source backend.
- D19: the original positional preprocessing call enables L2 normalization in
  the pinned environment. We preserve that call unmodified. The independent
  SVD oracle supports both policies for investigation, not for silently changing
  the canonical algorithm.
- D02: independent held-out evaluation still excludes the R script's
  test-dependent degeneracy filter, as required by shared project instructions.
  We therefore do **not** claim that this evaluator reproduces every published
  postprocessing step. A legacy R path must be explicit and reviewed before use.
- D13: invalid IDs, nonfinite inputs and undefined correlations fail in the
  adapter; the malformed upstream warning branch is not used as error handling.

Default return shapes are adapted to image × voxel arrays, with one alpha per
voxel even for a single target. Shape adaptation changes no numerical values.
The ridge source's full `cv_values_` storage is retained; this is not a
memory-optimized or GPU backend. Small-array CPU regression tests are not vision
network training. Float32 feature arrays remain float32 before the original
ridge implementation promotes them to float64.

## Validation and remaining implementation

53 tests pass on `is248155`: original source integrity/license resources,
actual calls into the vendored RidgeCVMod (with the SVD oracle disabled), five
synthetic source fixtures, per-voxel alpha/prediction parity, projection and RSA,
source profile constraints, version-mismatch rejection, original split ordering,
and test-response-independent fitting/layer choice. The earlier independent
SVD/brute-force LOO tests continue to pass. Source fixtures were regenerated
using the exact C1 NumPy alpha expression.

The previous encoding-reference branch remains an ancestor of this branch;
this change adds source reuse rather than replacing collaborators' work.
Model loaders/preprocessing/hook execution, LAION beta/ROI adapters, GSN ceilings,
R statistical analysis and the full historical execution environment remain
pending. No real-data, GPU or remote experiment has run. Do not interpret source
reuse of these numerical components as completion of the entire study.
