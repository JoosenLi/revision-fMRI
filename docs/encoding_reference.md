# SRP, ridge and held-out evaluation reference

**Updated implementation policy:** use `DeepNSDEncodingEvaluator` for the
source-backed research path. It runs the original ridge file and original
RSA/SRP functions; see [upstream_reuse.md](upstream_reuse.md). The independent
SVD evaluator described below is retained as a test oracle only. It does not
determine the canonical method or provide a silent fallback.

Implemented on 2026-10-26 on `is248155`, using only small synthetic arrays. This
implements the numerical portion of T5/T6; it is not a dataset/model-extraction
pipeline or an approved scientific protocol. No network weights, fMRI data or GPU
experiments are involved.

## A consequential source finding

The unmodified released `ridge_gcv_mod.py` was executed with NumPy 1.26.4, SciPy
1.11.4 and scikit-learn 1.3.2. This is a **compatible reference environment**, not
a recovered historical environment. Its `_preprocess_data` signature has
`normalize` as the fourth parameter. The released source passes `self.copy_X`
in that position, enabling L2 feature normalization with default settings. The
initial audit's claim that this normalization was disabled was wrong and has
been corrected in D19. This normalization follows outer-training StandardScaler;
it is fitted once before the analytic LOO calculation, not refitted per fold.

`legacy_l2_normalize=True` explicitly reproduces this observed behavior.
`False` computes the independently tested ordinary unnormalized ridge variant,
but has no historical-equivalence claim. Neither is silently chosen for users.
Recovering the original Python environment/author intent remains necessary
before scientific experiments. The paper/code alpha-range discrepancy is also
still unresolved: the caller must supply the grid in order.

## Public interfaces

- `features.jl_dimensions(n_samples, eps=...)`: explicit JL calculation;
  `(1000, .1)` gives 5,920. No implicit LAION sample count.
- `features.project_features(..., n_components=..., seed=...)`: sklearn signed
  sparse random projection with automatic density, including upsampling. Keeps
  float32 input features at float32. Records dimensions, density, seed, source
  feature/image-order digests, sparse matrix digest and sklearn version.
- `encoding.fit_ridge_reference(..., alphas=..., legacy_l2_normalize=...)`:
  independent float64 SVD implementation of per-voxel Pearson-LOOCV selection.
  Returns coefficients, intercepts, selected alphas, centered LOO predictions
  and per-alpha scores. Intercept is unpenalized; first supplied alpha wins ties.
  Constant response columns and undefined Pearson scores raise explicit errors.
- `evaluation.ReferenceEncodingEvaluator.from_config(...)`: uses the configured
  ridge precision; currently rejects anything except float64. It applies SRP,
  fits StandardScaler only on outer training rows, runs ridge, and returns
  training predictions without response-mean restoration and test predictions
  with the fitted intercept. Responses from test rows never enter fitting.
- `metrics.versa(predicted, observed)`: Pearson correspondence of upper-triangle
  Pearson-distance RDMs over voxel patterns, not mean voxelwise correlation.
- `evaluation.evaluate_layers(...)`: takes explicit raw feature candidates for
  one model/subject/ROI. It independently maximizes training cRSA and veRSA,
  retaining first candidates in ties, then scores only the selected layer for
  each metric against test responses. It excludes C7 test-dependent filtering.

Example with already-local, already-ID-aligned API objects:

```python
from revision_fmri.config import load_config
from revision_fmri.evaluation import (
    RELEASED_CODE_ALPHAS, ReferenceEncodingEvaluator, evaluate_layers,
)

config = load_config()  # explicit YAML via REVISION_FMRI_CONFIG
encoder = ReferenceEncodingEvaluator.from_config(
    config,
    n_components=5920,  # explicit original-1000-image budget; not a LAION default
    seed=0,
    legacy_l2_normalize=True,  # observed sklearn 1.3.2 source behavior
)
# candidates: tuple[FeatureSet, ...]; responses: ResponseSet; split: EvaluationSplit
# result = evaluate_layers(candidates, responses, split, encoder,
#                          alphas=RELEASED_CODE_ALPHAS)
```

Do not run this example on real data until the dimension policy, alpha grid,
normalization, ROI, repeated-trial aggregation and split protocol are reviewed.
Features and responses must have identical ID order. Split IDs determine the
ordered train/test slices and must exactly cover the supplied images. Each
prediction includes its train/test image IDs; the response contract fixes voxel
order. Raw features remain available for cRSA; already-projected features are
rejected as cRSA or fresh-SRP input.

## Verification and fixture provenance

`tests/fixtures/encoding_reference.json` contains five small synthetic cases:
wide, tall, upsampling, near-collinear/constant-feature, and single target. The
reference inputs are float32 raw features with nonzero voxel response means and
varying signal/noise levels. The generator invokes the **unmodified** released
`RidgeCVMod`, checks its SHA-256 before import, and records the source URL, digest,
preprocessing signature and numerical package versions. It contains no data from
participants and no third-party implementation source.

Regenerate with an already-local audited source file:

```bash
.venv/bin/python -m pip install -e '.[test,encoding]'
.venv/bin/python scripts/generate_encoding_reference.py \
  --source /path/to/local/DeepNSD/source_code/pressures/ridge_gcv_mod.py \
  --output tests/fixtures/encoding_reference.json
.venv/bin/python -m pytest -q
```

All five cases match selected alphas exactly and coefficients, intercepts,
centered LOO/test predictions and alpha scores within predeclared `atol=1e-9`,
`rtol=1e-8`. Full SRP→scaler→ridge predictions are compared with the source
fixtures, and veRSA scores use absolute tolerance 1e-9. Separate brute-force
leave-one-out solves check both normalization policies at absolute tolerance
1e-12. Tests cover JL dimension, preserved feature precision, same-width
projection reuse, invalid grids, tie order, mismatched image IDs, constant
targets, test-feature independence of fitted scaler/coefficients, and
invariance of alpha/layer choice to changed test responses.

Run on 2026-10-26: **42 tests passed** (including the original 20 foundation
tests). This is fixture-level equivalence, not proof for every possible
conditioning regime. Float32 ridge is deliberately not implemented or approved;
no automatic float64 fallback is performed. GPU numerical parity is untested.

Observed maximum absolute errors across these five fixtures were 2.25e-14 for
coefficients, zero for intercepts, 3.79e-13 for centered LOO predictions,
1.21e-13 for test predictions and 2.40e-14 for alpha scores. All selected alphas
matched exactly. The wheel built successfully and its imports/JL calculation
passed outside the checkout without importing torch; `pip check` found no broken
requirements.

## Limits and next tasks

The SVD implementation is a readable small-array reference, not a production
memory-optimized backend. Layer evaluation retains predicted arrays for each
candidate; it is unsuitable for full-dataset execution without a later explicit
streaming/cache design. Sample weights, alternate alpha objectives and prediction
uncertainty are unsupported. Nearly singular leverage denominators fail instead
of silently regularizing or changing alpha grids.

Metadata includes array, projection, scaler, coefficient, split-ID, voxel-order,
repeat-count and candidate-order digests. These are result provenance, not an
implemented feature-cache storage system. Model checkpoints/transforms/hooks,
LAION adapters/ROI rules, GSN ceilings and R statistical parity remain pending.
The next scientific decision is D19 alongside D01/D02; implementation should
then proceed to data/ROI adapters once those inputs and policies are explicit.

D19 evidence update (2026-10-27): historical source signatures confirm the L2
binding across five pre-1.4 versions. See [G3 evidence](encoding_fidelity_regressions.md);
this supersedes treating the behavior as merely a 1.3.2 artifact. No numeric
default is changed; the exact historical environment remains unknown.
