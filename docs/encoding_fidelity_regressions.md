# Encoding fidelity regression coverage (issue #12 G1–G3)

## G1: orchestration against a transcription of released code

`tests/test_end_to_end_fidelity.py` transcribes C1 `get_splithalf_xy` /
`run_benchmarking` and C6 `get_rdms` / `get_splithalf_rdms` at DeepNSD
`36f6a34727dccb5eec8fa49364b210a049a388d7`. Expected results do not use the
package's projection, metric, selection or encoding wrappers. They use sklearn
SRP/scaling and the unchanged vendored RidgeCVMod / compare_rdms primitives.
This is an independent orchestration check, not an independent implementation
of the original ridge mathematics (the SVD/component fixtures cover that).

Synthetic seed 1201: 80 images, three float32 layers of differing widths, 30
voxels with nonzero means. SRP fits all images before even/odd splitting;
StandardScaler fits training features only; alpha-selected centered LOOCV
predictions score training veRSA, and coefficient-plus-intercept predictions
score test veRSA. Brain RDMs are formed on the whole response matrix then sliced.
Compare every layer's training cRSA/veRSA, each metric's selected layer and held-out
score at absolute tolerance 1e-10 (rtol=0), plus alpha choices and prediction arrays.
The projection budget 9 is explicitly a tiny fixture, not a primary-run setting.
No C7 test-dependent filtering is included. No real scientific experiment is run.

## G2: original subject/ROI size eligibility

[C6 benchmark.get_rdms](https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/brain_data/benchmark.py#L127)
constructs an RDM only for `target_responses.shape[0] > 10` (voxel axis).
`evaluate_layers` now raises for fewer than 11 voxels before metric evaluation
or encoding. Primary, generic and cached facades all inherit this shared guard.
Failing explicitly instead of silently omitting the ROI is an engineering
adaptation; the eligibility boundary is released-code behavior. Report provenance
records voxel count and the minimum rule. Tests exercise 1, 10 and 11 voxels.

Low-level RDM/metric primitives and ridge component fixtures remain usable on tiny
arrays for mathematical validation; they are not subject/ROI benchmark results.
Orchestration fixtures now use >=11 voxels. No override or synthetic bypass was
added to the evaluator. This rule does not resolve anatomical ROI mapping or
reliability thresholds (G5 / issue #14).

## G3: historical sklearn signature evidence for D19

Official sklearn `_preprocess_data` sources were inspected on 2026-10-27:

| Tag | Positional argument 4 | Evidence |
|---|---|---|
| 0.24.2 | normalize | [source](https://github.com/scikit-learn/scikit-learn/blob/0.24.2/sklearn/linear_model/_base.py) |
| 1.0.2 | normalize | [source](https://github.com/scikit-learn/scikit-learn/blob/1.0.2/sklearn/linear_model/_base.py) |
| 1.1.3 | normalize | [source](https://github.com/scikit-learn/scikit-learn/blob/1.1.3/sklearn/linear_model/_base.py) |
| 1.2.2 | normalize | [source](https://github.com/scikit-learn/scikit-learn/blob/1.2.2/sklearn/linear_model/_base.py) |
| 1.3.2 | normalize | [source](https://github.com/scikit-learn/scikit-learn/blob/1.3.2/sklearn/linear_model/_base.py) |
| 1.4.0 | none; fit_intercept keyword-only | [source](https://github.com/scikit-learn/scikit-learn/blob/1.4.0/sklearn/linear_model/_base.py) |

Downloaded source SHA256 values and extracted signatures are retained in
[signature evidence](provenance/sklearn_preprocess_signatures.json). No sklearn
installation was replaced, and the older versions were not executed.

The released C4 call comments out `_normalize`, so its fourth positional argument
is `self.copy_X` (default True). This binds to normalize in all five inspected
pre-1.4 versions. Therefore `legacy_l2_normalize=True` is historically grounded
released-code behavior, not merely an accidental property of the 1.3.2 test
profile. The unmodified positional call is incompatible with the 1.4.0 signature.
These signatures do not prove every dependency combination runs, nor identify
the authors' exact environment. Numerical execution remains pinned to sklearn
1.3.2. No normalization algorithm or default is changed in this follow-up.

G4 alpha-grid selection and G5 OTC/reliability decisions remain with the owner/
collaborators in issue #14. Candidate enumeration is the next independent stage
per F1; it is not mixed into this numerical-regression PR.
