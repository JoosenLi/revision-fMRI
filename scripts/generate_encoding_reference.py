"""Generate tiny synthetic fixtures using unmodified, local DeepNSD ridge source.

Requires the pinned encoding extra. No network, model, dataset or GPU access.
The source file is supplied explicitly, verified, and never vendored here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import inspect
import numpy as np
import scipy
import sklearn
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model._ridge import _preprocess_data

SOURCE_SHA256 = 'ea80dc539534f5f7370a796c3b2d91fbc73b5e0e4c94a78355616e4db7e62659'
ALPHAS = (.1, 1., 10., 100., 1000., 10000., 100000.)


def rsa(a, b):
    i = np.triu_indices(len(a), k=1)
    return float(pearsonr((1-np.corrcoef(a))[i], (1-np.corrcoef(b))[i]).statistic)


def generate(source, output):
    if sklearn.__version__ != '1.3.2' or np.__version__ != '1.26.4' or scipy.__version__ != '1.11.4':
        raise ValueError('Use the pinned encoding extra for fixture generation')
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Unexpected DeepNSD ridge source bytes')
    spec = importlib.util.spec_from_file_location('deepnsd_reference', source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    cases = []
    for name, width, projections, voxels in [('wide', 18, 24, 5), ('tall', 8, 5, 5),
                                            ('upsample', 3, 24, 5), ('collinear', 8, 5, 5),
                                            ('single_target', 8, 5, 1)]:
        rng = np.random.default_rng(207)
        raw = rng.normal(size=(24, width)).astype('float32')
        if name == 'collinear':
            raw[:, 1] = raw[:, 0] + 1e-6 * rng.normal(size=24)
            raw[:, -1] = 3.0
        # Different signal/noise levels yield distinct per-voxel alpha choices.
        responses = raw.astype(float) @ rng.normal(size=(width, voxels))
        responses += rng.normal(size=responses.shape) * np.linspace(.1, 8, voxels)
        responses += np.arange(voxels) * 5 + 2
        train, test = np.arange(16), np.arange(16, 24)
        srp = SparseRandomProjection(n_components=projections, random_state=0)
        projected = srp.fit_transform(raw)
        scaler = StandardScaler()
        xtrain, xtest = scaler.fit_transform(projected[train]), scaler.transform(projected[test])
        reg = legacy.RidgeCVMod(alphas=ALPHAS, store_cv_values=True,
                               alpha_per_target=True, scoring='pearson_r').fit(xtrain, responses[train])
        alpha = np.atleast_1d(reg.alpha_)
        selected = np.array([ALPHAS.index(float(a)) for a in alpha])
        loo = np.take_along_axis(reg.cv_values_, selected[None, :, None], axis=2)[:, :, 0]
        predicted = xtest @ np.atleast_2d(reg.coef_).T + reg.intercept_
        values = dict(name=name, raw=raw.tolist(), responses=responses.tolist(),
                      n_components=projections, seed=0, alphas=ALPHAS,
                      train=train.tolist(), test=test.tolist(), projected=projected.tolist(),
                      xtrain=xtrain.tolist(), xtest=xtest.tolist(),
                      coefficients=np.atleast_2d(reg.coef_).tolist(),
                      intercept=np.atleast_1d(reg.intercept_).tolist(), alpha=alpha.tolist(),
                      loo_centered=loo.tolist(), test_predictions=predicted.tolist(),
                      alpha_scores=[[float(pearsonr(responses[train, j], reg.cv_values_[:, j, k]).statistic)
                                     for k in range(len(ALPHAS))] for j in range(voxels)])
        if voxels > 1:
            values['train_versa'] = rsa(loo, responses[train])
            values['test_versa'] = rsa(predicted, responses[test])
        cases.append(values)
    payload = dict(source_sha256=SOURCE_SHA256,
                   source_url='https://github.com/ColinConwell/DeepNSD/blob/36f6a34727dccb5eec8fa49364b210a049a388d7/source_code/pressures/ridge_gcv_mod.py',
                   environment=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                   preprocess_signature=str(inspect.signature(_preprocess_data)),
                   historical_environment_claim=False, cases=cases)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
