"""Explicit partitions and train-only selection; legacy test filtering is excluded."""
from dataclasses import dataclass
from typing import Mapping, Protocol
import numpy as np
from numpy.typing import NDArray
from .data import ResponseSet, unique_ids
from .features import FeatureSet
from .metrics import crsa

# Released C1 default, NOT a resolution of the conflicting paper range.
RELEASED_CODE_ALPHAS = tuple(np.logspace(-1, 5, 7).tolist())


@dataclass(frozen=True)
class EvaluationSplit:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    protocol_id: str
    source_digest: str

    def __post_init__(self):
        unique_ids(self.train_ids, 'Training IDs')
        unique_ids(self.test_ids, 'Test IDs')
        if set(self.train_ids) & set(self.test_ids):
            raise ValueError('An image cannot occur in both training and test sets')
        if not self.protocol_id or not self.source_digest:
            raise ValueError('Split protocol and provenance digest are required')

    def validate_coverage(self, image_ids: tuple[str, ...]) -> None:
        unique_ids(image_ids, 'Dataset IDs')
        if set(image_ids) != set(self.train_ids) | set(self.test_ids):
            raise ValueError('Split IDs must exactly partition the supplied image set')


def select_layer(training_scores: Mapping[str, float]) -> str:
    """One model/subject/ROI/metric. First inserted layer wins an exact tie.

    Test scores are intentionally not accepted. Legacy C7 eligibility filtering
    must be implemented separately under a documented protocol if required.
    """
    if not training_scores or any(not k or not np.isfinite(v) for k, v in training_scores.items()):
        raise ValueError('Require finite training scores and nonempty layer IDs')
    return max(training_scores, key=training_scores.__getitem__)


def evaluate_crsa(features: FeatureSet, responses: ResponseSet) -> float:
    if features.image_ids != responses.image_ids:
        raise ValueError('Feature and response image IDs must have identical order')
    if features.representation != 'raw_flattened':
        raise ValueError('Original cRSA requires raw flattened features')
    return crsa(features.values, responses.values)


@dataclass(frozen=True)
class EncodingPredictions:
    """Ridge adapter result with explicit prediction semantics and image order."""
    train_loo_centered: NDArray
    test_with_intercept: NDArray
    alpha_per_voxel: NDArray
    provenance: Mapping[str, str]
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


class EncodingEvaluator(Protocol):
    """Common interface for encoding backends."""
    def fit_predict(self, features: FeatureSet, responses: ResponseSet,
                    split: EvaluationSplit, *, alphas: tuple[float, ...]) -> EncodingPredictions: ...


@dataclass(frozen=True)
class _EncodingPipeline:
    """Shared validated orchestration; algorithms are supplied by explicit backends."""
    n_components: int
    seed: int
    legacy_l2_normalize: bool
    precision: str

    def __post_init__(self):
        if self.precision != 'float64':
            raise ValueError('Only float64 ridge is validated; float32 backend is not implemented')
        if type(self.legacy_l2_normalize) is not bool:
            raise ValueError('Explicit legacy L2 normalization policy is required')
        if type(self.n_components) is not int or self.n_components < 1:
            raise ValueError('Explicit positive projection count is required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Explicit unsigned 32-bit seed is required')

    @classmethod
    def from_config(cls, config, *, n_components: int, seed: int, legacy_l2_normalize: bool):
        """Honor runtime ridge precision; do not silently override float32 settings."""
        return cls(n_components, seed, legacy_l2_normalize, config.compute.ridge_precision)

    def _fit_ridge(self, x, y, alphas):
        raise NotImplementedError('An explicit encoding backend is required')

    def _backend_metadata(self):
        raise NotImplementedError('An explicit encoding backend is required')

    def fit_predict(self, features: FeatureSet, responses: ResponseSet,
                    split: EvaluationSplit, *, alphas: tuple[float, ...]) -> EncodingPredictions:
        import json
        import scipy
        import sklearn
        from sklearn.preprocessing import StandardScaler
        from .features import project_features
        from .provenance import array_digest, metadata_digest
        if features.image_ids != responses.image_ids:
            raise ValueError('Feature and response image IDs must have identical order')
        backend_metadata = self._backend_metadata()
        split.validate_coverage(responses.image_ids)
        indices = {name: i for i, name in enumerate(responses.image_ids)}
        train = [indices[name] for name in split.train_ids]
        test = [indices[name] for name in split.test_ids]
        projected = project_features(features, n_components=self.n_components, seed=self.seed)
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(projected.values[train])
        xtest = scaler.transform(projected.values[test])
        fit = self._fit_ridge(xtrain, responses.values[train], alphas)
        provenance = dict(projected.provenance)
        provenance.update({
            **backend_metadata, 'numpy_version': np.__version__,
            'scipy_version': scipy.__version__, 'sklearn_version': sklearn.__version__,
            'ridge_precision': self.precision, 'legacy_l2_normalize': str(self.legacy_l2_normalize),
            'alphas': json.dumps(list(alphas)), 'split_protocol': split.protocol_id,
            'split_source_digest': split.source_digest,
            'split_ids_sha256': metadata_digest({'train': split.train_ids, 'test': split.test_ids}),
            'train_responses_sha256': array_digest(responses.values[train]),
            'response_provenance_sha256': metadata_digest(dict(responses.provenance)),
            'voxel_order_sha256': metadata_digest(responses.voxel_ids),
            'scaler_sha256': metadata_digest({'mean': array_digest(scaler.mean_), 'scale': array_digest(scaler.scale_)}),
            'coefficients_sha256': array_digest(fit.coefficients),
            'intercept_sha256': array_digest(fit.intercept),
            'training_predictions': 'LOOCV; outer-training response mean NOT restored',
        })
        return EncodingPredictions(fit.loo_centered, fit.predict(xtest), fit.alpha_per_voxel,
                                   provenance, split.train_ids, split.test_ids)


@dataclass(frozen=True)
class ReferenceEncodingEvaluator(_EncodingPipeline):
    """Independent SVD test oracle; not the canonical research entry point."""

    def _fit_ridge(self, x, y, alphas):
        from .encoding import fit_ridge_reference
        return fit_ridge_reference(x, y, alphas=alphas, legacy_l2_normalize=self.legacy_l2_normalize)

    def _backend_metadata(self):
        return {'backend': 'float64-svd-reference-v1'}


@dataclass(frozen=True)
class DeepNSDEncodingEvaluator(_EncodingPipeline):
    """Canonical source-backed backend. SVD backend is retained as a test oracle.

    Public configuration cannot disable the behavior of the original ridge file.
    The inherited flag is fixed to True and rejected otherwise, never rewritten.
    """

    def __post_init__(self):
        super().__post_init__()
        if not self.legacy_l2_normalize:
            raise ValueError('DeepNSD source backend preserves original normalization; cannot disable it')
        if self.seed != 0:
            raise ValueError('Released DeepNSD profile requires SRP seed=0')

    @classmethod
    def from_config(cls, config, *, n_components: int):
        return cls(n_components=n_components, seed=0, legacy_l2_normalize=True,
                   precision=config.compute.ridge_precision)

    def _backend_metadata(self):
        from .deepnsd import require_compatible_environment
        return require_compatible_environment()

    def _fit_ridge(self, x, y, alphas):
        from .deepnsd import fit_ridge_deepnsd
        if tuple(alphas) != RELEASED_CODE_ALPHAS:
            raise ValueError('Released-code backend requires RELEASED_CODE_ALPHAS; alternative grids need a separate protocol')
        return fit_ridge_deepnsd(x, y, alphas=alphas)


def deepnsd_shared1000_split(image_ids: tuple[str, ...], *, source_digest: str) -> EvaluationSplit:
    """C1 even/odd image order, strictly for an explicitly ordered shared1000 set."""
    unique_ids(image_ids, 'Shared1000 image IDs')
    if len(image_ids) != 1000:
        raise ValueError('Original shared1000 split requires exactly 1000 image IDs')
    return EvaluationSplit(image_ids[::2], image_ids[1::2], 'deepnsd-shared1000-even-odd-v1', source_digest)


@dataclass(frozen=True)
class LayerEvaluation:
    """One model/subject/ROI; metric-specific selection precedes test scoring."""
    selected_layers: Mapping[str, str]
    training_scores: Mapping[str, Mapping[str, float]]
    test_scores: Mapping[str, float]
    predictions: Mapping[str, EncodingPredictions]
    provenance: Mapping[str, str]


def evaluate_layers(layers: tuple[FeatureSet, ...], responses: ResponseSet,
                    split: EvaluationSplit, encoder: EncodingEvaluator, *,
                    alphas: tuple[float, ...]) -> LayerEvaluation:
    """Evaluate an explicitly supplied candidate set, never infer hook eligibility.

    Caller invokes separately for each model/subject/ROI. No C7 test-dependent
    filtering. Only selected cRSA/veRSA layers are scored against test responses.
    """
    from .metrics import versa
    from .provenance import metadata_digest, array_digest
    if not layers or len({f.layer_id for f in layers}) != len(layers):
        raise ValueError('Candidate layers must be nonempty and have unique IDs')
    if len({f.model_id for f in layers}) != 1:
        raise ValueError('Evaluate one model at a time')
    for f in layers:
        if f.image_ids != responses.image_ids or f.representation != 'raw_flattened':
            raise ValueError('All candidate features must be raw and image-ID aligned')
    split.validate_coverage(responses.image_ids)
    index = {name: i for i, name in enumerate(responses.image_ids)}
    train, test = [index[x] for x in split.train_ids], [index[x] for x in split.test_ids]
    if min(len(train), len(test)) < 3:
        raise ValueError('RSA evaluation requires at least three images per partition')
    training = {'crsa': {}, 'versa': {}}
    predictions = {}
    for f in layers:
        training['crsa'][f.layer_id] = crsa(f.values[train], responses.values[train])
        pred = encoder.fit_predict(f, responses, split, alphas=alphas)
        predictions[f.layer_id] = pred
        training['versa'][f.layer_id] = versa(pred.train_loo_centered, responses.values[train])
    selected = {metric: select_layer(scores) for metric, scores in training.items()}
    lookup = {f.layer_id: f for f in layers}
    test_scores = {
        'crsa': crsa(lookup[selected['crsa']].values[test], responses.values[test]),
        'versa': versa(predictions[selected['versa']].test_with_intercept, responses.values[test]),
    }
    provenance = {
        'model_id': layers[0].model_id, 'subject_id': responses.subject_id,
        'roi_id': responses.roi_id, 'space': responses.space,
        'candidate_order_sha256': metadata_digest([f.layer_id for f in layers]),
        'response_values_sha256': array_digest(responses.values),
        'repeat_counts_sha256': metadata_digest(responses.repeat_counts),
        'selection': 'max training RSA; first candidate wins ties; no test-dependent filter',
    }
    return LayerEvaluation(selected, training, test_scores, predictions, provenance)
