"""Feature API. Raw flattened features and SRP features are distinct artifacts."""
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Any
import numpy as np
from numpy.typing import NDArray
from .data import unique_ids


@dataclass(frozen=True)
class FeatureSet:
    image_ids: tuple[str, ...]
    model_id: str
    layer_id: str
    values: NDArray
    representation: str
    provenance: Mapping[str, str]

    def __post_init__(self):
        unique_ids(self.image_ids, 'Image IDs')
        if self.values.ndim != 2 or self.values.shape[0] != len(self.image_ids) or self.values.shape[1] == 0:
            raise ValueError('Features must have image × feature axes')
        if not np.isfinite(self.values).all():
            raise ValueError('Features must be finite')
        if self.representation not in {'raw_flattened', 'srp'}:
            raise ValueError('Feature representation must be explicit')
        if not all((self.model_id, self.layer_id, self.provenance)):
            raise ValueError('Model, layer and provenance are required')


class FeatureExtractor(Protocol):
    """Backend must reproduce the audited hook/dedup/candidate semantics."""
    def extract(self, model: Any, images: Iterable[Any], *, image_ids: tuple[str, ...],
                candidate_layers: tuple[str, ...]) -> Iterable[FeatureSet]: ...


def jl_dimensions(n_samples: int, *, eps: float) -> int:
    """Explicit JL policy helper. No implicit choice of LAION sample count."""
    if type(n_samples) is not int or n_samples < 2 or not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError('Require n_samples >= 2 and 0 < eps < 1')
    from ._vendor.deepnsd.reduction import compute_jl_lemma
    return int(compute_jl_lemma(n_samples, eps=eps))


def project_features(features: FeatureSet, *, n_components: int, seed: int) -> FeatureSet:
    """Legacy sklearn SRP with explicit dimensions, auto density and upsampling.

    Fit on all image features as in C3; randomness depends on width/seed, not
    responses. This is an encoding transform, never a cRSA input transform.
    """
    if features.representation != 'raw_flattened':
        raise ValueError('SRP requires raw flattened features; do not project twice')
    if features.values.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Raw feature precision must explicitly be float32 or float64')
    if type(n_components) is not int or n_components < 1:
        raise ValueError('Projection count must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('SRP seed must be an unsigned 32-bit integer')
    import sklearn
    from sklearn.random_projection import SparseRandomProjection
    from ._vendor.deepnsd.reduction import get_feature_map_srps
    from .provenance import array_digest, metadata_digest
    srp = SparseRandomProjection(n_components=n_components, density='auto', random_state=seed)
    values = get_feature_map_srps(features.values, n_projections=n_components,
                                  upsampling=True, eps=0.1, seed=seed, save_outputs=False)
    # Original API returns only projected values. Reconstruct its seed-identical
    # matrix for provenance, without a second feature transformation.
    srp.fit(features.values)
    components = srp.components_.tocsr()
    provenance = dict(features.provenance)
    provenance.update({
        'raw_features_sha256': array_digest(features.values),
        'image_order_sha256': metadata_digest(features.image_ids),
        'projection': 'sklearn.SparseRandomProjection', 'sklearn_version': sklearn.__version__,
        'n_components': str(n_components), 'seed': str(seed), 'density': repr(srp.density_),
        'projection_sha256': metadata_digest({
            'shape': components.shape, 'data': array_digest(components.data),
            'indices': array_digest(components.indices), 'indptr': array_digest(components.indptr)}),
    })
    return FeatureSet(features.image_ids, features.model_id, features.layer_id, values, 'srp', provenance)
