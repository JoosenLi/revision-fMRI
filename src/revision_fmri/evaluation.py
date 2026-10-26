"""Explicit partitions and train-only selection; legacy test filtering is excluded."""
from dataclasses import dataclass
from typing import Mapping, Protocol
import numpy as np
from numpy.typing import NDArray
from .data import ResponseSet, unique_ids
from .features import FeatureSet
from .metrics import crsa

# Released C1 default, NOT a resolution of the conflicting paper range.
RELEASED_CODE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)


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
    """Future ridge adapter result; never infer training prediction semantics."""
    train_loo_centered: NDArray
    test_with_intercept: NDArray
    alpha_per_voxel: NDArray
    provenance: Mapping[str, str]


class EncodingEvaluator(Protocol):
    """SRP/scaler/ridge adapter remains pending legacy numerical parity tests."""
    def fit_predict(self, features: FeatureSet, responses: ResponseSet,
                    split: EvaluationSplit, *, alphas: tuple[float, ...]) -> EncodingPredictions: ...
