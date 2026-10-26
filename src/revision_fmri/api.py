"""Stable collaborator-facing API v1; original methods stay behind this boundary."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import CachedModel
    from .splits import RecordedSplit

from .config import LocalConfig, load_config
from .data import ResponseSet
from .features import FeatureSet
from .evaluation import (DeepNSDEncodingEvaluator, EvaluationSplit,
                         RELEASED_CODE_ALPHAS, evaluate_layers)
from .provenance import metadata_digest

API_VERSION = '1'


@dataclass(frozen=True)
class DeepNSDProtocol:
    """Named source-backed protocol; callers must justify the projection dimension.

    Original algorithm settings (alpha grid, seed, prediction semantics) are not
    tuning knobs. This name does not claim that LAION inputs equal original NSD.
    """
    protocol_id: str
    n_components: int
    projection_policy: str

    def __post_init__(self):
        if not self.protocol_id or not isinstance(self.protocol_id, str):
            raise ValueError('A nonempty protocol ID is required')
        if type(self.n_components) is not int or self.n_components < 1:
            raise ValueError('Explicit positive projection count is required')
        if not self.projection_policy or not isinstance(self.projection_policy, str):
            raise ValueError('Describe the projection policy; no implicit LAION default')

    def to_dict(self) -> dict:
        return {**asdict(self), 'backend': 'deepnsd-vendored-sklearn132-v1',
                'alphas': list(RELEASED_CODE_ALPHAS), 'srp_seed': 0,
                'ridge_precision': 'float64',
                'selection': 'training-only; no C7 test-dependent filter',
                'training_prediction_semantics': 'centered LOOCV'}


@dataclass(frozen=True)
class MetricResult:
    selected_layer: str
    training_score: float
    test_score: float


@dataclass(frozen=True)
class EvaluationReport:
    """Portable JSON summary for one model × subject × ROI, without local paths.

    Predictions are deliberately excluded: large arrays need a separate future
    cache API. Per-voxel chosen alphas retain voxel ID order for reproducibility.
    """
    model_id: str
    subject_id: str
    roi_id: str
    space: str
    protocol: Mapping[str, object]
    train_image_ids: tuple[str, ...]
    test_image_ids: tuple[str, ...]
    voxel_ids: tuple[str, ...]
    response_provenance: Mapping[str, str]
    metrics: Mapping[str, MetricResult]
    layer_training_scores: Mapping[str, Mapping[str, float]]
    alpha_per_layer: Mapping[str, tuple[float, ...]]
    encoding_provenance: Mapping[str, Mapping[str, str]]
    evaluation_provenance: Mapping[str, str]

    def to_dict(self) -> dict:
        payload = {'api_version': API_VERSION, **asdict(self)}
        # Enforce JSON serialization and finite numbers at the API boundary.
        return json.loads(json.dumps(payload, allow_nan=False))

    @property
    def content_sha256(self) -> str:
        return metadata_digest(self.to_dict())


class ReproductionPipeline:
    """Single public orchestrator. Adapters supply ID-aligned arrays, not algorithms.

    Construction never initializes models/devices or accesses data files.
    Evaluation currently runs the small-array numerical backend on supplied
    features, not vision inference or a production-scale fMRI experiment.
    """
    def __init__(self, config: LocalConfig, protocol: DeepNSDProtocol):
        self.config = config
        self.protocol = protocol
        self._encoder = DeepNSDEncodingEvaluator.from_config(config, n_components=protocol.n_components)

    @classmethod
    def from_config(cls, path: str | Path | None = None, *, protocol: DeepNSDProtocol):
        """Use an explicit local YAML or REVISION_FMRI_CONFIG."""
        return cls(load_config(path), protocol)

    def evaluate_model(self, *, features: Sequence[FeatureSet], responses: ResponseSet,
                       split: EvaluationSplit) -> EvaluationReport:
        """One model/subject/ROI per call; candidate order breaks training-score ties."""
        layers = tuple(features)
        result = evaluate_layers(layers, responses, split, self._encoder, alphas=RELEASED_CODE_ALPHAS)
        metrics = {metric: MetricResult(layer, result.training_scores[metric][layer], result.test_scores[metric])
                   for metric, layer in result.selected_layers.items()}
        return EvaluationReport(
            layers[0].model_id, responses.subject_id, responses.roi_id, responses.space,
            self.protocol.to_dict(), split.train_ids, split.test_ids, responses.voxel_ids,
            dict(responses.provenance),
            metrics, result.training_scores,
            {layer: tuple(float(a) for a in pred.alpha_per_voxel) for layer, pred in result.predictions.items()},
            {layer: dict(pred.provenance) for layer, pred in result.predictions.items()},
            dict(result.provenance),
        )

    def evaluate_primary_laion_model(self, *, features: Sequence[FeatureSet],
                                     responses: ResponseSet, dataset_release: str,
                                     repeat_policy: str) -> EvaluationReport:
        """Primary 561/560 split; reuse training-only selection, never aggregate trials.

        repeat_policy must match the prepared responses' aggregation provenance.
        This records a caller decision, not approval/implementation of that policy.
        Projection dimensions remain explicit in this pipeline's DeepNSDProtocol.
        """
        from .primary import load_primary_laion_split, CANONICAL_ORDER_SHA256, SOURCE_SHA256
        split = load_primary_laion_split(self.config, dataset_release=dataset_release,
                                         repeat_policy=repeat_policy)
        if responses.provenance.get('dataset_release') != dataset_release:
            raise ValueError('Response dataset_release differs from primary protocol')
        if responses.provenance.get('aggregation') != repeat_policy:
            raise ValueError('Response aggregation must match explicit repeat_policy')
        report = self.evaluate_model(features=features, responses=responses, split=split.split)
        return replace(report, evaluation_provenance={**report.evaluation_provenance,
            'analysis_role': 'primary_replication', 'split_protocol_id': split.split.protocol_id,
            'split_sha256': split.split.source_digest, 'split_source_revision': split.source_revision,
            'split_dataset_release': dataset_release, 'split_pool': split.pool,
            'canonical_order_sha256': CANONICAL_ORDER_SHA256,
            'canonical_source_file_sha256': SOURCE_SHA256, 'repeat_policy': repeat_policy,
            'repeat_policy_status': 'explicit caller choice; aggregation not performed here'})

    def save_report(self, report: EvaluationReport, *, run_id: str) -> Path:
        """Write a new summary under configured results_root; never overwrite.

        Run IDs are filenames, not paths. Large predictions/features are not saved
        by this API. Existing artifacts and institutional source data are untouched.
        """
        if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', run_id):
            raise ValueError('run_id must be 1–128 letters, digits, hyphens or underscores')
        payload = report.to_dict()
        if payload['protocol'] != self.protocol.to_dict():
            raise ValueError('Report protocol differs from this pipeline')
        document = json.dumps({'content_sha256': report.content_sha256, 'report': payload},
                              indent=2, allow_nan=False)+'\n'
        root = self.config.storage['results_root']
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'{run_id}.json'
        # Exclusive creation also prevents writing through an existing symlink.
        with path.open('x') as stream:
            stream.write(document)
        return path

    def evaluate_cached_model(self, *, model: 'CachedModel', responses: ResponseSet,
                              split: 'RecordedSplit') -> EvaluationReport:
        """Evaluate a CachedModel against a RecordedSplit using configured storage.

        Verify identity/content and requested model/layer order before invoking
        the same scientific evaluator used by evaluate_model. No model inference.
        """
        from .cache import FeatureStore
        if responses.provenance.get('dataset_release') != split.dataset_release:
            raise ValueError('Response dataset_release must match the pinned split release')
        store = FeatureStore(self.config)
        layers = []
        for layer_id, key in zip(model.layer_ids, model.feature_keys):
            feature = store.get(key, expected_identity=model.identity)
            if feature.model_id != model.model_id or feature.layer_id != layer_id:
                raise ValueError('Cached model/layer differs from declared candidate order')
            layers.append(replace(feature, provenance={**feature.provenance,
                          'feature_cache_key': key,
                          'feature_identity_sha256': metadata_digest(asdict(model.identity))}))
        report = self.evaluate_model(features=layers, responses=responses, split=split.split)
        return replace(report, evaluation_provenance={**report.evaluation_provenance,
            'cached_model_sha256': metadata_digest(asdict(model)),
            'feature_identity': json.dumps(asdict(model.identity), sort_keys=True),
            'split_dataset_release': split.dataset_release,
            'split_source_revision': split.source_revision, 'split_pool': split.pool})
