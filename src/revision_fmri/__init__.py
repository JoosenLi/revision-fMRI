"""Shared public API; imports never load vision models, GPUs or external data."""
from .api import API_VERSION, DeepNSDProtocol, EvaluationReport, MetricResult, ReproductionPipeline
from .data import DatasetAdapter, ResponseSet
from .features import FeatureExtractor, FeatureSet
from .hooks import CapturedHookBatch, capture_feature_batch
from .evaluation import EvaluationSplit
from .models import ModelLoader, ModelSpec
from .vision import ClassificationTransform, LoadedVisionModel, TorchvisionAlexNetLoader
from .stimuli import StimulusBatch, StimulusReader
from .statistics import StatisticalAdapter, StatisticalSpec
from .laion import LaionNiftiAdapter
from .roi import VoxelMap, OTCProtocol, ROISelection, select_otc
from .response import NormalizedSession, normalize_session
from .cache import FeatureIdentity, FeatureStore, CachedModel
from .splits import RecordedSplit, load_split
from .primary import load_primary_laion_split
from .repeats import (TrialPresentation, TrialSession, aggregate_matched3, MATCHED3_POLICY,
                      validate_matched3_responses, write_trial_selection_manifest)
from .official_splits import (load_official_split, official_split_inventory,
                             TrialPartition, partition_trials, validate_holdout_family)

__version__ = "0.1.0"
__all__ = ["StimulusBatch", "StimulusReader", "TrialPresentation", "TrialSession", "aggregate_matched3", "MATCHED3_POLICY",
           "validate_matched3_responses", "write_trial_selection_manifest","API_VERSION", "DeepNSDProtocol", "EvaluationReport", "MetricResult",
           "ReproductionPipeline", "DatasetAdapter", "ResponseSet", "FeatureExtractor",
           "FeatureSet", "CapturedHookBatch", "capture_feature_batch", "EvaluationSplit", "ModelLoader", "ModelSpec",
           "ClassificationTransform", "LoadedVisionModel", "TorchvisionAlexNetLoader",
           "StatisticalAdapter", "StatisticalSpec", "LaionNiftiAdapter",
           "VoxelMap", "OTCProtocol", "ROISelection", "select_otc",
           "NormalizedSession", "normalize_session", "FeatureIdentity", "FeatureStore",
           "CachedModel", "RecordedSplit", "load_split", "load_primary_laion_split", "load_official_split",
           "official_split_inventory", "TrialPartition", "partition_trials", "validate_holdout_family"]
