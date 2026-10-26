"""Shared public API; imports never load vision models, GPUs or external data."""
from .api import API_VERSION, DeepNSDProtocol, EvaluationReport, MetricResult, ReproductionPipeline
from .data import DatasetAdapter, ResponseSet
from .features import FeatureExtractor, FeatureSet
from .evaluation import EvaluationSplit
from .models import ModelLoader, ModelSpec
from .statistics import StatisticalAdapter, StatisticalSpec
from .laion import LaionNiftiAdapter
from .roi import VoxelMap, OTCProtocol, ROISelection, select_otc
from .response import NormalizedSession, normalize_session

__version__ = "0.1.0"
__all__ = ["API_VERSION", "DeepNSDProtocol", "EvaluationReport", "MetricResult",
           "ReproductionPipeline", "DatasetAdapter", "ResponseSet", "FeatureExtractor",
           "FeatureSet", "EvaluationSplit", "ModelLoader", "ModelSpec",
           "StatisticalAdapter", "StatisticalSpec", "LaionNiftiAdapter",
           "VoxelMap", "OTCProtocol", "ROISelection", "select_otc",
           "NormalizedSession", "normalize_session"]
