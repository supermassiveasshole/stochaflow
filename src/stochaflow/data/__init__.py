"""Object-oriented data factories, mixtures, splits, and loaders."""

from .collation import ImageBatchCollator
from .contracts import (
    DatasetBuildRequest,
    DatasetFactory,
    DatasetFactoryContext,
    DatasetMixture,
    DatasetRole,
    DatasetSelection,
    DatasetView,
    ResolutionBucket,
    ResolutionBucketPolicy,
)
from .datasets import (
    BucketImageTransform,
    BucketedVisionDataset,
    CIFAR10DatasetFactory,
    Flowers102DatasetFactory,
    MNISTDatasetFactory,
)
from .pipeline import DataBundle, DataLoaderFactory, DataPipeline, SplitData
from .samplers import BucketBatchSampler, MixtureBatchSampler
from .splits import (
    KFoldSplitStrategy,
    OfficialSplitStrategy,
    RandomHoldoutSplitStrategy,
    SplitStrategy,
    TrainOnlySplitStrategy,
)

__all__ = [
    "BucketBatchSampler",
    "BucketImageTransform",
    "BucketedVisionDataset",
    "CIFAR10DatasetFactory",
    "DataBundle",
    "DataLoaderFactory",
    "DataPipeline",
    "DatasetBuildRequest",
    "DatasetFactory",
    "DatasetFactoryContext",
    "DatasetMixture",
    "DatasetRole",
    "DatasetSelection",
    "DatasetView",
    "Flowers102DatasetFactory",
    "ImageBatchCollator",
    "KFoldSplitStrategy",
    "MNISTDatasetFactory",
    "MixtureBatchSampler",
    "OfficialSplitStrategy",
    "RandomHoldoutSplitStrategy",
    "ResolutionBucket",
    "ResolutionBucketPolicy",
    "SplitData",
    "SplitStrategy",
    "TrainOnlySplitStrategy",
]
