import typing
from abc import ABC

from monai.transforms import *

from .custom import *

__all__ = [
    "TrainTransforms",
    "ValidationTransforms",
    "PredictionTransforms",
]


class Transforms(ABC):
    def __init__(self):
        self._transforms = Compose(
            [
                CustomLoadImaged(keys=["img", "label"]),
                CustomNormalizeImaged(
                    keys=["img"],
                    allow_missing_keys=True,
                    channel_wise=False,
                    percentiles=[0.0, 99.5],
                ),
                EnsureChannelFirstd(keys=["img", "label"], channel_dim=-1),
                RemoveRepeatedChanneld(keys=["label"], repeats=3),
                ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            ]
        )

    @property
    def transforms(self) -> Compose:
        return self._transforms


class TrainTransforms(Transforms):
    def __init__(self):
        super().__init__()
        self._transforms.transforms = self._transforms.transforms + (
                RandZoomd(
                    keys=["img", "label"],
                    prob=0.5,
                    min_zoom=0.25,
                    max_zoom=1.5,
                    mode=["area", "nearest"],
                    keep_size=False,
                ),
                SpatialPadd(keys=["img", "label"], spatial_size=512),
                RandSpatialCropd(
                    keys=["img", "label"], roi_size=512, random_size=False
                ),
                RandAxisFlipd(keys=["img", "label"], prob=0.5),
                RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=(0, 1)),
                IntensityDiversification(
                    keys=["img", "label"], allow_missing_keys=True
                ),
                RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
                RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
                RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
                RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
                RandGaussianSharpend(keys=["img"], prob=0.25),
                EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
        )


class ValidationTransforms(Transforms):
    pass


class PredictionTransforms(Transforms):
    def __init__(self):
        super().__init__()
        self._transforms = Compose(
            [
                CustomLoadImage(image_only=True),
                CustomNormalizeImage(channel_wise=False, percentiles=[0.0, 99.5]),
                AsChannelFirst(channel_dim=-1),
                ScaleIntensity(),
                EnsureType(data_type="tensor"),
            ]
        )
