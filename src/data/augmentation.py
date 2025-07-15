import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
import random
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift
from abc import ABC, abstractmethod


class BaseAugmentation(ABC):
    """Base class for all augmentation techniques."""

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def should_apply(self) -> bool:
        """Determine whether to apply augmentation based on probability."""
        return np.random.random() < self.probability

    def _validate_features(self, features: np.ndarray) -> None:
        """Validate that features have the correct shape and format."""
        if not isinstance(features, np.ndarray):
            raise ValueError("Features must be a numpy array")

        if features.ndim != 4:
            raise ValueError(
                f"Features must be 4D array (frames, height, width, channels), got {features.ndim}D"
            )

        frames, height, width, channels = features.shape

        # channels must be 6
        if channels != 6:
            raise ValueError(
                f"Features must have 6 channels (RGB + optical flow x/y + magnitude), got {channels}"
            )

        # frames, height, width, channels must be positive integers
        if frames <= 0 or height <= 0 or width <= 0 or channels <= 0:
            raise ValueError(
                f"Invalid dimensions: frames={frames}, height={height}, width={width}"
            )

    @abstractmethod
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Apply augmentation to features."""
        pass


class HorizontalFlip(BaseAugmentation):
    """Horizontal flip augmentation for video features."""

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        # Flip RGB and magnitude
        flipped = np.flip(features, axis=2)  # Flip width dimension
        # Negate flow X component
        flipped[..., 3] = -flipped[..., 3]

        return flipped


class RandomCrop(BaseAugmentation):
    """Random cropping with resize back to original size."""

    def __init__(self, crop_ratio: float = 0.8, probability: float = 0.5):
        super().__init__(probability)
        self.crop_ratio = max(0.5, min(1.0, crop_ratio))

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        frames, height, width, channels = features.shape

        # Calculate crop dimensions
        crop_height = int(height * self.crop_ratio)
        crop_width = int(width * self.crop_ratio)

        # Early return
        if crop_height == height and crop_width == width:
            return features

        # Random crop position
        top = np.random.randint(0, height - crop_height + 1)
        left = np.random.randint(0, width - crop_width + 1)

        # Crop all frames
        cropped = features[:, top : top + crop_height, left : left + crop_width, :]

        # Pre-allocate result array
        resized = np.empty_like(features, dtype=features.dtype)

        # Process all frames
        for i in range(frames):
            resized[i] = cv2.resize(
                cropped[i], (width, height), interpolation=cv2.INTER_LINEAR
            )

        # Vectorized clipping and magnitude calculation
        np.clip(resized[..., :3], 0, 1, out=resized[..., :3])  # RGB in-place
        np.clip(resized[..., 3:5], -1, 1, out=resized[..., 3:5])  # Flow in-place
        resized[..., 5] = np.linalg.norm(resized[..., 3:5], axis=-1)  # Magnitude

        return resized


class ColorJitter(BaseAugmentation):
    """Color jittering for RGB channels only."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        probability: float = 0.5,
    ):
        super().__init__(probability)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        augmented = features.copy()
        rgb_channels = augmented[..., :3]

        # Brightness adjustment
        brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        rgb_channels = np.clip(rgb_channels * brightness_factor, 0, 1)

        # Contrast adjustment
        contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        mean = np.mean(rgb_channels, axis=(1, 2), keepdims=True)
        rgb_channels = np.clip((rgb_channels - mean) * contrast_factor + mean, 0, 1)

        # Saturation adjustment
        saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
        for frame_idx in range(features.shape[0]):
            frame_rgb = rgb_channels[frame_idx]

            # Convert to HSV
            hsv = cv2.cvtColor(frame_rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
            # Adjust saturation
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0, 1)
            # Convert back to RGB
            rgb_channels[frame_idx] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            rgb_channels[frame_idx] = np.clip(rgb_channels[frame_idx], 0, 1)

        augmented[..., :3] = rgb_channels
        return augmented


class RandomRotation(BaseAugmentation):
    """Random rotation augmentation."""

    def __init__(self, max_angle: float = 30.0, probability: float = 0.5):
        super().__init__(probability)
        self.max_angle = max_angle

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        frames, height, width, channels = features.shape
        rotated = np.zeros_like(features)

        # Convert angle to radians for flow rotation
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for frame_idx in range(frames):
            # Rotate RGB channels
            for channel in [0, 1, 2]:
                rotated[frame_idx, :, :, channel] = scipy_rotate(
                    features[frame_idx, :, :, channel],
                    angle,
                    reshape=False,
                    mode="nearest",
                )

            # Special handling for optical flow
            flow_x = features[frame_idx, :, :, 3]
            flow_y = features[frame_idx, :, :, 4]

            # Rotate flow vectors
            rotated_flow_x = cos_a * flow_x - sin_a * flow_y
            rotated_flow_y = sin_a * flow_x + cos_a * flow_y

            # Rotate the flow fields spatially
            rotated[frame_idx, :, :, 3] = scipy_rotate(
                rotated_flow_x, angle, reshape=False, mode="nearest"
            )
            rotated[frame_idx, :, :, 4] = scipy_rotate(
                rotated_flow_y, angle, reshape=False, mode="nearest"
            )

        # Ensure valid ranges
        rotated[..., :3] = np.clip(rotated[..., :3], 0, 1)
        rotated[..., 3:5] = np.clip(rotated[..., 3:5], -1, 1)
        rotated[..., 5] = np.linalg.norm(rotated[..., 3:5], axis=-1)

        return rotated


class RandomTranslation(BaseAugmentation):
    """Random translation augmentation."""

    def __init__(self, max_shift: float = 0.1, probability: float = 0.5):
        super().__init__(probability)
        self.max_shift = max_shift

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        frames, height, width, channels = features.shape

        # Calculate shift amounts
        shift_y = int(np.random.uniform(-self.max_shift, self.max_shift) * height)
        shift_x = int(np.random.uniform(-self.max_shift, self.max_shift) * width)

        translated = scipy_shift(
            features, [0, shift_y, shift_x, 0], mode="constant", cval=0
        )

        # Ensure valid ranges
        translated[..., :3] = np.clip(translated[..., :3], 0, 1)
        translated[..., 3:5] = np.clip(translated[..., 3:5], -1, 1)
        translated[..., 5] = np.linalg.norm(translated[..., 3:5], axis=-1)

        return translated


class RandomErasing(BaseAugmentation):
    """Random erasing augmentation."""

    def __init__(
        self,
        area_ratio_range: Tuple[float, float] = (0.02, 0.2),
        aspect_ratio_range: Tuple[float, float] = (0.3, 3.0),
        probability: float = 0.5,
    ):
        super().__init__(probability)
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        frames, height, width, channels = features.shape
        area = height * width

        # Random erasing parameters
        target_area = np.random.uniform(*self.area_ratio_range) * area
        aspect_ratio = np.random.uniform(*self.aspect_ratio_range)

        h = int(np.sqrt(target_area * aspect_ratio))
        w = int(np.sqrt(target_area / aspect_ratio))

        if h < height and w < width:
            top = np.random.randint(0, height - h)
            left = np.random.randint(0, width - w)

            erased = features.copy()

            # Fill with random values for RGB channels
            erased[:, top : top + h, left : left + w, :3] = np.random.random(
                (frames, h, w, 3)
            )
            # Fill with zeros for flow channels
            erased[:, top : top + h, left : left + w, 3:] = 0

            return erased

        return features


class FrameDropping(BaseAugmentation):
    """Randomly drop frames and duplicate adjacent frames to maintain sequence length."""

    def __init__(self, drop_ratio: float = 0.1, probability: float = 0.5):
        super().__init__(probability)
        self.drop_ratio = max(0.0, min(0.5, drop_ratio))

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        frames = features.shape[0]

        if frames < 2:
            return features

        # Calculate number of frames to drop
        num_drop = int(frames * self.drop_ratio)
        if num_drop == 0 or num_drop == frames:
            return features

        # Randomly select frames to drop
        drop_indices = np.random.choice(frames, num_drop, replace=False)
        drop_indices = np.sort(drop_indices)

        result = features.copy()
        for drop_idx in drop_indices:
            if drop_idx == 0:
                result[drop_idx] = features[drop_idx + 1]
            else:
                result[drop_idx] = features[drop_idx - 1]

        return result


class TemporalReversal(BaseAugmentation):
    """Reverse the temporal order of frames."""

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)

        if not self.should_apply():
            return features

        reversed_features = features[::-1].copy()

        # Negate optical flow when reversing time
        reversed_features[..., 3:5] = -reversed_features[..., 3:5]

        return reversed_features


class VideoAugmentation:
    """Comprehensive video augmentation pipeline."""

    def __init__(
        self,
        spatial_augmentations: Optional[List[BaseAugmentation]] = None,
        temporal_augmentations: Optional[List[BaseAugmentation]] = None,
        probability: float = 0.5,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Default spatial augmentations
        self.spatial_augmentations = spatial_augmentations or [
            HorizontalFlip(probability=probability),
            RandomCrop(crop_ratio=0.8, probability=probability),
            ColorJitter(probability=probability),
            RandomRotation(max_angle=30.0, probability=probability),
            RandomTranslation(max_shift=0.1, probability=probability),
            RandomErasing(probability=probability),
        ]

        # Default temporal augmentations
        self.temporal_augmentations = temporal_augmentations or [
            TemporalReversal(probability=probability),
            FrameDropping(probability=probability),
        ]

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Apply augmentations to video features."""
        # Apply temporal augmentations first
        for aug in self.temporal_augmentations:
            features = aug(features)

        # Apply spatial augmentations
        for aug in self.spatial_augmentations:
            features = aug(features)

        return features
