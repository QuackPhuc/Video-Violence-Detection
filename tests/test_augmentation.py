import pytest
import numpy as np
from src.data.augmentation import (
    HorizontalFlip,
    RandomCrop,
    ColorJitter,
    RandomRotation,
    RandomTranslation,
    RandomErasing,
    FrameDropping,
    TemporalReversal,
    VideoAugmentation,
)


class TestDataHelper:
    """Helper class to create test data."""

    @staticmethod
    def create_video_features(frames=32, height=224, width=224, channels=6):
        """Create test video features with RGB + optical flow + magnitude."""
        features = np.random.rand(frames, height, width, channels).astype(np.float32)

        # RGB channels [0, 1]
        features[..., :3] = np.clip(features[..., :3], 0, 1)

        # Flow channels [-1, 1]
        features[..., 3:5] = np.clip(features[..., 3:5] * 2 - 1, -1, 1)

        # Magnitude channel
        features[..., 5] = np.linalg.norm(features[..., 3:5], axis=-1)

        return features

    @staticmethod
    def assert_valid_ranges(features):
        """Assert that feature values are in valid ranges."""
        # RGB channels should be in [0, 1]
        assert np.all(features[..., :3] >= 0), "RGB values below 0"
        assert np.all(features[..., :3] <= 1), "RGB values above 1"

        # Flow channels should be in [-1, 1]
        assert np.all(features[..., 3:5] >= -1), "Flow values below -1"
        assert np.all(features[..., 3:5] <= 1), "Flow values above 1"

        # Magnitude should be non-negative
        assert np.all(features[..., 5] >= 0), "Magnitude values below 0"


class TestBaseAugmentation:
    """Test the base augmentation class."""

    def test_validate_features(self):
        """Test that validate features works correctly."""
        aug = HorizontalFlip()

        # Test with 6 channels
        features = TestDataHelper.create_video_features()
        aug._validate_features(features)
        assert True

        # Test with 3 channels
        features = features[..., :3]
        with pytest.raises(ValueError):
            aug._validate_features(features)

        # Test with invalid dimensions
        features = TestDataHelper.create_video_features(
            frames=0, height=0, width=0, channels=6
        )
        with pytest.raises(ValueError):
            aug._validate_features(features)

    def test_probability_mechanism(self):
        """Test that probability mechanism works correctly."""
        # Test with probability 0 (never apply)
        aug = HorizontalFlip(probability=0.0)
        for _ in range(100):
            assert not aug.should_apply()

        # Test with probability 1 (always apply)
        aug = HorizontalFlip(probability=1.0)
        for _ in range(100):
            assert aug.should_apply()


class TestHorizontalFlip:
    """Test horizontal flip augmentation."""

    def test_shape_preservation(self):
        """Test that shape is preserved."""
        features = TestDataHelper.create_video_features()
        aug = HorizontalFlip(probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

    def test_flip_functionality(self):
        """Test that horizontal flip actually flips the image."""
        features = TestDataHelper.create_video_features()
        aug = HorizontalFlip(probability=1.0)
        result = aug(features)

        # Check that image is flipped horizontally (width dimension)
        expected_flip = np.flip(features, axis=2)
        expected_flip[..., 3] = -expected_flip[..., 3]  # Negate flow X

        np.testing.assert_array_equal(result, expected_flip)

    def test_value_ranges(self):
        """Test that value ranges are preserved."""
        features = TestDataHelper.create_video_features()
        aug = HorizontalFlip(probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)

    def test_no_flip_when_probability_zero(self):
        """Test that no flip occurs when probability is 0."""
        features = TestDataHelper.create_video_features()
        aug = HorizontalFlip(probability=0.0)
        result = aug(features)
        np.testing.assert_array_equal(result, features)


class TestRandomCrop:
    """Test random crop augmentation."""

    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        features = TestDataHelper.create_video_features()
        aug = RandomCrop(crop_ratio=0.8, probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

    def test_crop_ratio_validation(self):
        """Test that crop ratio is properly validated."""
        # Crop ratio should be clamped to [0.5, 1.0]
        aug = RandomCrop(crop_ratio=0.3)
        assert aug.crop_ratio == 0.5

        aug = RandomCrop(crop_ratio=1.2)
        assert aug.crop_ratio == 1.0

    def test_value_ranges(self):
        """Test that value ranges are preserved after crop and resize."""
        features = TestDataHelper.create_video_features()
        aug = RandomCrop(crop_ratio=0.8, probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)


class TestColorJitter:
    """Test color jitter augmentation."""

    def test_shape_preservation(self):
        """Test that shape is preserved."""
        features = TestDataHelper.create_video_features()
        aug = ColorJitter(probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

    def test_rgb_only_modification(self):
        """Test that only RGB channels are modified."""
        features = TestDataHelper.create_video_features()
        original_flow = features[..., 3:].copy()

        aug = ColorJitter(probability=1.0)
        result = aug(features)

        # Flow and magnitude channels should be unchanged
        np.testing.assert_array_equal(result[..., 3:], original_flow)

    def test_value_ranges(self):
        """Test that RGB values stay in [0, 1] range."""
        features = TestDataHelper.create_video_features()
        aug = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)

    def test_color_jitter_parameters(self):
        """Test ColorJitter with different parameter combinations."""
        features = TestDataHelper.create_video_features()

        # Test with zero parameters (minimal change due to floating point precision)
        aug = ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, probability=1.0)
        result = aug(features)
        # Allow small floating point differences
        np.testing.assert_array_almost_equal(
            result[..., :3], features[..., :3], decimal=5
        )

        # Test with extreme parameters
        aug = ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)


class TestRandomRotation:
    """Test random rotation augmentation."""

    def test_shape_preservation(self):
        """Test that shape is preserved."""
        features = TestDataHelper.create_video_features()
        aug = RandomRotation(max_angle=15.0, probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

    def test_value_ranges(self):
        """Test that value ranges are preserved after rotation."""
        features = TestDataHelper.create_video_features()
        aug = RandomRotation(max_angle=15.0, probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)


class TestRandomTranslation:
    """Test random translation augmentation."""

    def test_shape_preservation(self):
        """Test that shape is preserved."""
        features = TestDataHelper.create_video_features()
        aug = RandomTranslation(max_shift=0.1, probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

    def test_shift_parameter(self):
        """Test shift parameter initialization."""
        aug = RandomTranslation(max_shift=0.2)
        assert aug.max_shift == 0.2

    def test_value_ranges(self):
        """Test that value ranges are preserved after translation."""
        features = TestDataHelper.create_video_features()
        aug = RandomTranslation(max_shift=0.1, probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)


class TestRandomErasing:
    """Test random erasing augmentation."""

    def test_shape_preservation(self):
        """Test that shape is preserved."""
        features = TestDataHelper.create_video_features()
        aug = RandomErasing(probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

    def test_parameter_ranges(self):
        """Test parameter range initialization."""
        aug = RandomErasing(area_ratio_range=(0.1, 0.3), aspect_ratio_range=(0.5, 2.0))
        assert aug.area_ratio_range == (0.1, 0.3)
        assert aug.aspect_ratio_range == (0.5, 2.0)

    def test_value_ranges(self):
        """Test that value ranges are preserved."""
        features = TestDataHelper.create_video_features()
        aug = RandomErasing(probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)

    def test_erasing_functionality(self):
        """Test that erasing actually modifies the image."""
        features = TestDataHelper.create_video_features(frames=4, height=64, width=64)
        aug = RandomErasing(
            area_ratio_range=(0.3, 0.5), probability=1.0  # Large area to ensure effect
        )
        result = aug(features)

        # At least some pixels should be different due to erasing
        assert not np.array_equal(result, features)


class TestFrameDropping:
    """Test frame dropping augmentation."""

    def test_frame_replacement(self):
        """Test that frames are replaced not actually dropped."""
        features = TestDataHelper.create_video_features(frames=20)
        aug = FrameDropping(drop_ratio=0.3, probability=1.0)
        result = aug(features)

        # Should have same number of frames (frames replaced, not dropped)
        assert result.shape == features.shape

    def test_value_ranges(self):
        """Test that value ranges are preserved."""
        features = TestDataHelper.create_video_features(frames=20)
        aug = FrameDropping(drop_ratio=0.2, probability=1.0)
        result = aug(features)
        TestDataHelper.assert_valid_ranges(result)


class TestTemporalReversal:
    """Test temporal reversal augmentation."""

    def test_frame_reversal(self):
        """Test that frames are reversed in time."""
        features = TestDataHelper.create_video_features()
        aug = TemporalReversal(probability=1.0)
        result = aug(features)

        # Check that frames are reversed
        np.testing.assert_array_equal(result[..., :3], features[::-1, ..., :3])

        # Flow should be negated
        np.testing.assert_array_equal(result[..., 3:5], -features[::-1, ..., 3:5])

    def test_shape_preservation(self):
        """Test that shape is preserved."""
        features = TestDataHelper.create_video_features()
        aug = TemporalReversal(probability=1.0)
        result = aug(features)
        assert result.shape == features.shape


class TestVideoAugmentation:
    """Test the comprehensive video augmentation pipeline."""

    def test_default_initialization(self):
        """Test default initialization."""
        aug_pipeline = VideoAugmentation()
        assert len(aug_pipeline.spatial_augmentations) > 0
        assert len(aug_pipeline.temporal_augmentations) > 0

    def test_custom_augmentations(self):
        """Test with custom augmentation lists."""
        spatial_augs = [HorizontalFlip(probability=0.5)]
        temporal_augs = [TemporalReversal(probability=0.3)]

        aug_pipeline = VideoAugmentation(
            spatial_augmentations=spatial_augs, temporal_augmentations=temporal_augs
        )

        assert len(aug_pipeline.spatial_augmentations) == 1
        assert len(aug_pipeline.temporal_augmentations) == 1

    def test_shape_preservation(self):
        """Test that pipeline preserves spatial dimensions."""
        features = TestDataHelper.create_video_features()
        aug_pipeline = VideoAugmentation()
        result = aug_pipeline(features)

        # Spatial dimensions should be preserved
        assert result.shape[1:] == features.shape[1:]

    def test_value_ranges(self):
        """Test that value ranges are preserved through pipeline."""
        features = TestDataHelper.create_video_features()
        aug_pipeline = VideoAugmentation()
        result = aug_pipeline(features)
        TestDataHelper.assert_valid_ranges(result)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_frame_video(self):
        """Test with single frame video."""
        features = TestDataHelper.create_video_features(frames=1)
        aug_pipeline = VideoAugmentation()
        result = aug_pipeline(features)
        assert result.shape[1:] == features.shape[1:]

    def test_small_image_size(self):
        """Test with small image sizes."""
        features = TestDataHelper.create_video_features(frames=8, height=16, width=16)
        aug_pipeline = VideoAugmentation()
        result = aug_pipeline(features)
        assert result.shape[1:] == features.shape[1:]

    def test_different_channel_counts(self):
        """Test with different channel counts."""
        # Test with only RGB channels - most augmentations assume 6 channels
        features = np.random.rand(8, 32, 32, 3).astype(np.float32)

        # Test with other augmentations, they should expect 6 channels
        with pytest.raises(ValueError):
            aug = ColorJitter(probability=1.0)
            aug(features)
        with pytest.raises(ValueError):
            aug = RandomCrop(probability=1.0)
            aug(features)
        with pytest.raises(ValueError):
            aug = RandomRotation(max_angle=15.0, probability=1.0)
            aug(features)

    def test_zero_probability_pipeline(self):
        """Test pipeline where all augmentations have zero probability."""
        spatial_augs = [HorizontalFlip(probability=0.0)]
        temporal_augs = [TemporalReversal(probability=0.0)]

        features = TestDataHelper.create_video_features()
        aug_pipeline = VideoAugmentation(
            spatial_augmentations=spatial_augs, temporal_augmentations=temporal_augs
        )
        result = aug_pipeline(features)

        np.testing.assert_array_equal(result, features)

    def test_random_erasing_edge_cases(self):
        """Test RandomErasing with edge cases."""
        features = TestDataHelper.create_video_features(frames=4, height=8, width=8)

        # Test with very small area ratio
        aug = RandomErasing(area_ratio_range=(0.001, 0.002), probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

        # Test with large area ratio that might fail to find valid region
        aug = RandomErasing(area_ratio_range=(0.9, 0.95), probability=1.0)
        result = aug(features)
        assert result.shape == features.shape

    def test_frame_dropping_edge_cases(self):
        """Test FrameDropping edge cases."""
        features = TestDataHelper.create_video_features(frames=10)

        # Test with high drop ratio
        aug = FrameDropping(drop_ratio=0.8, probability=1.0)
        result = aug(features)
        assert result.shape[0] == features.shape[0]  # Should have same number of frames

        # Test with single frame
        features_single = TestDataHelper.create_video_features(frames=1)
        result = aug(features_single)
        np.testing.assert_array_equal(result, features_single)  # Should return original


class TestCoverage:
    """Additional tests for better coverage."""

    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        features = TestDataHelper.create_video_features()

        aug_pipeline1 = VideoAugmentation(seed=42)
        result1 = aug_pipeline1(features.copy())

        aug_pipeline2 = VideoAugmentation(seed=42)
        result2 = aug_pipeline2(features.copy())

        np.testing.assert_array_equal(result1, result2)

    def test_random_rotation_zero_angle(self):
        """Test RandomRotation with zero angle range."""
        features = TestDataHelper.create_video_features()
        aug = RandomRotation(max_angle=0.0, probability=1.0)
        result = aug(features)
        # With zero angle, result should be very close to original
        assert result.shape == features.shape

    def test_empty_augmentation_lists(self):
        """Test VideoAugmentation with empty augmentation lists."""
        # Create pipeline with explicitly empty lists
        aug_pipeline = VideoAugmentation(
            spatial_augmentations=[], temporal_augmentations=[]
        )

        # If the lists are empty, the default lists should be used
        assert len(aug_pipeline.spatial_augmentations) == 6
        assert len(aug_pipeline.temporal_augmentations) == 2


if __name__ == "__main__":
    pytest.main([__file__])
