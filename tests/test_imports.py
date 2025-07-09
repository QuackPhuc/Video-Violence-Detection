"""Test script to verify all imports work correctly."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_data_imports():
    """Test data module imports."""
    print("Testing data module imports...")
    try:
        from src.data.dataset import NpzVideoDataset
        from src.data.augmentation import (
            BaseAugmentation,
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
        from src.data import NpzVideoDataset, VideoAugmentation

        print("✓ Data module imports successful")
    except Exception as e:
        print(f"✗ Data module import error: {e}")
        return False
    return True


def test_model_imports():
    """Test model module imports."""
    print("\nTesting model module imports...")
    try:
        from src.models.two_stream_cnn import (
            ConvBlock3D,
            StreamBlock,
            TwoStreamGated3DCNN,
        )
        from src.models import TwoStreamGated3DCNN

        print("✓ Model module imports successful")
    except Exception as e:
        print(f"✗ Model module import error: {e}")
        return False
    return True


def test_training_imports():
    """Test training module imports."""
    print("\nTesting training module imports...")
    try:
        from src.training.trainer import (
            EarlyStopping,
            MetricTracker,
            VideoActionTrainer,
        )
        from src.training import VideoActionTrainer

        print("✓ Training module imports successful")
    except Exception as e:
        print(f"✗ Training module import error: {e}")
        return False
    return True


def test_utils_imports():
    """Test utils module imports."""
    print("\nTesting utils module imports...")
    try:
        from src.utils.metrics import (
            MetricsCalculator,
            calculate_video_metrics,
            EarlyStopping,
        )
        from src.utils import MetricsCalculator, calculate_video_metrics

        print("✓ Utils module imports successful")
    except Exception as e:
        print(f"✗ Utils module import error: {e}")
        return False
    return True


def test_external_dependencies():
    """Test external dependencies."""
    print("\nTesting external dependencies...")
    dependencies = {
        "torch": "PyTorch",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "tqdm": "tqdm",
        "yaml": "PyYAML",
        "scipy": "SciPy",
        "pandas": "Pandas",
        "sklearn": "Scikit-learn",
    }

    all_good = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError:
            print(f"✗ {name} not installed")
            all_good = False

    return all_good


def test_model_instantiation():
    """Test model instantiation."""
    print("\nTesting model instantiation...")
    try:
        from src.models import TwoStreamGated3DCNN

        model = TwoStreamGated3DCNN(num_classes=1)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully with {param_count:,} parameters")
    except Exception as e:
        print(f"✗ Model instantiation error: {e}")
        return False
    return True


def test_dataset_instantiation():
    """Test dataset instantiation with dummy data."""
    print("\nTesting dataset instantiation...")
    try:
        from src.data import NpzVideoDataset
        import tempfile
        import os
        import pandas as pd

        # Create temporary directory and files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy CSV
            csv_path = os.path.join(tmpdir, "test.csv")
            df = pd.DataFrame(
                {"file_name": ["test1.npz", "test2.npz"], "label": [0, 1]}
            )
            df.to_csv(csv_path, index=False)

            # Create dummy npz files
            import numpy as np

            for filename in ["test1.npz", "test2.npz"]:
                file_path = os.path.join(tmpdir, filename)
                data = np.random.rand(16, 224, 224, 6).astype(np.float32)
                np.savez(file_path, data=data)

            # Test dataset creation
            dataset = NpzVideoDataset(
                data_dir=tmpdir, csv_file=csv_path, transform=None, channel_first=True
            )

            print(f"✓ Dataset created successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Dataset instantiation error: {e}")
        return False
    return True


def test_augmentation_instantiation():
    """Test augmentation instantiation."""
    print("\nTesting augmentation instantiation...")
    try:
        from src.data.augmentation import VideoAugmentation

        aug = VideoAugmentation()
        print(
            f"✓ Augmentation pipeline created with {len(aug.spatial_augmentations)} spatial and {len(aug.temporal_augmentations)} temporal augmentations"
        )
    except Exception as e:
        print(f"✗ Augmentation instantiation error: {e}")
        return False
    return True


def test_metrics_calculator():
    """Test metrics calculator instantiation."""
    print("\nTesting metrics calculator...")
    try:
        from src.utils import MetricsCalculator
        import torch

        calc = MetricsCalculator()

        # Test with dummy data
        outputs = torch.randn(10, 1)
        labels = torch.randint(0, 2, (10,))
        calc.update(outputs, labels)
        metrics = calc.compute_metrics()

        expected_keys = ["accuracy", "precision", "recall", "f1_score"]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

        print("✓ Metrics calculator works correctly")
    except Exception as e:
        print(f"✗ Metrics calculator error: {e}")
        return False
    return True


if __name__ == "__main__":
    print("Video Violence Detection - Import Test")
    print("=" * 50)

    results = []
    results.append(test_data_imports())
    results.append(test_model_imports())
    results.append(test_training_imports())
    results.append(test_utils_imports())
    results.append(test_external_dependencies())
    results.append(test_model_instantiation())
    results.append(test_dataset_instantiation())
    results.append(test_augmentation_instantiation())
    results.append(test_metrics_calculator())

    print("\n" + "=" * 50)
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
