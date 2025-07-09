import pytest
import numpy as np
import torch
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import NpzVideoDataset


class TestNpzVideoDataset:
    """Test NpzVideoDataset class."""

    @pytest.fixture
    def temp_dataset_setup(self):
        """Create temporary dataset for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV file
            csv_path = os.path.join(tmpdir, "labels.csv")
            df = pd.DataFrame(
                {
                    "file_name": [
                        "video1.npz",
                        "video2.npz",
                        "video3.npz",
                        "subdir/video4.npz",
                    ],
                    "label": [0, 1, 0, 1],
                }
            )
            df.to_csv(csv_path, index=False)

            # Create subdirectory
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir, exist_ok=True)

            # Create npz files
            for i, filename in enumerate(["video1.npz", "video2.npz", "video3.npz"]):
                file_path = os.path.join(tmpdir, filename)
                data = np.random.rand(16, 224, 224, 6).astype(np.float32)
                np.savez(file_path, data=data)

            # Create npz file in subdirectory
            subfile_path = os.path.join(subdir, "video4.npz")
            data = np.random.rand(8, 112, 112, 6).astype(np.float32)
            np.savez(subfile_path, data=data)

            yield tmpdir, csv_path

    def test_dataset_initialization(self, temp_dataset_setup):
        """Test dataset initialization."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(
            data_dir=tmpdir, csv_file=csv_path, channel_first=False
        )

        assert len(dataset) == 4
        assert dataset.data_dir == tmpdir
        assert not dataset.channel_first

    def test_dataset_with_channel_first(self, temp_dataset_setup):
        """Test dataset with channel_first=True."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(
            data_dir=tmpdir, csv_file=csv_path, channel_first=True
        )

        data, label = dataset[0]
        # Should be (C, T, H, W) format
        assert data.shape[0] == 6  # channels first
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)

    def test_dataset_without_channel_first(self, temp_dataset_setup):
        """Test dataset with channel_first=False."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(
            data_dir=tmpdir, csv_file=csv_path, channel_first=False
        )

        data, label = dataset[0]
        # Should be (T, H, W, C) format
        assert data.shape[-1] == 6  # channels last
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)

    def test_get_item(self, temp_dataset_setup):
        """Test __getitem__ method."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)

        for i in range(len(dataset)):
            data, label = dataset[i]
            assert isinstance(data, torch.Tensor)
            assert isinstance(label, int)
            assert label in [0, 1]
            assert data.dtype == torch.float32

    def test_label_distribution(self, temp_dataset_setup):
        """Test get_label_distribution method."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)

        distribution = dataset.get_label_distribution()
        assert distribution[0] == 2  # 2 non-violent videos
        assert distribution[1] == 2  # 2 violent videos

    def test_sample_info(self, temp_dataset_setup):
        """Test get_sample_info method."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)

        info = dataset.get_sample_info(0)
        assert "index" in info
        assert "file_path" in info
        assert "filename" in info
        assert "label" in info
        assert "file_size" in info
        assert info["index"] == 0
        assert info["label"] in [0, 1]

    def test_caching(self, temp_dataset_setup):
        """Test caching functionality."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path, cache_size=2)

        # Load first sample - should be cached
        data1, label1 = dataset[0]
        assert len(dataset.cache) == 1

        # Load second sample - should be cached
        data2, label2 = dataset[1]
        assert len(dataset.cache) == 2

        # Load third sample - cache should not grow beyond cache_size
        data3, label3 = dataset[2]
        assert len(dataset.cache) == 2

    def test_transform_application(self, temp_dataset_setup):
        """Test that transforms are applied correctly."""
        tmpdir, csv_path = temp_dataset_setup

        # Simple transform that adds 1 to all values
        def simple_transform(data):
            return data + 1.0

        dataset = NpzVideoDataset(
            data_dir=tmpdir, csv_file=csv_path, transform=simple_transform
        )

        data, label = dataset[0]
        # Since we added 1 to all values, minimum should be >= 1
        assert data.min() >= 1.0

    def test_invalid_csv(self):
        """Test error handling for invalid CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid CSV without required columns
            csv_path = os.path.join(tmpdir, "invalid.csv")
            df = pd.DataFrame({"wrong_column": [1, 2, 3]})
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="CSV file must contain"):
                NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)

    def test_nonexistent_directory(self):
        """Test error handling for non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid CSV
            csv_path = os.path.join(tmpdir, "labels.csv")
            df = pd.DataFrame({"file_name": ["video1.npz"], "label": [0]})
            df.to_csv(csv_path, index=False)

            # Try to create dataset with non-existent directory
            with pytest.raises(ValueError, match="No valid .npz files found"):
                NpzVideoDataset(data_dir="/nonexistent/path", csv_file=csv_path)

    def test_missing_npz_key(self, temp_dataset_setup):
        """Test error handling for npz files without 'data' key."""
        tmpdir, csv_path = temp_dataset_setup

        # Create npz file without 'data' key
        bad_file_path = os.path.join(tmpdir, "bad_video.npz")
        np.savez(bad_file_path, wrong_key=np.random.rand(16, 224, 224, 6))

        # Update CSV to include bad file
        df = pd.read_csv(csv_path)
        new_row = pd.DataFrame({"file_name": ["bad_video.npz"], "label": [0]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)

        dataset = NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)

        # Should raise error when trying to load the bad file
        with pytest.raises(KeyError, match="'data' key not found"):
            dataset[4]  # The bad file should be at index 4

    def test_file_extension_parameter(self, temp_dataset_setup):
        """Test file_extension parameter."""
        tmpdir, csv_path = temp_dataset_setup

        # Test with default extension
        dataset = NpzVideoDataset(
            data_dir=tmpdir, csv_file=csv_path, file_extension=".npz"
        )
        assert len(dataset) == 4

        # Test with different extension (should find no files)
        dataset_mp4 = NpzVideoDataset(
            data_dir=tmpdir, csv_file=csv_path, file_extension=".mp4"
        )
        # Should still be 4 because it scans for files with the extension
        # but they won't match the CSV, so actually should be 0
        with pytest.raises(ValueError, match="No valid .mp4 files found"):
            len(dataset_mp4)

    def test_path_normalization(self, temp_dataset_setup):
        """Test that path normalization works correctly on different OS."""
        tmpdir, csv_path = temp_dataset_setup

        dataset = NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)

        # Should find the subdirectory file correctly
        assert len(dataset) == 4

        # Check that all files are found regardless of path separators in CSV
        for i in range(len(dataset)):
            info = dataset.get_sample_info(i)
            assert os.path.exists(info["file_path"])


class TestNpzVideoDatasetEdgeCases:
    """Test edge cases for NpzVideoDataset."""

    def test_empty_directory(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "empty.csv")
            df = pd.DataFrame({"file_name": [], "label": []})
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="No valid .npz files found"):
                NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)

    def test_mismatched_csv_and_files(self):
        """Test when CSV references files that don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV with non-existent files
            csv_path = os.path.join(tmpdir, "labels.csv")
            df = pd.DataFrame(
                {"file_name": ["nonexistent1.npz", "nonexistent2.npz"], "label": [0, 1]}
            )
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="No valid .npz files found"):
                NpzVideoDataset(data_dir=tmpdir, csv_file=csv_path)


if __name__ == "__main__":
    pytest.main([__file__])
