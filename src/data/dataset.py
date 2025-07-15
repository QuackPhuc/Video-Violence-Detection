import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List, Dict
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class NpzVideoDataset(Dataset):
    """PyTorch Dataset for loading video clips from .npz files.

    Each .npz file should contain a 'data' key with shape [T, H, W, C] where:
    - T: number of frames
    - H: height
    - W: width
    - C: 6 channels (3 RGB + 2 optical flow + 1 magnitude)

    Labels are loaded from a CSV file with columns:
    - file_name: relative path to the .npz file
    - label: 0 for non-violent, 1 for violent
    """

    def __init__(
        self,
        data_dir: str,
        csv_file: str,
        transform: Optional[Callable] = None,
        channel_first: bool = True,
        cache_size: Optional[int] = None,
        file_extension: str = ".npz",
    ):
        """Initialize the dataset.

        Args:
            data_dir: Path to directory containing .npz files
            csv_file: Path to CSV file containing file names and labels
            transform: Optional transform to apply to samples
            channel_first: If True, convert to (C, T, H, W) format
            cache_size: Number of samples to cache in memory (None for no caching)
            file_extension: File extension to look for
        """
        self.data_dir = data_dir
        self.transform = transform
        self.channel_first = channel_first
        self.file_extension = file_extension

        # Load CSV file
        try:
            self.label_df = pd.read_csv(csv_file)
            if (
                "file_name" not in self.label_df.columns
                or "label" not in self.label_df.columns
            ):
                raise ValueError(
                    "CSV file must contain 'file_name' and 'label' columns"
                )

            # Create a mapping from file_name to label for faster lookup
            self.label_mapping = dict(
                zip(self.label_df["file_name"], self.label_df["label"])
            )
            logger.info(f"Loaded {len(self.label_mapping)} labels from {csv_file}")

        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            raise

        # Find all npz files that exist in both data_dir and CSV
        self.file_list = self._scan_directory(data_dir)

        if not self.file_list:
            raise ValueError(
                f"No valid {file_extension} files found in {data_dir} that match CSV labels"
            )

        logger.info(f"Found {len(self.file_list)} valid files in {data_dir}")

        # Initialize cache
        self.cache_size = cache_size
        self.cache: Dict[int, Tuple[torch.Tensor, int]] = {}

    def _scan_directory(self, directory: str) -> List[str]:
        """Recursively scan directory for npz files that have corresponding labels in CSV."""
        file_list = []

        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return file_list

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(self.file_extension):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)

                    # Normalize path separators
                    relative_path = relative_path.replace(os.path.sep, "/")

                    # Check if this file has a label in CSV
                    if relative_path in self.label_mapping:
                        file_list.append(file_path)
                    else:
                        logger.warning(f"File {relative_path} not found in CSV labels")

        return sorted(file_list)

    def _extract_label(self, file_path: str) -> int:
        """Extract label from filename using CSV mapping."""
        relative_path = os.path.relpath(file_path, self.data_dir)
        # Normalize path separators for Windows compatibility
        relative_path = relative_path.replace(os.path.sep, "/")

        if relative_path not in self.label_mapping:
            raise KeyError(f"Label not found for file: {relative_path}")

        return self.label_mapping[relative_path]

    def _load_data(self, file_path: str) -> np.ndarray:
        """Load data from npz file."""
        try:
            with np.load(file_path) as data:
                if "data" not in data:
                    raise KeyError(f"'data' key not found in {file_path}")
                return data["data"]
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (data, label) where:
            - data: torch.Tensor of shape (C, T, H, W) or (T, H, W, C)
            - label: int (0 or 1)
        """
        # Check cache first
        if self.cache_size and idx in self.cache:
            return self.cache[idx]

        # Get file path and label
        file_path = self.file_list[idx]
        label = self._extract_label(file_path)

        # Load data
        data = self._load_data(file_path)

        # Apply transforms
        if self.transform:
            data = self.transform(data)

        # Convert to tensor
        data = torch.tensor(data.copy(), dtype=torch.float32)

        # Convert to channel-first format if needed
        if self.channel_first and data.dim() == 4:
            # From (T, H, W, C) to (C, T, H, W)
            data = data.permute(3, 0, 1, 2)

        # Cache if enabled
        if self.cache_size and len(self.cache) < self.cache_size:
            self.cache[idx] = (data, label)

        return data, label

    def get_label_distribution(self) -> Dict[int, int]:
        """Get the distribution of labels in the dataset."""
        distribution = {0: 0, 1: 0}

        for file_path in self.file_list:
            label = self._extract_label(file_path)
            distribution[label] += 1

        return distribution

    def get_sample_info(self, idx: int) -> Dict[str, any]:
        """Get information about a specific sample without loading it."""
        file_path = self.file_list[idx]

        return {
            "index": idx,
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "label": self._extract_label(file_path),
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        }
