import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import EarlyStopping, MetricTracker, VideoActionTrainer


class TestEarlyStopping:
    """Test EarlyStopping class in trainer module."""

    def test_initialization(self):
        """Test EarlyStopping initialization."""
        early_stop = EarlyStopping(patience=5, min_delta=0.01, mode="min")

        assert early_stop.patience == 5
        assert early_stop.min_delta == 0.01
        assert early_stop.mode == "min"
        assert early_stop.counter == 0
        assert early_stop.best_value == float("inf")
        assert not early_stop.early_stop

    def test_call_improvement(self):
        """Test __call__ with improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        # First call should always be improvement
        should_stop = early_stop(1.0)
        assert not should_stop
        assert early_stop.best_value == 1.0
        assert early_stop.counter == 0

        # Significant improvement
        should_stop = early_stop(0.8)
        assert not should_stop
        assert early_stop.best_value == 0.8
        assert early_stop.counter == 0

    def test_call_no_improvement(self):
        """Test __call__ without improvement."""
        early_stop = EarlyStopping(patience=2, min_delta=0.01, mode="min")

        early_stop(1.0)

        # No improvement
        should_stop = early_stop(1.0)
        assert not should_stop
        assert early_stop.counter == 1

        # Still no improvement - should trigger stopping
        should_stop = early_stop(1.0)
        assert should_stop
        assert early_stop.early_stop

    def test_state_dict(self):
        """Test state_dict method."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        early_stop(1.0)
        early_stop(1.0)  # counter = 1

        state = early_stop.state_dict()

        assert state["counter"] == 1
        assert state["best_value"] == 1.0
        assert state["early_stop"] == False

    def test_load_state_dict(self):
        """Test load_state_dict method."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        state = {"counter": 2, "best_value": 0.5, "early_stop": True}

        early_stop.load_state_dict(state)

        assert early_stop.counter == 2
        assert early_stop.best_value == 0.5
        assert early_stop.early_stop == True


class TestMetricTracker:
    """Test MetricTracker class."""

    def test_initialization(self):
        """Test MetricTracker initialization."""
        tracker = MetricTracker()

        assert tracker.metrics == {}
        assert tracker.counts == {}

    def test_reset(self):
        """Test reset functionality."""
        tracker = MetricTracker()

        # Add some data
        tracker.update({"loss": 1.0, "accuracy": 0.8}, n=5)
        assert len(tracker.metrics) > 0

        # Reset should clear everything
        tracker.reset()
        assert tracker.metrics == {}
        assert tracker.counts == {}

    def test_update_single_metric(self):
        """Test update with single metric."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0}, n=2)

        assert tracker.metrics["loss"] == 2.0  # 1.0 * 2
        assert tracker.counts["loss"] == 2

    def test_update_multiple_metrics(self):
        """Test update with multiple metrics."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0, "accuracy": 0.8, "f1": 0.7}, n=3)

        assert tracker.metrics["loss"] == 3.0
        assert tracker.metrics["accuracy"] == 2.4
        assert tracker.metrics["f1"] == 2.1
        assert all(count == 3 for count in tracker.counts.values())

    def test_update_accumulation(self):
        """Test accumulation over multiple updates."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0}, n=2)
        tracker.update({"loss": 0.5}, n=4)

        assert tracker.metrics["loss"] == 4.0  # (1.0*2) + (0.5*4)
        assert tracker.counts["loss"] == 6  # 2 + 4

    def test_average_single_metric(self):
        """Test average calculation for single metric."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0}, n=2)
        tracker.update({"loss": 0.5}, n=4)

        avg = tracker.average()

        expected_avg = 4.0 / 6  # total / count
        assert avg["loss"] == expected_avg

    def test_average_multiple_metrics(self):
        """Test average calculation for multiple metrics."""
        tracker = MetricTracker()

        tracker.update({"loss": 1.0, "accuracy": 0.8}, n=2)
        tracker.update({"loss": 0.5, "accuracy": 0.9}, n=3)

        avg = tracker.average()

        assert avg["loss"] == (2.0 + 1.5) / 5  # (1.0*2 + 0.5*3) / (2+3)
        assert avg["accuracy"] == (1.6 + 2.7) / 5  # (0.8*2 + 0.9*3) / (2+3)

    def test_average_empty(self):
        """Test average with no data."""
        tracker = MetricTracker()

        avg = tracker.average()

        assert avg == {}


class DummyModel(nn.Module):
    """Simple dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x.mean(dim=[2, 3, 4]))  # Reduce spatial/temporal dims


class TestVideoActionTrainer:
    """Test VideoActionTrainer class."""

    @pytest.fixture
    def dummy_setup(self):
        """Create dummy setup for testing."""
        # Create dummy model
        model = DummyModel()

        # Create dummy data
        data = torch.randn(20, 6, 8, 32, 32)  # 20 samples
        labels = torch.randint(0, 2, (20,)).float()
        dataset = TensorDataset(data, labels)

        # Create data loaders
        train_loader = DataLoader(dataset[:16], batch_size=4)
        val_loader = DataLoader(dataset[16:], batch_size=4)

        # Other components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        return {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "criterion": criterion,
            "optimizer": optimizer,
            "device": device,
        }

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.barrier")
    def test_trainer_initialization(
        self,
        mock_barrier,
        mock_broadcast,
        mock_all_reduce,
        mock_world_size,
        mock_rank,
        dummy_setup,
    ):
        """Test trainer initialization."""
        mock_rank.return_value = 0
        mock_world_size.return_value = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = VideoActionTrainer(
                model=dummy_setup["model"],
                train_loader=dummy_setup["train_loader"],
                val_loader=dummy_setup["val_loader"],
                criterion=dummy_setup["criterion"],
                optimizer=dummy_setup["optimizer"],
                rank=0,
                world_size=1,
                device=dummy_setup["device"],
                checkpoint_dir=tmpdir,
            )

            assert trainer.rank == 0
            assert trainer.world_size == 1
            assert trainer.start_epoch == 1
            assert trainer.best_val_loss == float("inf")
            assert isinstance(trainer.early_stopping, EarlyStopping)
            assert isinstance(trainer.train_metrics, MetricTracker)
            assert isinstance(trainer.val_metrics, MetricTracker)

    @patch("torch.distributed.all_reduce")
    def test_compute_metrics(self, mock_all_reduce, dummy_setup):
        """Test _compute_metrics method."""
        mock_all_reduce.side_effect = lambda tensor, op: None  # No-op

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = VideoActionTrainer(
                model=dummy_setup["model"],
                train_loader=dummy_setup["train_loader"],
                val_loader=dummy_setup["val_loader"],
                criterion=dummy_setup["criterion"],
                optimizer=dummy_setup["optimizer"],
                rank=0,
                world_size=1,
                device=dummy_setup["device"],
                checkpoint_dir=tmpdir,
            )

            # Test with perfect predictions
            outputs = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
            labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

            metrics = trainer._compute_metrics(outputs, labels)

            assert "accuracy" in metrics
            assert metrics["accuracy"] == 1.0  # Perfect predictions

    def test_save_checkpoint(self, dummy_setup):
        """Test save_checkpoint method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = VideoActionTrainer(
                model=dummy_setup["model"],
                train_loader=dummy_setup["train_loader"],
                val_loader=dummy_setup["val_loader"],
                criterion=dummy_setup["criterion"],
                optimizer=dummy_setup["optimizer"],
                rank=0,
                world_size=1,
                device=dummy_setup["device"],
                checkpoint_dir=tmpdir,
            )

            # Save checkpoint
            trainer.save_checkpoint(epoch=5, is_best=True)

            # Check files were created
            assert os.path.exists(os.path.join(tmpdir, "checkpoint_epoch_5.pth"))
            assert os.path.exists(os.path.join(tmpdir, "latest_checkpoint.pth"))
            assert os.path.exists(os.path.join(tmpdir, "best_model.pth"))
            assert os.path.exists(os.path.join(tmpdir, "training_history.json"))

    def test_load_checkpoint(self, dummy_setup):
        """Test load_checkpoint method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = VideoActionTrainer(
                model=dummy_setup["model"],
                train_loader=dummy_setup["train_loader"],
                val_loader=dummy_setup["val_loader"],
                criterion=dummy_setup["criterion"],
                optimizer=dummy_setup["optimizer"],
                rank=0,
                world_size=1,
                device=dummy_setup["device"],
                checkpoint_dir=tmpdir,
            )

            # Save a checkpoint first
            trainer.best_val_loss = 0.5
            trainer.save_checkpoint(epoch=3)

            # Reset trainer state
            trainer.start_epoch = 1
            trainer.best_val_loss = float("inf")

            # Load checkpoint
            checkpoint_path = os.path.join(tmpdir, "checkpoint_epoch_3.pth")
            trainer.load_checkpoint(checkpoint_path)

            assert trainer.start_epoch == 4  # epoch + 1
            assert trainer.best_val_loss == 0.5

    def test_runtime_management(self, dummy_setup):
        """Test runtime management features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = VideoActionTrainer(
                model=dummy_setup["model"],
                train_loader=dummy_setup["train_loader"],
                val_loader=dummy_setup["val_loader"],
                criterion=dummy_setup["criterion"],
                optimizer=dummy_setup["optimizer"],
                rank=0,
                world_size=1,
                device=dummy_setup["device"],
                checkpoint_dir=tmpdir,
                max_runtime_hours=0.001,  # Very short runtime
            )

            # Test runtime limit check
            import time

            trainer.training_start_time = time.time() - 10  # 10 seconds ago

            runtime_exceeded = trainer._check_runtime_limit()
            assert runtime_exceeded  # Should exceed 0.001 hours

    def test_elapsed_time_calculation(self, dummy_setup):
        """Test elapsed time calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = VideoActionTrainer(
                model=dummy_setup["model"],
                train_loader=dummy_setup["train_loader"],
                val_loader=dummy_setup["val_loader"],
                criterion=dummy_setup["criterion"],
                optimizer=dummy_setup["optimizer"],
                rank=0,
                world_size=1,
                device=dummy_setup["device"],
                checkpoint_dir=tmpdir,
            )

            # Test with no start time
            elapsed = trainer._get_elapsed_hours()
            assert elapsed == 0.0

            # Test with start time
            import time

            trainer.training_start_time = time.time() - 3600  # 1 hour ago
            elapsed = trainer._get_elapsed_hours()
            assert 0.9 < elapsed < 1.1  # Should be close to 1 hour


if __name__ == "__main__":
    pytest.main([__file__])
